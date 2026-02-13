from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from .naming import NameScope


def _require_static_dim(dim: Any, name: str) -> int:
    assert dim is not None and not isinstance(dim, str), f"{name} must be static; got {dim}"
    return int(dim)


def _tensor_rank(tensor: gs.Tensor) -> int:
    if hasattr(tensor, "shape") and tensor.shape is not None:
        return len(tensor.shape)
    return 0


def _tensor_height(tensor: gs.Tensor, axis: int = 2) -> int:
    assert hasattr(tensor, "shape") and tensor.shape is not None, (
        f"tensor {tensor.name} has no static shape information"
    )
    assert len(tensor.shape) > axis, f"tensor {tensor.name} rank is too small for axis {axis}"
    return _require_static_dim(tensor.shape[axis], f"tensor {tensor.name} height")


def _clone_shape_with_height(
    shape: Optional[Sequence[Any]],
    axis: int,
    height: int,
) -> Optional[List[Any]]:
    if shape is None:
        return None
    if len(shape) <= axis:
        return None
    new_shape = list(shape)
    new_shape[axis] = height
    return new_shape


def _make_constant(name_scope: NameScope, values: np.ndarray) -> gs.Constant:
    return gs.Constant(name_scope.make("tsplit_const"), values)


def _make_slice(
    name_scope: NameScope,
    data: gs.Tensor,
    start: int,
    end: int,
    axis: int,
) -> Tuple[gs.Variable, gs.Node]:
    starts = _make_constant(name_scope, np.array([start], dtype=np.int64))
    ends = _make_constant(name_scope, np.array([end], dtype=np.int64))
    axes = _make_constant(name_scope, np.array([axis], dtype=np.int64))
    steps = _make_constant(name_scope, np.array([1], dtype=np.int64))

    out_shape = None
    if hasattr(data, "shape") and data.shape is not None:
        out_shape = _clone_shape_with_height(data.shape, axis, end - start)
    out = gs.Variable(
        name_scope.make(f"{data.name}_slice"),
        dtype=data.dtype,
        shape=out_shape,
    )
    node = gs.Node(op="Slice", inputs=[data, starts, ends, axes, steps], outputs=[out])
    return out, node


def _make_concat(
    name_scope: NameScope,
    inputs: Sequence[gs.Tensor],
    axis: int,
    shape_hint: Optional[Sequence[Any]] = None,
) -> Tuple[gs.Variable, gs.Node]:
    assert inputs, "concat inputs must be non-empty"
    out_shape = None
    if shape_hint is not None:
        out_shape = list(shape_hint)
    out = gs.Variable(name_scope.make("tsplit_concat"), dtype=inputs[0].dtype, shape=out_shape)
    node = gs.Node(op="Concat", inputs=inputs, outputs=[out], attrs={"axis": axis})
    return out, node


def _make_pad(
    name_scope: NameScope,
    data: gs.Tensor,
    pad_top: int,
    pad_bottom: int,
) -> Tuple[gs.Variable, gs.Node]:
    rank = _tensor_rank(data)
    assert rank == 4, f"Pad expects 4D NCHW tensors; got rank {rank} for {data.name}"

    pads = [0, 0, pad_top, 0, 0, 0, pad_bottom, 0]
    pads_const = _make_constant(name_scope, np.array(pads, dtype=np.int64))
    const_dtype = data.dtype or np.float32
    const_val = _make_constant(name_scope, np.array(0, dtype=const_dtype))

    out_shape = None
    if hasattr(data, "shape") and data.shape is not None:
        out_shape = list(data.shape)
        out_shape[2] = _require_static_dim(out_shape[2], f"{data.name} height") + pad_top + pad_bottom

    out = gs.Variable(name_scope.make(f"{data.name}_pad"), dtype=data.dtype, shape=out_shape)
    node = gs.Node(
        op="Pad",
        inputs=[data, pads_const, const_val],
        outputs=[out],
        attrs={"mode": "constant"},
    )
    return out, node
