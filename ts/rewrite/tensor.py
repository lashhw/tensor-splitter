from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from .naming import NameScope


def _is_constant(tensor: gs.Tensor) -> bool:
    return isinstance(tensor, gs.Constant)


def _tensor_height(tensor: gs.Tensor) -> int:
    return tensor.shape[2]


def _shape_with_dim_size(
    shape: Optional[List[Any]],
    dim: int,
    size: int,
) -> Optional[List[Any]]:
    new_shape = list(shape)
    new_shape[dim] = size
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
    assert hasattr(data, "shape") and data.shape is not None, (
        f"Slice expects static tensor shape for {data.name}"
    )
    starts = _make_constant(name_scope, np.array([start], dtype=np.int64))
    ends = _make_constant(name_scope, np.array([end], dtype=np.int64))
    axes = _make_constant(name_scope, np.array([axis], dtype=np.int64))
    steps = _make_constant(name_scope, np.array([1], dtype=np.int64))

    out_shape = _shape_with_dim_size(data.shape, axis, end - start)
    out = gs.Variable(
        name_scope.make(f"{data.name}_slice"),
        dtype=data.dtype,
        shape=out_shape,
    )
    node = gs.Node(op="Slice", inputs=[data, starts, ends, axes, steps], outputs=[out])
    return out, node


def _make_concat(
    name_scope: NameScope,
    inputs: List[gs.Tensor],
    axis: int,
    shape_hint: Optional[List[Any]] = None,
) -> Tuple[gs.Variable, gs.Node]:
    assert inputs, "concat inputs must be non-empty"
    out_shape = None
    if shape_hint is not None:
        out_shape = list(shape_hint)
    out = gs.Variable(name_scope.make("tsplit_concat"), dtype=inputs[0].dtype, shape=out_shape)
    node = gs.Node(op="Concat", inputs=inputs, outputs=[out], attrs={"axis": axis})
    return out, node
