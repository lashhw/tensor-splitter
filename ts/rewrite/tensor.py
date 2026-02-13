from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from .naming import NameScope


def _require_static_dim(dim: Any, name: str) -> int:
    if dim is None or isinstance(dim, str):
        raise RuntimeError(f"{name} must be static; got {dim}")
    return int(dim)


def _tensor_rank(tensor: gs.Tensor) -> int:
    if hasattr(tensor, "shape") and tensor.shape is not None:
        return len(tensor.shape)
    return 0


def _tensor_height(tensor: gs.Tensor, axis: int = 2) -> int:
    if not hasattr(tensor, "shape") or tensor.shape is None:
        raise RuntimeError(f"tensor {tensor.name} has no static shape information")
    if len(tensor.shape) <= axis:
        raise RuntimeError(f"tensor {tensor.name} rank is too small for axis {axis}")
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
    if not inputs:
        raise RuntimeError("concat inputs must be non-empty")
    out_shape = None
    if shape_hint is not None:
        out_shape = list(shape_hint)
    out = gs.Variable(name_scope.make("tsplit_concat"), dtype=inputs[0].dtype, shape=out_shape)
    node = gs.Node(op="Concat", inputs=inputs, outputs=[out], attrs={"axis": axis})
    return out, node


def _slice_from_tiles(
    name_scope: NameScope,
    tiles: Sequence[gs.Variable],
    ranges: Sequence[tuple[int, int]],
    start: int,
    end: int,
    axis: int,
) -> Tuple[gs.Variable, List[gs.Node]]:
    if start >= end:
        raise RuntimeError(f"invalid slice range [{start},{end})")
    created_nodes = []
    pieces = []
    for tile, (s, e) in zip(tiles, ranges):
        overlap_start = max(s, start)
        overlap_end = min(e, end)
        if overlap_start >= overlap_end:
            continue
        rel_start = overlap_start - s
        rel_end = overlap_end - s
        if rel_start == 0 and rel_end == (e - s):
            pieces.append(tile)
        else:
            piece, piece_node = _make_slice(name_scope, tile, rel_start, rel_end, axis)
            created_nodes.append(piece_node)
            pieces.append(piece)

    if not pieces:
        raise RuntimeError(f"slice [{start},{end}) does not overlap any tiles")
    if len(pieces) == 1:
        return pieces[0], created_nodes

    out_shape = None
    if hasattr(tiles[0], "shape") and tiles[0].shape is not None:
        out_shape = _clone_shape_with_height(tiles[0].shape, axis, end - start)
    concat_out, concat_node = _make_concat(name_scope, pieces, axis, shape_hint=out_shape)
    created_nodes.append(concat_node)
    return concat_out, created_nodes


def _make_pad(
    name_scope: NameScope,
    data: gs.Tensor,
    pad_top: int,
    pad_bottom: int,
) -> Tuple[gs.Variable, gs.Node]:
    rank = _tensor_rank(data)
    if rank != 4:
        raise RuntimeError(f"Pad expects 4D NCHW tensors; got rank {rank} for {data.name}")

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
