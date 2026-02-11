from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import onnx_graphsurgeon as gs

from .naming import NameScope


def _get_attr(node: gs.Node, name: str, default=None):
    if node.attrs is None:
        return default
    return node.attrs.get(name, default)


def _as_int_list(value, length: Optional[int] = None) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        out = [int(v) for v in value]
    else:
        out = [int(value)]
    if length is not None and len(out) != length:
        raise RuntimeError(f"expected list of length {length}, got {out}")
    return out


def _require_static_dim(dim, name: str) -> int:
    if dim is None or isinstance(dim, str):
        raise RuntimeError(f"{name} must be static; got {dim}")
    return int(dim)


def _tensor_rank(tensor) -> int:
    if hasattr(tensor, "shape") and tensor.shape is not None:
        return len(tensor.shape)
    return 0


def _tensor_height(tensor, axis: int = 2) -> int:
    if not hasattr(tensor, "shape") or tensor.shape is None:
        raise RuntimeError(f"tensor {tensor.name} has no static shape information")
    if len(tensor.shape) <= axis:
        raise RuntimeError(f"tensor {tensor.name} rank is too small for axis {axis}")
    return _require_static_dim(tensor.shape[axis], f"tensor {tensor.name} height")


def _clone_shape_with_height(shape: Optional[Sequence[int]], axis: int, height: int):
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
    data,
    start: int,
    end: int,
    axis: int,
    nodes: List[gs.Node],
):
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
    nodes.append(node)
    return out


def _make_concat(
    name_scope: NameScope,
    inputs,
    axis: int,
    nodes: List[gs.Node],
    shape_hint=None,
):
    if not inputs:
        raise RuntimeError("concat inputs must be non-empty")
    out_shape = None
    if shape_hint is not None:
        out_shape = list(shape_hint)
    out = gs.Variable(name_scope.make("tsplit_concat"), dtype=inputs[0].dtype, shape=out_shape)
    node = gs.Node(op="Concat", inputs=inputs, outputs=[out], attrs={"axis": axis})
    nodes.append(node)
    return out


def _slice_from_tiles(
    name_scope: NameScope,
    tiles,
    ranges: Sequence[tuple[int, int]],
    start: int,
    end: int,
    axis: int,
    nodes: List[gs.Node],
):
    if start >= end:
        raise RuntimeError(f"invalid slice range [{start},{end})")
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
            piece = _make_slice(name_scope, tile, rel_start, rel_end, axis, nodes)
            pieces.append(piece)

    if not pieces:
        raise RuntimeError(f"slice [{start},{end}) does not overlap any tiles")
    if len(pieces) == 1:
        return pieces[0]

    out_shape = None
    if hasattr(tiles[0], "shape") and tiles[0].shape is not None:
        out_shape = _clone_shape_with_height(tiles[0].shape, axis, end - start)
    return _make_concat(name_scope, pieces, axis, nodes, shape_hint=out_shape)


def _make_pad(
    name_scope: NameScope,
    data,
    pad_top: int,
    pad_bottom: int,
    nodes: List[gs.Node],
):
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
    nodes.append(node)
    return out


def _conv_params(node: gs.Node):
    auto_pad = _get_attr(node, "auto_pad", "NOTSET")
    if auto_pad not in (None, "NOTSET", ""):
        raise RuntimeError(f"Conv auto_pad {auto_pad} is not supported")

    strides = _as_int_list(_get_attr(node, "strides", [1, 1]), length=2)
    dilations = _as_int_list(_get_attr(node, "dilations", [1, 1]), length=2)
    pads = _as_int_list(_get_attr(node, "pads", [0, 0, 0, 0]), length=4)
    kernel_shape = _as_int_list(_get_attr(node, "kernel_shape", None))

    if kernel_shape is None:
        if len(node.inputs) < 2:
            raise RuntimeError("Conv node missing weight input for kernel_shape inference")
        weight = node.inputs[1]
        if not hasattr(weight, "shape") or weight.shape is None:
            raise RuntimeError("Conv weight has no shape for kernel_shape inference")
        if len(weight.shape) < 4:
            raise RuntimeError("Conv weight has invalid shape for kernel_shape inference")
        kernel_shape = [int(weight.shape[-2]), int(weight.shape[-1])]

    if len(kernel_shape) != 2:
        raise RuntimeError(f"Only 2D Conv supported; got kernel_shape {kernel_shape}")

    return kernel_shape, strides, dilations, pads


def _conv_attrs_with_height_pad(node: gs.Node, pads: Sequence[int]):
    attrs = dict(node.attrs) if node.attrs else {}
    attrs["pads"] = pads
    if "auto_pad" in attrs:
        attrs["auto_pad"] = "NOTSET"
    return attrs
