from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from src.tiling import conv_input_slice_for_output, conv_output_height, partition_ranges

HEIGHT_AXIS = 2

UNARY_OPS = {
    "Relu",
    "Sigmoid",
    "Tanh",
    "Identity",
}
UNARY_CONST_OPS = {
    "Clip",
    "BatchNormalization",
}
BINARY_OPS = {"Add", "Mul", "Sub", "Div"}


class NameScope:
    """Generates unique, graph-safe names for newly created tensors."""

    def __init__(self, existing: Sequence[str]):
        self.existing = set(existing)
        self.counter = 0

    def make(self, base: str) -> str:
        base = base.replace(":", "_")
        name = f"{base}_{self.counter}"
        self.counter += 1
        while name in self.existing:
            name = f"{base}_{self.counter}"
            self.counter += 1
        self.existing.add(name)
        return name


@dataclass
class TileBlock:
    """Tracks the nodes that implement a single tile of an original node."""

    orig_index: int
    tile_id: int
    nodes: List[gs.Node]

    def assign_priority(self, priority: Dict[gs.Node, int], order: int) -> None:
        for node in self.nodes:
            priority[node] = order


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


def _tensor_rank(tensor: gs.Tensor) -> int:
    if hasattr(tensor, "shape") and tensor.shape is not None:
        return len(tensor.shape)
    return 0


def _tensor_height(tensor: gs.Tensor, axis: int = HEIGHT_AXIS) -> int:
    if not hasattr(tensor, "shape") or tensor.shape is None:
        raise RuntimeError(f"tensor {tensor.name} has no static shape information")
    if len(tensor.shape) <= axis:
        raise RuntimeError(f"tensor {tensor.name} rank is too small for axis {axis}")
    return _require_static_dim(tensor.shape[axis], f"tensor {tensor.name} height")


def _clone_shape_with_height(shape, axis: int, height: int):
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
    nodes: List[gs.Node],
) -> gs.Variable:
    starts = _make_constant(name_scope, np.array([start], dtype=np.int64))
    ends = _make_constant(name_scope, np.array([end], dtype=np.int64))
    axes = _make_constant(name_scope, np.array([axis], dtype=np.int64))
    steps = _make_constant(name_scope, np.array([1], dtype=np.int64))

    out_shape = None
    if hasattr(data, "shape") and data.shape is not None:
        out_shape = _clone_shape_with_height(data.shape, axis, end - start)
    out = gs.Variable(name_scope.make(f"{data.name}_slice"), dtype=data.dtype, shape=out_shape)
    node = gs.Node(op="Slice", inputs=[data, starts, ends, axes, steps], outputs=[out])
    nodes.append(node)
    return out


def _make_concat(
    name_scope: NameScope,
    inputs: Sequence[gs.Tensor],
    axis: int,
    nodes: List[gs.Node],
    shape_hint=None,
) -> gs.Variable:
    if not inputs:
        raise RuntimeError("concat inputs must be non-empty")
    out_shape = None
    if shape_hint is not None:
        out_shape = list(shape_hint)
    out = gs.Variable(name_scope.make("tsplit_concat"), dtype=inputs[0].dtype, shape=out_shape)
    node = gs.Node(op="Concat", inputs=list(inputs), outputs=[out], attrs={"axis": axis})
    nodes.append(node)
    return out


def _slice_from_tiles(
    name_scope: NameScope,
    tiles: Sequence[gs.Tensor],
    ranges: Sequence[Tuple[int, int]],
    start: int,
    end: int,
    axis: int,
    nodes: List[gs.Node],
) -> gs.Tensor:
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
    data: gs.Tensor,
    pad_top: int,
    pad_bottom: int,
    nodes: List[gs.Node],
) -> gs.Variable:
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
        out_shape[HEIGHT_AXIS] = (
            _require_static_dim(out_shape[HEIGHT_AXIS], f"{data.name} height")
            + pad_top
            + pad_bottom
        )

    out = gs.Variable(name_scope.make(f"{data.name}_pad"), dtype=data.dtype, shape=out_shape)
    node = gs.Node(
        op="Pad",
        inputs=[data, pads_const, const_val],
        outputs=[out],
        attrs={"mode": "constant"},
    )
    nodes.append(node)
    return out


def _conv_params(node: gs.Node) -> Tuple[List[int], List[int], List[int], List[int]]:
    auto_pad = _get_attr(node, "auto_pad", "NOTSET")
    if auto_pad not in (None, "NOTSET", ""):
        raise RuntimeError(f"Conv auto_pad {auto_pad} not supported in v1")

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


def _conv_attrs_with_height_pad(node: gs.Node, pads: List[int]):
    attrs = dict(node.attrs) if node.attrs else {}
    attrs["pads"] = pads
    if "auto_pad" in attrs:
        attrs["auto_pad"] = "NOTSET"
    return attrs


def _ensure_supported_op(node: gs.Node) -> None:
    if node.op in UNARY_OPS:
        return
    if node.op in UNARY_CONST_OPS:
        return
    if node.op in BINARY_OPS:
        return
    if node.op == "Conv":
        return
    raise RuntimeError(f"unsupported op {node.op} for v1 tiling")


def _build_unary_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Tensor],
    nodes: List[gs.Node],
) -> Tuple[List[gs.Tensor], List[TileBlock]]:
    out_tiles = []
    blocks = []

    for tile_id, tile in enumerate(tiles):
        out_shape = tile.shape if hasattr(tile, "shape") else None
        out = gs.Variable(
            name_scope.make(f"{node.outputs[0].name}_tile{tile_id}"),
            dtype=tile.dtype,
            shape=out_shape,
        )
        new_node = gs.Node(op=node.op, inputs=[tile], outputs=[out], attrs=dict(node.attrs) if node.attrs else {})
        nodes.append(new_node)
        out_tiles.append(out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=[new_node]))

    return out_tiles, blocks


def _build_unary_const_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Tensor],
    nodes: List[gs.Node],
    main_input_index: int,
) -> Tuple[List[gs.Tensor], List[TileBlock]]:
    out_tiles = []
    blocks = []

    for tile_id, tile in enumerate(tiles):
        inputs = list(node.inputs)
        inputs[main_input_index] = tile
        for idx, inp in enumerate(inputs):
            if idx == main_input_index:
                continue
            if not isinstance(inp, gs.Constant):
                raise RuntimeError(
                    f"node {node.name or node.op} has unsupported external variable input {inp.name}"
                )
        out_shape = tile.shape if hasattr(tile, "shape") else None
        out = gs.Variable(
            name_scope.make(f"{node.outputs[0].name}_tile{tile_id}"),
            dtype=tile.dtype,
            shape=out_shape,
        )
        new_node = gs.Node(op=node.op, inputs=inputs, outputs=[out], attrs=dict(node.attrs) if node.attrs else {})
        nodes.append(new_node)
        out_tiles.append(out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=[new_node]))

    return out_tiles, blocks


def _build_binary_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Tensor],
    nodes: List[gs.Node],
    main_input_index: int,
) -> Tuple[List[gs.Tensor], List[TileBlock]]:
    out_tiles = []
    blocks = []

    for tile_id, tile in enumerate(tiles):
        inputs = list(node.inputs)
        inputs[main_input_index] = tile
        for idx, inp in enumerate(inputs):
            if idx == main_input_index:
                continue
            if not isinstance(inp, gs.Constant):
                raise RuntimeError(
                    f"binary op {node.name or node.op} requires constant external input; got {inp.name}"
                )
        out_shape = tile.shape if hasattr(tile, "shape") else None
        out = gs.Variable(
            name_scope.make(f"{node.outputs[0].name}_tile{tile_id}"),
            dtype=tile.dtype,
            shape=out_shape,
        )
        new_node = gs.Node(op=node.op, inputs=inputs, outputs=[out], attrs=dict(node.attrs) if node.attrs else {})
        nodes.append(new_node)
        out_tiles.append(out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=[new_node]))

    return out_tiles, blocks


def _build_conv_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Tensor],
    ranges: Sequence[Tuple[int, int]],
    splits: int,
    nodes: List[gs.Node],
) -> Tuple[List[gs.Tensor], List[Tuple[int, int]], List[TileBlock]]:
    kernel_shape, strides, dilations, pads = _conv_params(node)
    k_h = kernel_shape[0]
    s_h = strides[0]
    d_h = dilations[0]
    pad_top = pads[0]
    pad_bottom = pads[2]

    h_in = ranges[-1][1]
    actual_h_in = _tensor_height(node.inputs[0])
    if actual_h_in != h_in:
        raise RuntimeError(
            f"Conv input height mismatch: tiles cover {h_in}, but tensor shape is {actual_h_in}"
        )
    out_height = _tensor_height(node.outputs[0])
    out_ranges = partition_ranges(out_height, splits)

    out_tiles = []
    blocks = []

    for tile_id, (y0, y1) in enumerate(out_ranges):
        block_nodes: List[gs.Node] = []
        slice_info = conv_input_slice_for_output(
            y0=y0,
            y1=y1,
            stride=s_h,
            dilation=d_h,
            kernel=k_h,
            pad_top=pad_top,
            h_in=h_in,
        )
        sliced = _slice_from_tiles(
            name_scope,
            tiles,
            ranges,
            slice_info.slice_start,
            slice_info.slice_end,
            axis=HEIGHT_AXIS,
            nodes=block_nodes,
        )

        padded = sliced
        if slice_info.pad_top or slice_info.pad_bottom:
            padded = _make_pad(
                name_scope,
                sliced,
                pad_top=slice_info.pad_top,
                pad_bottom=slice_info.pad_bottom,
                nodes=block_nodes,
            )

        new_pads = [0, pads[1], 0, pads[3]]
        attrs = _conv_attrs_with_height_pad(node, new_pads)
        conv_inputs = list(node.inputs)
        conv_inputs[0] = padded

        expected = conv_output_height(
            slice_info.slice_end - slice_info.slice_start,
            k_h,
            s_h,
            d_h,
            slice_info.pad_top,
            slice_info.pad_bottom,
        )

        out_shape = None
        if node.outputs[0].shape is not None:
            out_shape = _clone_shape_with_height(node.outputs[0].shape, HEIGHT_AXIS, expected)
        conv_out = gs.Variable(
            name_scope.make(f"{node.outputs[0].name}_tile{tile_id}"),
            dtype=node.outputs[0].dtype,
            shape=out_shape,
        )
        conv_node = gs.Node(op="Conv", inputs=conv_inputs, outputs=[conv_out], attrs=attrs)
        block_nodes.append(conv_node)

        if expected != (y1 - y0):
            if expected < (y1 - y0):
                raise RuntimeError(
                    f"Conv tile output shorter than expected: expected {y1 - y0}, got {expected}"
                )
            conv_out = _make_slice(name_scope, conv_out, 0, y1 - y0, HEIGHT_AXIS, block_nodes)

        nodes.extend(block_nodes)
        out_tiles.append(conv_out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=block_nodes))

    return out_tiles, out_ranges, blocks


def _build_entry_tiles(
    name_scope: NameScope,
    entry: gs.Tensor,
    splits: int,
    nodes: List[gs.Node],
) -> Tuple[List[gs.Tensor], List[Tuple[int, int]]]:
    h_in = _tensor_height(entry)
    ranges = partition_ranges(h_in, splits)
    tiles = []
    for start, end in ranges:
        tile = _make_slice(name_scope, entry, start, end, HEIGHT_AXIS, nodes)
        tiles.append(tile)
    return tiles, ranges

