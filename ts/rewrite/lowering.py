from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from .catalog import BINARY_OPS, UNARY_CONST_OPS, UNARY_OPS, TileBlock
from .conv import _conv_attrs_with_height_pad, _conv_params
from .naming import NameScope
from .tensor import _clone_shape_with_height, _make_pad, _make_slice, _slice_from_tiles, _tensor_height
from .tiling import _conv_input_slice_for_output, _conv_output_height, _partition_ranges


def _ensure_supported_op(node: gs.Node) -> None:
    if node.op in UNARY_OPS:
        return
    if node.op in UNARY_CONST_OPS:
        return
    if node.op in BINARY_OPS:
        return
    if node.op == "Conv":
        return
    raise RuntimeError(f"unsupported op {node.op} for tiled rewrite")


def _build_unary_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Variable],
    nodes: List[gs.Node],
) -> Tuple[List[gs.Variable], List[TileBlock]]:
    out_tiles = []
    blocks = []

    for tile_id, tile in enumerate(tiles):
        out_shape = tile.shape if hasattr(tile, "shape") else None
        out = gs.Variable(
            name_scope.make(f"{node.outputs[0].name}_tile{tile_id}"),
            dtype=tile.dtype,
            shape=out_shape,
        )
        new_node = gs.Node(
            op=node.op,
            inputs=[tile],
            outputs=[out],
            attrs=dict(node.attrs) if node.attrs else {},
        )
        nodes.append(new_node)
        out_tiles.append(out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=[new_node]))

    return out_tiles, blocks


def _build_unary_const_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Variable],
    nodes: List[gs.Node],
    main_input_index: int,
) -> Tuple[List[gs.Variable], List[TileBlock]]:
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
        new_node = gs.Node(
            op=node.op,
            inputs=inputs,
            outputs=[out],
            attrs=dict(node.attrs) if node.attrs else {},
        )
        nodes.append(new_node)
        out_tiles.append(out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=[new_node]))

    return out_tiles, blocks


def _build_binary_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Variable],
    ranges: Sequence[Tuple[int, int]],
    nodes: List[gs.Node],
    main_input_index: int,
) -> Tuple[List[gs.Variable], List[TileBlock]]:
    if not tiles:
        return [], []

    main_rank = len(tiles[0].shape) if tiles[0].shape is not None else 0
    split_axis = 2
    full_height = ranges[-1][1]

    def _tile_constant_input(constant: gs.Constant, tile_id: int, start: int, end: int) -> gs.Constant:
        values = np.asarray(constant.values)
        const_rank = values.ndim
        const_axis = split_axis + (const_rank - main_rank)

        if const_axis < 0 or const_axis >= const_rank:
            return constant

        const_dim = values.shape[const_axis]
        if const_dim == 1:
            return constant
        if const_dim != full_height:
            raise RuntimeError(
                f"binary op {node.name or node.op} constant input {constant.name} has split-axis "
                f"dimension {const_dim}, expected 1 or {full_height}"
            )

        slices = [slice(None)] * const_rank
        slices[const_axis] = slice(start, end)
        tile_values = np.ascontiguousarray(values[tuple(slices)])
        return gs.Constant(name_scope.make(f"{constant.name}_tile{tile_id}"), values=tile_values)

    out_tiles = []
    blocks = []

    for tile_id, (tile, (start, end)) in enumerate(zip(tiles, ranges)):
        inputs = list(node.inputs)
        inputs[main_input_index] = tile
        for idx, inp in enumerate(inputs):
            if idx == main_input_index:
                continue
            if not isinstance(inp, gs.Constant):
                raise RuntimeError(
                    f"binary op {node.name or node.op} requires constant external input; got {inp.name}"
                )
            inputs[idx] = _tile_constant_input(inp, tile_id, start, end)

        out_shape = tile.shape if hasattr(tile, "shape") else None
        out = gs.Variable(
            name_scope.make(f"{node.outputs[0].name}_tile{tile_id}"),
            dtype=tile.dtype,
            shape=out_shape,
        )
        new_node = gs.Node(
            op=node.op,
            inputs=inputs,
            outputs=[out],
            attrs=dict(node.attrs) if node.attrs else {},
        )
        nodes.append(new_node)
        out_tiles.append(out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=[new_node]))

    return out_tiles, blocks


def _build_conv_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Variable],
    ranges: Sequence[Tuple[int, int]],
    tile_count: int,
    nodes: List[gs.Node],
) -> Tuple[List[gs.Variable], List[Tuple[int, int]], List[TileBlock]]:
    kernel_shape, strides, dilations, pads = _conv_params(node)
    k_h = kernel_shape[0]
    s_h = strides[0]
    d_h = dilations[0]
    pad_top = pads[0]

    h_in = ranges[-1][1]
    actual_h_in = _tensor_height(node.inputs[0])
    if actual_h_in != h_in:
        raise RuntimeError(
            f"Conv input height mismatch: tiles cover {h_in}, but tensor shape is {actual_h_in}"
        )

    out_height = _tensor_height(node.outputs[0])
    out_ranges = _partition_ranges(out_height, tile_count)

    out_tiles = []
    blocks = []

    for tile_id, (y0, y1) in enumerate(out_ranges):
        block_nodes = []
        slice_info = _conv_input_slice_for_output(
            y0=y0,
            y1=y1,
            stride=s_h,
            dilation=d_h,
            kernel=k_h,
            pad_top=pad_top,
            h_in=h_in,
        )
        sliced, slice_nodes = _slice_from_tiles(
            name_scope,
            tiles,
            ranges,
            slice_info.slice_start,
            slice_info.slice_end,
            axis=2,
        )
        block_nodes.extend(slice_nodes)

        padded = sliced
        if slice_info.pad_top or slice_info.pad_bottom:
            padded, pad_node = _make_pad(
                name_scope,
                sliced,
                pad_top=slice_info.pad_top,
                pad_bottom=slice_info.pad_bottom,
            )
            block_nodes.append(pad_node)

        new_pads = [0, pads[1], 0, pads[3]]
        attrs = _conv_attrs_with_height_pad(node, new_pads)
        conv_inputs = list(node.inputs)
        conv_inputs[0] = padded

        expected = _conv_output_height(
            slice_info.slice_end - slice_info.slice_start,
            k_h,
            s_h,
            d_h,
            slice_info.pad_top,
            slice_info.pad_bottom,
        )

        out_shape = None
        if node.outputs[0].shape is not None:
            out_shape = _clone_shape_with_height(node.outputs[0].shape, 2, expected)
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
            conv_out, conv_trim_node = _make_slice(name_scope, conv_out, 0, y1 - y0, 2)
            block_nodes.append(conv_trim_node)

        nodes.extend(block_nodes)
        out_tiles.append(conv_out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=block_nodes))

    return out_tiles, out_ranges, blocks


def _build_tiled_op(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Variable],
    ranges: Sequence[Tuple[int, int]],
    tile_count: int,
    nodes: List[gs.Node],
    main_idx: int,
) -> Tuple[List[gs.Variable], Sequence[Tuple[int, int]], List[TileBlock]]:
    if node.op == "Conv":
        return _build_conv_tiles(name_scope, node, orig_index, tiles, ranges, tile_count, nodes)
    if node.op in UNARY_OPS:
        next_tiles, blocks = _build_unary_tiles(name_scope, node, orig_index, tiles, nodes)
        return next_tiles, ranges, blocks
    if node.op in UNARY_CONST_OPS:
        next_tiles, blocks = _build_unary_const_tiles(
            name_scope,
            node,
            orig_index,
            tiles,
            nodes,
            main_idx,
        )
        return next_tiles, ranges, blocks
    if node.op in BINARY_OPS:
        next_tiles, blocks = _build_binary_tiles(
            name_scope,
            node,
            orig_index,
            tiles,
            ranges,
            nodes,
            main_idx,
        )
        return next_tiles, ranges, blocks
    raise RuntimeError(f"unsupported op {node.op}")
