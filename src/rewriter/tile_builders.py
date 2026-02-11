from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import onnx_graphsurgeon as gs

from ..tiling import conv_input_slice_for_output, conv_output_height, partition_ranges
from .tensor_ops import (
    _clone_shape_with_height,
    _conv_attrs_with_height_pad,
    _conv_params,
    _make_concat,
    _make_pad,
    _make_slice,
    _slice_from_tiles,
    _tensor_height,
)
from .types import BINARY_OPS, UNARY_CONST_OPS, UNARY_OPS, NameScope, TileBlock


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
    tiles,
    nodes: List[gs.Node],
):
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
    tiles,
    nodes: List[gs.Node],
    main_input_index: int,
):
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
    tiles,
    nodes: List[gs.Node],
    main_input_index: int,
):
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
    tiles,
    ranges: Sequence[Tuple[int, int]],
    splits: int,
    nodes: List[gs.Node],
):
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
    out_ranges = partition_ranges(out_height, splits)

    out_tiles = []
    blocks = []

    for tile_id, (y0, y1) in enumerate(out_ranges):
        block_nodes = []
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
            axis=2,
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
            conv_out = _make_slice(name_scope, conv_out, 0, y1 - y0, 2, block_nodes)

        nodes.extend(block_nodes)
        out_tiles.append(conv_out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=block_nodes))

    return out_tiles, out_ranges, blocks


def _build_entry_tiles(
    name_scope: NameScope,
    entry,
    splits: int,
    nodes: List[gs.Node],
):
    h_in = _tensor_height(entry)
    ranges = partition_ranges(h_in, splits)
    tiles = []
    for start, end in ranges:
        tile = _make_slice(name_scope, entry, start, end, 2, nodes)
        tiles.append(tile)
    return tiles, ranges


def _apply_schedule_priority(blocks: Sequence[TileBlock], schedule: Sequence[Tuple[int, int]]) -> Dict:
    schedule_pos = {pair: idx for idx, pair in enumerate(schedule)}
    priority = {}
    for block in blocks:
        order = schedule_pos.get((block.orig_index, block.tile_id))
        if order is None:
            continue
        block.assign_priority(priority, order)
    return priority


def _build_tiled_op(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles,
    ranges,
    splits: int,
    nodes: List[gs.Node],
    main_idx: int,
):
    if node.op == "Conv":
        return _build_conv_tiles(name_scope, node, orig_index, tiles, ranges, splits, nodes)
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
            nodes,
            main_idx,
        )
        return next_tiles, ranges, blocks
    raise RuntimeError(f"unsupported op {node.op}")


def _build_group_tiles(
    name_scope: NameScope,
    group_info,
    group_cfg,
    node_index_map,
):
    nodes = []
    blocks = []

    for node in group_info.nodes:
        _ensure_supported_op(node)

    tiles, ranges = _build_entry_tiles(name_scope, group_info.entry_tensor, group_cfg.tile_count, nodes)

    for node in group_info.nodes:
        orig_index = node_index_map[id(node)]
        main_idx = group_info.main_input_index[id(node)]
        tiles, ranges, op_blocks = _build_tiled_op(
            name_scope,
            node,
            orig_index,
            tiles,
            ranges,
            group_cfg.tile_count,
            nodes,
            main_idx,
        )
        blocks.extend(op_blocks)

    return tiles, nodes, blocks


def _build_group_output(name_scope: NameScope, tiles, shape_hint, nodes: List[gs.Node]):
    return _make_concat(name_scope, tiles, axis=2, nodes=nodes, shape_hint=shape_hint)
