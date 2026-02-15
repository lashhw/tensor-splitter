from __future__ import annotations

from typing import List, Optional, Tuple

import onnx_graphsurgeon as gs

from .catalog import SUPPORTED_GROUP_OPS, TileBlock
from .conv import _conv_attrs_with_height_pad
from .naming import NameScope
from .tensor import _clone_shape_with_height
from .tiling import ConvInputSlice


def _ensure_supported_op(node: gs.Node) -> None:
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op {node.op} for tiled rewrite"


def _build_conv_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: List[gs.Variable],
    out_ranges: List[Tuple[int, int]],
    conv_slices: List[ConvInputSlice],
    conv_base_pads: List[int],
    main_input_index: int,
) -> Tuple[List[gs.Variable], List[gs.Node], List[TileBlock]]:
    pads = conv_base_pads
    assert len(conv_slices) == len(out_ranges), (
        f"conv_slices length {len(conv_slices)} must match out_ranges length {len(out_ranges)}"
    )
    assert len(pads) == 4, f"conv_base_pads must have length 4; got {pads}"

    out_tiles = []
    new_nodes = []
    blocks = []

    for tile_id, (tile, (y0, y1), slice_info) in enumerate(zip(tiles, out_ranges, conv_slices)):
        block_nodes = []

        new_pads = [slice_info.pad_top, pads[1], slice_info.pad_bottom, pads[3]]
        attrs = _conv_attrs_with_height_pad(node, new_pads)
        conv_inputs = list(node.inputs)
        conv_inputs[main_input_index] = tile

        expected = y1 - y0

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

        out_tiles.append(conv_out)
        new_nodes.extend(block_nodes)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=block_nodes))

    return out_tiles, new_nodes, blocks


def _build_non_conv_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: List[gs.Variable],
    main_input_index: int,
) -> Tuple[List[gs.Variable], List[gs.Node], List[TileBlock]]:
    out_tiles = []
    new_nodes: List[gs.Node] = []
    blocks = []

    for idx, inp in enumerate(node.inputs):
        if idx == main_input_index:
            continue
        assert isinstance(inp, gs.Constant), (
            f"node {node.name or node.op} has unsupported external variable input {inp.name}"
        )

    for tile_id, tile in enumerate(tiles):
        inputs = list(node.inputs)
        inputs[main_input_index] = tile
        assert tile.shape is not None, f"node {node.name or node.op} tile {tile_id} must have static shape"
        out_shape = tile.shape
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
        new_nodes.append(new_node)
        out_tiles.append(out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=[new_node]))

    return out_tiles, new_nodes, blocks


def _build_tiled_op(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: List[gs.Variable],
    out_ranges: List[Tuple[int, int]],
    main_idx: int,
    conv_slices: Optional[List[ConvInputSlice]] = None,
    conv_base_pads: Optional[List[int]] = None,
) -> Tuple[List[gs.Variable], List[gs.Node], List[TileBlock]]:
    if node.op == "Conv":
        return _build_conv_tiles(name_scope, node, orig_index, tiles, out_ranges, conv_slices, conv_base_pads, main_idx)
    else:
        return _build_non_conv_tiles(name_scope, node, orig_index, tiles, main_idx)
