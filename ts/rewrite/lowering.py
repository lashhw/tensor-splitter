from __future__ import annotations

from typing import List, Optional

import onnx_graphsurgeon as gs

from .conv import conv_attrs_with_height_pad
from .naming import NameScope
from .tensor import clone_shape_with_height
from .types import ConvSlice, HeightRange, TileBlock, TiledOpBuild


def _build_conv_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: List[gs.Variable],
    out_ranges: List[HeightRange],
    conv_slices: List[ConvSlice],
    conv_base_pads: List[int],
    main_input_index: int,
) -> TiledOpBuild:
    out_tiles = []
    new_nodes = []
    blocks = []

    assert len(tiles) == len(out_ranges) == len(conv_slices), (
        "Conv tiling requires aligned tiles, output ranges, and Conv slice metadata"
    )

    for tile_id, (tile, (y0, y1), slice_info) in enumerate(zip(tiles, out_ranges, conv_slices)):
        new_pads = [slice_info.pad_top, conv_base_pads[1], slice_info.pad_bottom, conv_base_pads[3]]
        attrs = conv_attrs_with_height_pad(node, new_pads)
        conv_inputs = list(node.inputs)
        conv_inputs[main_input_index] = tile

        out_shape = clone_shape_with_height(node.outputs[0].shape, 2, y1 - y0)
        conv_out = gs.Variable(
            name_scope.make(f"{node.outputs[0].name}_tile{tile_id}"),
            dtype=node.outputs[0].dtype,
            shape=out_shape,
        )
        out_tiles.append(conv_out)

        conv_node = gs.Node(
            op="Conv",
            inputs=conv_inputs,
            outputs=[conv_out],
            attrs=attrs
        )
        new_nodes.append(conv_node)

        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, node=conv_node))

    return TiledOpBuild(output_tiles=out_tiles, nodes=new_nodes, blocks=blocks)


def _build_non_conv_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: List[gs.Variable],
    main_input_index: int,
) -> TiledOpBuild:
    out_tiles = []
    new_nodes = []
    blocks = []

    for tile_id, tile in enumerate(tiles):
        inputs = list(node.inputs)
        inputs[main_input_index] = tile

        out = gs.Variable(
            name_scope.make(f"{node.outputs[0].name}_tile{tile_id}"),
            dtype=node.outputs[0].dtype,
            shape=tile.shape,
        )
        out_tiles.append(out)

        new_node = gs.Node(
            op=node.op,
            inputs=inputs,
            outputs=[out],
            attrs=dict(node.attrs)
        )
        new_nodes.append(new_node)

        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, node=new_node))

    return TiledOpBuild(output_tiles=out_tiles, nodes=new_nodes, blocks=blocks)


def build_tiled_op(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: List[gs.Variable],
    out_ranges: List[HeightRange],
    main_idx: int,
    conv_slices: Optional[List[ConvSlice]] = None,
    conv_base_pads: Optional[List[int]] = None,
) -> TiledOpBuild:
    if node.op == "Conv":
        assert conv_slices is not None, "Conv nodes require Conv slice metadata"
        assert conv_base_pads is not None, "Conv nodes require base pads"
        return _build_conv_tiles(
            name_scope,
            node,
            orig_index,
            tiles,
            out_ranges,
            conv_slices,
            conv_base_pads,
            main_idx,
        )
    return _build_non_conv_tiles(name_scope, node, orig_index, tiles, main_idx)
