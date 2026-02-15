from __future__ import annotations

from typing import List, Sequence, Tuple

import onnx_graphsurgeon as gs

from .catalog import SUPPORTED_GROUP_OPS, TileBlock
from .conv import _conv_attrs_with_height_pad, _conv_params
from .naming import NameScope
from .tensor import _clone_shape_with_height, _make_pad, _make_slice, _tensor_height
from .tiling import _conv_input_slice_for_output, _conv_output_height


def _ensure_supported_op(node: gs.Node) -> None:
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op {node.op} for tiled rewrite"


def _build_conv_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Variable],
    in_ranges: Sequence[Tuple[int, int]],
    out_ranges: Sequence[Tuple[int, int]],
    main_input_index: int,
) -> Tuple[List[gs.Variable], List[gs.Node], List[TileBlock]]:
    kernel_shape, strides, dilations, pads = _conv_params(node)
    k_h = kernel_shape[0]
    s_h = strides[0]
    d_h = dilations[0]
    pad_top = pads[0]

    h_in = _tensor_height(node.inputs[main_input_index])
    assert len(tiles) == len(in_ranges) and len(tiles) == len(out_ranges), (
        f"Conv tile/range mismatch: tiles={len(tiles)}, in_ranges={len(in_ranges)}, "
        f"out_ranges={len(out_ranges)}"
    )

    out_tiles = []
    new_nodes: List[gs.Node] = []
    blocks = []

    for tile_id, (tile, (in_start, in_end), (y0, y1)) in enumerate(zip(tiles, in_ranges, out_ranges)):
        block_nodes = []
        tile_h = _tensor_height(tile)
        expected_tile_h = in_end - in_start
        assert tile_h == expected_tile_h, (
            f"Conv tile {tile_id} height mismatch: tensor has {tile_h}, range requires {expected_tile_h}"
        )
        slice_info = _conv_input_slice_for_output(
            y0=y0,
            y1=y1,
            stride=s_h,
            dilation=d_h,
            kernel=k_h,
            pad_top=pad_top,
            h_in=h_in,
        )
        expected_in_range = (slice_info.slice_start, slice_info.slice_end)
        assert expected_in_range == (in_start, in_end), (
            f"planned Conv input range mismatch for tile {tile_id}: planned [{in_start},{in_end}), "
            f"required [{expected_in_range[0]},{expected_in_range[1]})"
        )

        padded = tile
        if slice_info.pad_top or slice_info.pad_bottom:
            padded, pad_node = _make_pad(
                name_scope,
                tile,
                pad_top=slice_info.pad_top,
                pad_bottom=slice_info.pad_bottom,
            )
            block_nodes.append(pad_node)

        new_pads = [0, pads[1], 0, pads[3]]
        attrs = _conv_attrs_with_height_pad(node, new_pads)
        conv_inputs = list(node.inputs)
        conv_inputs[main_input_index] = padded

        expected = _conv_output_height(
            in_end - in_start,
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
            assert expected >= (y1 - y0), (
                f"Conv tile output shorter than expected: expected {y1 - y0}, got {expected}"
            )
            conv_out, conv_trim_node = _make_slice(name_scope, conv_out, 0, y1 - y0, 2)
            block_nodes.append(conv_trim_node)

        new_nodes.extend(block_nodes)
        out_tiles.append(conv_out)
        blocks.append(TileBlock(orig_index=orig_index, tile_id=tile_id, nodes=block_nodes))

    return out_tiles, new_nodes, blocks


def _build_non_conv_tiles(
    name_scope: NameScope,
    node: gs.Node,
    orig_index: int,
    tiles: Sequence[gs.Variable],
    main_input_index: int,
) -> Tuple[List[gs.Variable], List[gs.Node], List[TileBlock]]:
    out_tiles = []
    new_nodes: List[gs.Node] = []
    blocks = []

    for tile_id, tile in enumerate(tiles):
        inputs = list(node.inputs)
        inputs[main_input_index] = tile
        for idx, inp in enumerate(inputs):
            if idx == main_input_index:
                continue
            assert isinstance(inp, gs.Constant), (
                f"node {node.name or node.op} has unsupported external variable input {inp.name}"
            )
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
    tiles: Sequence[gs.Variable],
    in_ranges: Sequence[Tuple[int, int]],
    out_ranges: Sequence[Tuple[int, int]],
    main_idx: int,
) -> Tuple[List[gs.Variable], List[gs.Node], List[TileBlock]]:
    if node.op == "Conv":
        return _build_conv_tiles(name_scope, node, orig_index, tiles, in_ranges, out_ranges, main_idx)
    else:
        return _build_non_conv_tiles(name_scope, node, orig_index, tiles, main_idx)
