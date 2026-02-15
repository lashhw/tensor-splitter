import onnx_graphsurgeon as gs

from .catalog import TileBlock
from .conv import _conv_attrs_with_height_pad
from .tensor import _shape_with_dim_size

def _build_conv_tiles(
    name_scope,
    node,
    orig_index,
    tiles,
    out_ranges,
    conv_slices,
    conv_base_pads,
    main_input_index,
):
    out_tiles = []
    new_nodes = []
    blocks = []

    for tile_id, (tile, (y0, y1), slice_info) in enumerate(zip(tiles, out_ranges, conv_slices)):
        new_pads = [slice_info.pad_top, conv_base_pads[1], slice_info.pad_bottom, conv_base_pads[3]]
        attrs = _conv_attrs_with_height_pad(node, new_pads)
        conv_inputs = list(node.inputs)
        conv_inputs[main_input_index] = tile

        out_shape = _shape_with_dim_size(node.outputs[0].shape, 2, y1 - y0)
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

    return out_tiles, new_nodes, blocks


def _build_non_conv_tiles(
    name_scope,
    node,
    orig_index,
    tiles,
    main_input_index,
):
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

    return out_tiles, new_nodes, blocks


def _build_tiled_op(
    name_scope,
    node,
    orig_index,
    tiles,
    out_ranges,
    main_idx,
    conv_slices=None,
    conv_base_pads=None,
):
    if node.op == "Conv":
        return _build_conv_tiles(name_scope, node, orig_index, tiles, out_ranges, conv_slices, conv_base_pads, main_idx)
    else:
        return _build_non_conv_tiles(name_scope, node, orig_index, tiles, main_idx)
