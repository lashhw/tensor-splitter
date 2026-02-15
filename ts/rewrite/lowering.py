import onnx_graphsurgeon as gs

from .conv import _build_conv_tiles


def _node_base_name(node):
    if node.name:
        return node.name
    if node.outputs and node.outputs[0].name:
        return node.outputs[0].name
    return node.op


def _build_non_conv_tiles(
    node,
    tiles,
    main_input_index,
):
    out_tiles = []
    new_nodes = []

    for tile_id, tile in enumerate(tiles):
        inputs = list(node.inputs)
        inputs[main_input_index] = tile

        out = gs.Variable(
            f"{node.outputs[0].name}_split{tile_id}",
            dtype=node.outputs[0].dtype,
            shape=tile.shape,
        )
        out_tiles.append(out)

        new_node = gs.Node(
            name=f"{_node_base_name(node)}_split{tile_id}",
            op=node.op,
            inputs=inputs,
            outputs=[out],
            attrs=dict(node.attrs)
        )
        new_nodes.append(new_node)

    return out_tiles, new_nodes


def _build_tiled_op(
    node,
    tiles,
    out_ranges,
    main_idx,
    conv_slices=None,
    conv_base_pads=None,
):
    if node.op == "Conv":
        return _build_conv_tiles(node, tiles, out_ranges, conv_slices, conv_base_pads, main_idx)
    else:
        return _build_non_conv_tiles(node, tiles, main_idx)
