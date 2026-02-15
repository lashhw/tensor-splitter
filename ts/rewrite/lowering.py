import numpy as np
import onnx_graphsurgeon as gs

from .conv import _conv_attrs_with_spatial_pad
from .tensor import _shape_with_dim_size


def _node_base_name(node):
    if node.name:
        return node.name
    if node.outputs and node.outputs[0].name:
        return node.outputs[0].name
    return node.op


def _make_constant(name, values):
    return gs.Constant(name, values)


def _split_suffix(split_h, split_w):
    return f"{split_h}_{split_w}"


def _build_entry_tiles(entry_tensor, entry_regions, tile_ids):
    tiles = []
    slice_nodes = []

    for (split_h, split_w), (h0, h1, w0, w1) in zip(tile_ids, entry_regions):
        suffix = _split_suffix(split_h, split_w)
        base_name = f"{entry_tensor.name}_slice{suffix}"
        starts = _make_constant(f"{base_name}_starts", np.array([h0, w0], dtype=np.int64))
        ends = _make_constant(f"{base_name}_ends", np.array([h1, w1], dtype=np.int64))
        axes = _make_constant(f"{base_name}_axes", np.array([2, 3], dtype=np.int64))
        steps = _make_constant(f"{base_name}_steps", np.array([1, 1], dtype=np.int64))
        out_shape = _shape_with_dim_size(entry_tensor.shape, 2, h1 - h0)
        out_shape = _shape_with_dim_size(out_shape, 3, w1 - w0)
        out = gs.Variable(
            f"{entry_tensor.name}_split{suffix}",
            dtype=entry_tensor.dtype,
            shape=out_shape,
        )
        slice_node = gs.Node(
            name=base_name,
            op="Slice",
            inputs=[entry_tensor, starts, ends, axes, steps],
            outputs=[out],
        )

        tiles.append(out)
        slice_nodes.append(slice_node)

    return tiles, slice_nodes


def _build_conv_tiles(
    node,
    tiles,
    tile_ids,
    out_regions,
    conv_slices,
    main_input_index,
):
    out_tiles = []
    new_nodes = []
    node_base_name = _node_base_name(node)

    for (split_h, split_w), tile, (y0, y1, x0, x1), slice_info in zip(
        tile_ids,
        tiles,
        out_regions,
        conv_slices,
    ):
        suffix = _split_suffix(split_h, split_w)
        attrs = _conv_attrs_with_spatial_pad(node, slice_info)
        conv_inputs = list(node.inputs)
        conv_inputs[main_input_index] = tile

        out_shape = _shape_with_dim_size(node.outputs[0].shape, 2, y1 - y0)
        out_shape = _shape_with_dim_size(out_shape, 3, x1 - x0)
        conv_out = gs.Variable(
            f"{node.outputs[0].name}_split{suffix}",
            dtype=node.outputs[0].dtype,
            shape=out_shape,
        )
        out_tiles.append(conv_out)

        conv_node = gs.Node(
            name=f"{node_base_name}_split{suffix}",
            op="Conv",
            inputs=conv_inputs,
            outputs=[conv_out],
            attrs=attrs,
        )
        new_nodes.append(conv_node)

    return out_tiles, new_nodes


def _build_non_conv_tiles(
    node,
    tiles,
    tile_ids,
    main_input_index,
):
    out_tiles = []
    new_nodes = []

    for (split_h, split_w), tile in zip(tile_ids, tiles):
        suffix = _split_suffix(split_h, split_w)
        inputs = list(node.inputs)
        inputs[main_input_index] = tile

        out = gs.Variable(
            f"{node.outputs[0].name}_split{suffix}",
            dtype=node.outputs[0].dtype,
            shape=tile.shape,
        )
        out_tiles.append(out)

        new_node = gs.Node(
            name=f"{_node_base_name(node)}_split{suffix}",
            op=node.op,
            inputs=inputs,
            outputs=[out],
            attrs=dict(node.attrs) if node.attrs else {},
        )
        new_nodes.append(new_node)

    return out_tiles, new_nodes


def _build_stage_tiles(
    node,
    tiles,
    tile_ids,
    out_regions,
    main_input_index,
    conv_slices=None,
):
    if node.op == "Conv":
        assert conv_slices is not None, "conv_slices are required for Conv lowering"
        return _build_conv_tiles(
            node,
            tiles,
            tile_ids,
            out_regions,
            conv_slices,
            main_input_index,
        )
    return _build_non_conv_tiles(node, tiles, tile_ids, main_input_index)


def _build_group_concat(tiles, tile_count, output_tensor):
    if type(tile_count) is int:
        tile_count = (tile_count, 1)
    tiles_h, tiles_w = tile_count
    extra_nodes = []
    out = gs.Variable(
        output_tensor.name,
        dtype=output_tensor.dtype,
        shape=list(output_tensor.shape),
    )

    if tiles_w == 1:
        node = gs.Node(
            name=f"{output_tensor.name}_concat",
            op="Concat",
            inputs=tiles,
            outputs=[out],
            attrs={"axis": 2},
        )
        return out, node, extra_nodes

    if tiles_h == 1:
        node = gs.Node(
            name=f"{output_tensor.name}_concat",
            op="Concat",
            inputs=tiles,
            outputs=[out],
            attrs={"axis": 3},
        )
        return out, node, extra_nodes

    row_outputs = []
    for split_h in range(tiles_h):
        row_tiles = tiles[split_h * tiles_w : (split_h + 1) * tiles_w]
        row_shape = list(output_tensor.shape)
        row_shape[2] = row_tiles[0].shape[2]
        row_output = gs.Variable(
            f"{output_tensor.name}_row{split_h}",
            dtype=output_tensor.dtype,
            shape=row_shape,
        )
        row_node = gs.Node(
            name=f"{output_tensor.name}_concat_row{split_h}",
            op="Concat",
            inputs=row_tiles,
            outputs=[row_output],
            attrs={"axis": 3},
        )
        row_outputs.append(row_output)
        extra_nodes.append(row_node)

    node = gs.Node(
        name=f"{output_tensor.name}_concat",
        op="Concat",
        inputs=row_outputs,
        outputs=[out],
        attrs={"axis": 2},
    )
    return out, node, extra_nodes
