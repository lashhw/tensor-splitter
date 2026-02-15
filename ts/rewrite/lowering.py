import numpy as np
import onnx_graphsurgeon as gs

from .conv import _conv_attrs_with_hw_pad
from .tensor import _shape_with_dim_size


def _node_base_name(node):
    if node.name:
        return node.name
    if node.outputs and node.outputs[0].name:
        return node.outputs[0].name
    return node.op


def _make_constant(name, values):
    return gs.Constant(name, values)


def _build_entry_tiles(entry_tensor, entry_ranges, axis=2):
    tiles = []
    slice_nodes = []
    for tile_id, ((start_h, end_h), (start_w, end_w)) in enumerate(entry_ranges):
        starts_arr = np.array([start_h, start_w], dtype=np.int64)
        ends_arr = np.array([end_h, end_w], dtype=np.int64)
        axes_arr = np.array([axis, axis + 1], dtype=np.int64)
        steps_arr = np.array([1, 1], dtype=np.int64)
        out_shape = _shape_with_dim_size(entry_tensor.shape, axis, end_h - start_h)
        out_shape = _shape_with_dim_size(out_shape, axis + 1, end_w - start_w)

        base_name = f"{entry_tensor.name}_slice{tile_id}"
        starts = _make_constant(f"{base_name}_starts", starts_arr)
        ends = _make_constant(f"{base_name}_ends", ends_arr)
        axes = _make_constant(f"{base_name}_axes", axes_arr)
        steps = _make_constant(f"{base_name}_steps", steps_arr)
        out = gs.Variable(
            f"{entry_tensor.name}_split{tile_id}",
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
    out_ranges,
    conv_slices,
    main_input_index,
):
    out_tiles = []
    new_nodes = []
    node_base_name = _node_base_name(node)
    for tile_id, (tile, ((y0, y1), (x0, x1)), slice_info) in enumerate(zip(tiles, out_ranges, conv_slices)):
        attrs = _conv_attrs_with_hw_pad(node, slice_info)
        out_shape = _shape_with_dim_size(node.outputs[0].shape, 2, y1 - y0)
        out_shape = _shape_with_dim_size(out_shape, 3, x1 - x0)

        conv_inputs = list(node.inputs)
        conv_inputs[main_input_index] = tile

        conv_out = gs.Variable(
            f"{node.outputs[0].name}_split{tile_id}",
            dtype=node.outputs[0].dtype,
            shape=out_shape,
        )
        out_tiles.append(conv_out)

        conv_node = gs.Node(
            name=f"{node_base_name}_split{tile_id}",
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
            attrs=dict(node.attrs) if node.attrs else {},
        )
        new_nodes.append(new_node)

    return out_tiles, new_nodes


def _build_stage_tiles(
    node,
    tiles,
    out_ranges,
    main_input_index,
    conv_slices=None,
):
    if node.op == "Conv":
        assert conv_slices is not None, "conv_slices are required for Conv lowering"
        return _build_conv_tiles(node, tiles, out_ranges, conv_slices, main_input_index)
    return _build_non_conv_tiles(node, tiles, main_input_index)


def _build_group_concat(tiles, axis, output_tensor, split_keys, tile_count):
    split_count_h, split_count_w = tile_count
    assert len(split_keys) == len(tiles), "split_keys and tiles must have the same length"

    tile_by_key = {key: tile for key, tile in zip(split_keys, tiles)}
    if split_count_w == 1:
        ordered_tiles = [tile_by_key[(split_id_h, 0)] for split_id_h in range(split_count_h)]
        out = gs.Variable(
            output_tensor.name,
            dtype=output_tensor.dtype,
            shape=list(output_tensor.shape),
        )
        node = gs.Node(
            name=f"{output_tensor.name}_concat",
            op="Concat",
            inputs=ordered_tiles,
            outputs=[out],
            attrs={"axis": axis},
        )
        return out, node

    concat_nodes = []
    row_outputs = []
    for split_id_h in range(split_count_h):
        row_tiles = [tile_by_key[(split_id_h, split_id_w)] for split_id_w in range(split_count_w)]
        row_out_shape = _shape_with_dim_size(output_tensor.shape, 2, row_tiles[0].shape[2])
        row_out = gs.Variable(
            f"{output_tensor.name}_row{split_id_h}",
            dtype=output_tensor.dtype,
            shape=row_out_shape,
        )
        concat_nodes.append(
            gs.Node(
                name=f"{output_tensor.name}_concat_row{split_id_h}",
                op="Concat",
                inputs=row_tiles,
                outputs=[row_out],
                attrs={"axis": 3},
            )
        )
        row_outputs.append(row_out)

    out = gs.Variable(
        output_tensor.name,
        dtype=output_tensor.dtype,
        shape=list(output_tensor.shape),
    )
    node = gs.Node(
        name=f"{output_tensor.name}_concat",
        op="Concat",
        inputs=row_outputs,
        outputs=[out],
        attrs={"axis": axis},
    )
    concat_nodes.append(node)
    return out, concat_nodes
