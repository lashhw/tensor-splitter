import numpy as np
import onnx_graphsurgeon as gs

from .conv import _conv_attrs_with_height_pad
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

    for tile_id, (start, end) in enumerate(entry_ranges):
        base_name = f"{entry_tensor.name}_slice{tile_id}"
        starts = _make_constant(f"{base_name}_starts", np.array([start], dtype=np.int64))
        ends = _make_constant(f"{base_name}_ends", np.array([end], dtype=np.int64))
        axes = _make_constant(f"{base_name}_axes", np.array([axis], dtype=np.int64))
        steps = _make_constant(f"{base_name}_steps", np.array([1], dtype=np.int64))
        out = gs.Variable(
            f"{entry_tensor.name}_split{tile_id}",
            dtype=entry_tensor.dtype,
            shape=_shape_with_dim_size(entry_tensor.shape, axis, end - start),
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
    conv_spec,
    main_input_index,
):
    out_tiles = []
    new_nodes = []
    node_base_name = _node_base_name(node)

    for tile_id, (tile, (y0, y1), slice_info) in enumerate(zip(tiles, out_ranges, conv_slices)):
        attrs = _conv_attrs_with_height_pad(node, conv_spec.pads, slice_info)
        conv_inputs = list(node.inputs)
        conv_inputs[main_input_index] = tile

        out_shape = _shape_with_dim_size(node.outputs[0].shape, 2, y1 - y0)
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
    conv_spec=None,
):
    if node.op == "Conv":
        assert conv_slices is not None, "conv_slices are required for Conv lowering"
        assert conv_spec is not None, "conv_spec is required for Conv lowering"
        return _build_conv_tiles(node, tiles, out_ranges, conv_slices, conv_spec, main_input_index)
    return _build_non_conv_tiles(node, tiles, main_input_index)


def _build_group_concat(tiles, axis, output_tensor):
    out = gs.Variable(
        output_tensor.name,
        dtype=output_tensor.dtype,
        shape=list(output_tensor.shape),
    )
    node = gs.Node(
        name=f"{output_tensor.name}_concat",
        op="Concat",
        inputs=tiles,
        outputs=[out],
        attrs={"axis": axis},
    )
    return out, node
