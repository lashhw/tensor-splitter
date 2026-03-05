import numpy as np
import onnx_graphsurgeon as gs

from .conv import _avg_pool_attrs_with_hw_pad, _conv_attrs_with_hw_pad
from .tensor import _shape_with_dim_size


def _node_base_name(node):
    if node.name:
        return node.name
    if node.outputs and node.outputs[0].name:
        return node.outputs[0].name
    return node.op


def _make_constant(name, values):
    return gs.Constant(name, values)


def _resolve_constant_input_values(tensor):
    if isinstance(tensor, gs.Constant):
        return np.asarray(tensor.values, dtype=np.int64).reshape(-1)

    producers = list(tensor.inputs)
    if len(producers) != 1 or producers[0].op != "Constant":
        return None

    value_attr = producers[0].attrs.get("value")
    if isinstance(value_attr, gs.Constant):
        return np.asarray(value_attr.values, dtype=np.int64).reshape(-1)
    if isinstance(value_attr, np.ndarray):
        return np.asarray(value_attr, dtype=np.int64).reshape(-1)
    return None


def _infer_spatial_axis(shape):
    assert shape is not None and len(shape) >= 4, (
        f"tensor shape must have rank >= 4 for spatial rewrite; got {shape}"
    )
    return len(shape) - 2


def _shape_with_hw(shape, h_size, w_size, axis=None):
    if axis is None:
        axis = _infer_spatial_axis(shape)
    out_shape = _shape_with_dim_size(shape, axis, h_size)
    return _shape_with_dim_size(out_shape, axis + 1, w_size)


def _build_entry_tiles(entry_tensor, entry_ranges, axis=None):
    if axis is None:
        axis = _infer_spatial_axis(entry_tensor.shape)
    tiles = []
    slice_nodes = []
    for tile_id, ((start_h, end_h), (start_w, end_w)) in enumerate(entry_ranges):
        starts_arr = np.array([start_h, start_w], dtype=np.int64)
        ends_arr = np.array([end_h, end_w], dtype=np.int64)
        axes_arr = np.array([axis, axis + 1], dtype=np.int64)
        steps_arr = np.array([1, 1], dtype=np.int64)
        out_shape = _shape_with_hw(
            entry_tensor.shape,
            h_size=end_h - start_h,
            w_size=end_w - start_w,
            axis=axis,
        )

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


def _build_tile_crop(tile, produced_range, required_range, split_id, name_prefix, axis=None):
    if axis is None:
        axis = _infer_spatial_axis(tile.shape)

    (prod_y0, _), (prod_x0, _) = produced_range
    (req_y0, req_y1), (req_x0, req_x1) = required_range

    rel_start_h = req_y0 - prod_y0
    rel_end_h = req_y1 - prod_y0
    rel_start_w = req_x0 - prod_x0
    rel_end_w = req_x1 - prod_x0

    starts_arr = np.array([rel_start_h, rel_start_w], dtype=np.int64)
    ends_arr = np.array([rel_end_h, rel_end_w], dtype=np.int64)
    axes_arr = np.array([axis, axis + 1], dtype=np.int64)
    steps_arr = np.array([1, 1], dtype=np.int64)
    base_name = f"{name_prefix}_crop_s{split_id[0]}_{split_id[1]}"

    starts = _make_constant(f"{base_name}_starts", starts_arr)
    ends = _make_constant(f"{base_name}_ends", ends_arr)
    axes = _make_constant(f"{base_name}_axes", axes_arr)
    steps = _make_constant(f"{base_name}_steps", steps_arr)
    out = gs.Variable(
        f"{base_name}_out",
        dtype=tile.dtype,
        shape=_shape_with_hw(tile.shape, h_size=req_y1 - req_y0, w_size=req_x1 - req_x0, axis=axis),
    )
    node = gs.Node(
        name=base_name,
        op="Slice",
        inputs=[tile, starts, ends, axes, steps],
        outputs=[out],
    )
    return out, node


def _build_tiled_node(
    node,
    split_id,
    input_tensors_by_index,
    out_range,
    spatial_slice=None,
):
    node_base_name = _node_base_name(node)

    new_inputs = list(node.inputs)
    for input_index, tensor in input_tensors_by_index.items():
        new_inputs[input_index] = tensor

    if node.op == "Constant":
        out_shape = list(node.outputs[0].shape) if node.outputs[0].shape is not None else None
    else:
        assert out_range is not None, f"out_range is required when lowering op {node.op}"
        (y0, y1), (x0, x1) = out_range
        out_shape = _shape_with_hw(node.outputs[0].shape, h_size=y1 - y0, w_size=x1 - x0)

    if node.op == "Reshape" and len(new_inputs) >= 2:
        shape_values = _resolve_constant_input_values(node.inputs[1])
        if shape_values is not None and shape_values.size >= 2:
            target_h = out_shape[-2]
            target_w = out_shape[-1]
            if isinstance(target_h, int) and isinstance(target_w, int):
                reshaped = shape_values.copy()
                if reshaped[-2] > 0:
                    reshaped[-2] = target_h
                if reshaped[-1] > 0:
                    reshaped[-1] = target_w
                new_inputs[1] = _make_constant(
                    f"{node_base_name}_shape_s{split_id[0]}_{split_id[1]}",
                    reshaped.astype(np.int64),
                )

    out = gs.Variable(
        f"{node.outputs[0].name}_split_s{split_id[0]}_{split_id[1]}",
        dtype=node.outputs[0].dtype,
        shape=out_shape,
    )

    if node.op == "Conv":
        assert spatial_slice is not None, "spatial slice is required when lowering Conv nodes"
        attrs = _conv_attrs_with_hw_pad(node, spatial_slice)
    elif node.op == "AveragePool":
        assert spatial_slice is not None, "spatial slice is required when lowering AveragePool nodes"
        attrs = _avg_pool_attrs_with_hw_pad(node, spatial_slice)
    else:
        attrs = dict(node.attrs) if node.attrs else {}

    new_node = gs.Node(
        name=f"{node_base_name}_split_s{split_id[0]}_{split_id[1]}",
        op=node.op,
        inputs=new_inputs,
        outputs=[out],
        attrs=attrs,
    )
    return out, new_node


def _build_group_concat(tiles, axis, output_tensor, split_keys, tile_count):
    split_count_h, split_count_w = tile_count
    assert len(split_keys) == len(tiles), "split_keys and tiles must have the same length"

    tile_by_key = {key: tile for key, tile in zip(split_keys, tiles)}
    concat_nodes = []
    output_shape = list(output_tensor.shape)

    def _make_output(name, shape):
        return gs.Variable(name, dtype=output_tensor.dtype, shape=shape)

    if split_count_w == 1:
        final_concat_inputs = [tile_by_key[(split_id_h, 0)] for split_id_h in range(split_count_h)]
    else:
        final_concat_inputs = []
        for split_id_h in range(split_count_h):
            row_concat_inputs = [tile_by_key[(split_id_h, split_id_w)] for split_id_w in range(split_count_w)]
            row_out_shape = _shape_with_dim_size(output_tensor.shape, 2, row_concat_inputs[0].shape[2])
            row_out = _make_output(f"{output_tensor.name}_row{split_id_h}", row_out_shape)
            concat_nodes.append(
                gs.Node(
                    name=f"{output_tensor.name}_concat_row{split_id_h}",
                    op="Concat",
                    inputs=row_concat_inputs,
                    outputs=[row_out],
                    attrs={"axis": 3},
                )
            )
            final_concat_inputs.append(row_out)

    out = _make_output(output_tensor.name, output_shape)
    concat_nodes.append(
        gs.Node(
            name=f"{output_tensor.name}_concat",
            op="Concat",
            inputs=final_concat_inputs,
            outputs=[out],
            attrs={"axis": axis},
        )
    )
    return out, concat_nodes
