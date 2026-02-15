from collections import namedtuple

import onnx_graphsurgeon as gs

from .tensor import _shape_with_dim_size


ConvInputSlice = namedtuple(
    "ConvInputSlice",
    ["slice_start", "slice_end", "pad_top", "pad_bottom"],
)


def _get_attr(node, name, default=None):
    if node.attrs is None:
        return default
    return node.attrs.get(name, default)


def _as_int_list(value, *, name, length):
    assert value is not None, f"Conv attribute {name} is required"
    assert isinstance(value, list), f"Conv attribute {name} must be a list; got {value!r}"
    assert len(value) == length, f"Conv attribute {name} must have length {length}; got {value}"
    return value


def _conv_params(node):
    attrs = node.attrs or {}
    assert "auto_pad" not in attrs, "Conv auto_pad must be unset in attrs"

    kernel_shape = _as_int_list(_get_attr(node, "kernel_shape"), name="kernel_shape", length=2)
    strides = _as_int_list(_get_attr(node, "strides"), name="strides", length=2)
    dilations = _as_int_list(_get_attr(node, "dilations"), name="dilations", length=2)
    pads = _as_int_list(_get_attr(node, "pads"), name="pads", length=4)
    assert dilations == [1, 1], f"Conv dilations {dilations} are not supported; expected [1, 1]"

    return kernel_shape, strides, pads


def _conv_attrs_with_height_pad(node, pads):
    attrs = dict(node.attrs) if node.attrs else {}
    attrs["pads"] = pads
    return attrs


def _conv_input_slice_for_output(y0, y1, stride, kernel, pad_top, h_in):
    x0 = y0 * stride - pad_top
    x1 = (y1 - 1) * stride - pad_top + kernel

    slice_start = max(0, x0)
    slice_end = min(x1, h_in)
    assert slice_start < slice_end

    return ConvInputSlice(
        slice_start=slice_start,
        slice_end=slice_end,
        pad_top=max(0, -x0),
        pad_bottom=max(0, x1 - h_in),
    )


def _build_conv_tiles(
    node,
    tiles,
    out_ranges,
    conv_slices,
    conv_base_pads,
    main_input_index,
):
    out_tiles = []
    new_nodes = []
    node_base_name = node.name or (node.outputs[0].name if node.outputs and node.outputs[0].name else node.op)

    for tile_id, (tile, (y0, y1), slice_info) in enumerate(zip(tiles, out_ranges, conv_slices)):
        new_pads = [slice_info.pad_top, conv_base_pads[1], slice_info.pad_bottom, conv_base_pads[3]]
        attrs = _conv_attrs_with_height_pad(node, new_pads)
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
