from collections import namedtuple

ConvSpec = namedtuple("ConvSpec", ["kernel_shape", "strides", "pads"])


ConvInputSlice = namedtuple(
    "ConvInputSlice",
    ["slice_start", "slice_end", "pad_top", "pad_bottom"],
)


def _get_attr(node, name):
    return node.attrs[name]


def _ensure_list(value, length):
    assert isinstance(value, list)
    assert len(value) == length
    return value


def _parse_conv_spec(node):
    attrs = node.attrs

    assert "auto_pad" not in attrs, "Conv auto_pad must be unset in attrs"
    kernel_shape = _ensure_list(attrs["kernel_shape"], length=2)
    strides = _ensure_list(attrs["strides"], length=2)
    dilations = _ensure_list(attrs["dilations"], length=2)
    pads = _ensure_list(attrs["pads"], length=4)
    assert dilations == [1, 1], f"Conv dilations {dilations} are not supported; expected [1, 1]"

    return ConvSpec(kernel_shape=kernel_shape, strides=strides, pads=pads)


def _conv_attrs_with_height_pad(node, slice_info):
    attrs = dict(node.attrs)
    attrs["pads"] = [slice_info.pad_top, attrs["pads"][1], slice_info.pad_bottom, attrs["pads"][3]]
    return attrs


def _conv_input_slice_for_output(y0, y1, spec, h_in):
    k_h = spec.kernel_shape[0]
    s_h = spec.strides[0]
    pad_top = spec.pads[0]

    x0 = y0 * s_h - pad_top
    x1 = (y1 - 1) * s_h - pad_top + k_h

    slice_start = max(0, x0)
    slice_end = min(x1, h_in)
    assert slice_start < slice_end

    return ConvInputSlice(
        slice_start=slice_start,
        slice_end=slice_end,
        pad_top=max(0, -x0),
        pad_bottom=max(0, x1 - h_in),
    )
