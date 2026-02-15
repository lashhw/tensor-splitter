from collections import namedtuple

_ConvSpec = namedtuple(
    "_ConvSpec",
    ["kernel_shape", "strides", "pads"]
)

_ConvInputSlice = namedtuple(
    "_ConvInputSlice",
    ["slice_start", "slice_end", "pad_top", "pad_bottom"],
)

_ConvInputSlice2D = namedtuple(
    "_ConvInputSlice2D",
    [
        "h_start",
        "h_end",
        "w_start",
        "w_end",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
    ],
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

    return _ConvSpec(kernel_shape=kernel_shape, strides=strides, pads=pads)


def _conv_attrs_with_height_pad(node, slice_info):
    attrs = dict(node.attrs)
    attrs["pads"] = [slice_info.pad_top, attrs["pads"][1], slice_info.pad_bottom, attrs["pads"][3]]
    return attrs


def _conv_attrs_with_spatial_pad(node, slice_info):
    attrs = dict(node.attrs)
    attrs["pads"] = [
        slice_info.pad_top,
        slice_info.pad_left,
        slice_info.pad_bottom,
        slice_info.pad_right,
    ]
    return attrs


def _conv_input_slice_1d(out_start, out_end, kernel, stride, pad_before, input_size):
    in_start = out_start * stride - pad_before
    in_end = (out_end - 1) * stride - pad_before + kernel

    slice_start = max(0, in_start)
    slice_end = min(in_end, input_size)
    assert slice_start < slice_end

    return slice_start, slice_end, max(0, -in_start), max(0, in_end - input_size)


def _conv_input_slice_for_output(y0, y1, spec, h_in):
    slice_start, slice_end, pad_top, pad_bottom = _conv_input_slice_1d(
        y0,
        y1,
        kernel=spec.kernel_shape[0],
        stride=spec.strides[0],
        pad_before=spec.pads[0],
        input_size=h_in,
    )

    return _ConvInputSlice(
        slice_start=slice_start,
        slice_end=slice_end,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )


def _conv_input_slice_for_output_2d(y0, y1, x0, x1, spec, h_in, w_in):
    h_start, h_end, pad_top, pad_bottom = _conv_input_slice_1d(
        y0,
        y1,
        kernel=spec.kernel_shape[0],
        stride=spec.strides[0],
        pad_before=spec.pads[0],
        input_size=h_in,
    )
    w_start, w_end, pad_left, pad_right = _conv_input_slice_1d(
        x0,
        x1,
        kernel=spec.kernel_shape[1],
        stride=spec.strides[1],
        pad_before=spec.pads[1],
        input_size=w_in,
    )

    return _ConvInputSlice2D(
        h_start=h_start,
        h_end=h_end,
        w_start=w_start,
        w_end=w_end,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )
