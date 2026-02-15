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
    ["height", "width"],
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


def _conv_attrs_with_hw_pad(node, slice_info_2d):
    attrs = dict(node.attrs)
    attrs["pads"] = [
        slice_info_2d.height.pad_top,
        slice_info_2d.width.pad_top,
        slice_info_2d.height.pad_bottom,
        slice_info_2d.width.pad_bottom,
    ]
    return attrs


def _conv_input_slice_for_output_axis(out0, out1, kernel, stride, pad_before, input_size):
    x0 = out0 * stride - pad_before
    x1 = (out1 - 1) * stride - pad_before + kernel

    slice_start = max(0, x0)
    slice_end = min(x1, input_size)
    assert slice_start < slice_end

    return _ConvInputSlice(
        slice_start=slice_start,
        slice_end=slice_end,
        pad_top=max(0, -x0),
        pad_bottom=max(0, x1 - input_size),
    )


def _conv_input_slice_for_output_2d(y0, y1, x0, x1, spec, h_in, w_in):
    return _ConvInputSlice2D(
        height=_conv_input_slice_for_output_axis(
            out0=y0,
            out1=y1,
            kernel=spec.kernel_shape[0],
            stride=spec.strides[0],
            pad_before=spec.pads[0],
            input_size=h_in,
        ),
        width=_conv_input_slice_for_output_axis(
            out0=x0,
            out1=x1,
            kernel=spec.kernel_shape[1],
            stride=spec.strides[1],
            pad_before=spec.pads[1],
            input_size=w_in,
        ),
    )
