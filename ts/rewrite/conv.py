from collections import namedtuple

ConvSpec = namedtuple("ConvSpec", ["kernel_shape", "strides", "pads"])


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


def _parse_conv_spec(node):
    attrs = node.attrs or {}
    assert "auto_pad" not in attrs, "Conv auto_pad must be unset in attrs"

    kernel_shape = _as_int_list(_get_attr(node, "kernel_shape"), name="kernel_shape", length=2)
    strides = _as_int_list(_get_attr(node, "strides"), name="strides", length=2)
    dilations = _as_int_list(_get_attr(node, "dilations"), name="dilations", length=2)
    pads = _as_int_list(_get_attr(node, "pads"), name="pads", length=4)
    assert dilations == [1, 1], f"Conv dilations {dilations} are not supported; expected [1, 1]"

    return ConvSpec(kernel_shape=kernel_shape, strides=strides, pads=pads)


def _conv_attrs_with_height_pad(node, base_pads, slice_info):
    attrs = dict(node.attrs) if node.attrs else {}
    attrs["pads"] = [slice_info.pad_top, base_pads[1], slice_info.pad_bottom, base_pads[3]]
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
