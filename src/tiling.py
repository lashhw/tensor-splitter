class TilingError(ValueError):
    pass


class ConvInputSlice:
    def __init__(self, slice_start, slice_end, pad_top, pad_bottom, x0, x1):
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.x0 = x0
        self.x1 = x1


def partition_ranges(total, splits):
    if total is None or total <= 0:
        raise TilingError(f"total must be > 0; got {total}")
    if splits <= 0:
        raise TilingError(f"splits must be > 0; got {splits}")
    if splits > total:
        raise TilingError(
            f"cannot split dimension {total} into {splits} non-empty tiles"
        )
    base = total // splits
    rem = total % splits
    ranges = []
    start = 0
    for i in range(splits):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def receptive_field(kernel, dilation):
    return (kernel - 1) * dilation + 1


def conv_input_slice_for_output(
    y0,
    y1,
    stride,
    dilation,
    kernel,
    pad_top,
    h_in,
):
    if y1 <= y0:
        raise TilingError(f"invalid output range [{y0},{y1})")
    rf = receptive_field(kernel, dilation)
    x0 = y0 * stride - pad_top
    x1 = (y1 - 1) * stride - pad_top + rf
    slice_start = max(0, x0)
    slice_end = min(h_in, x1)
    pad_top_local = max(0, -x0)
    pad_bottom_local = max(0, x1 - h_in)
    return ConvInputSlice(
        slice_start=slice_start,
        slice_end=slice_end,
        pad_top=pad_top_local,
        pad_bottom=pad_bottom_local,
        x0=x0,
        x1=x1,
    )


def conv_output_height(
    h_in,
    kernel,
    stride,
    dilation,
    pad_top,
    pad_bottom,
):
    rf = receptive_field(kernel, dilation)
    return ((h_in + pad_top + pad_bottom - rf) // stride) + 1
