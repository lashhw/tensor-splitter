from collections import namedtuple

ConvInputSlice = namedtuple(
    "ConvInputSlice",
    ["slice_start", "slice_end", "pad_top", "pad_bottom"],
)


def _partition_ranges(total, tile_count):
    base = total // tile_count
    rem = total % tile_count
    ranges = []
    start = 0
    for i in range(tile_count):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _conv_input_slice_for_output(
    y0,
    y1,
    stride,
    kernel,
    pad_top,
    h_in,
):
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
