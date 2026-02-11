from dataclasses import dataclass


@dataclass(frozen=True)
class ConvInputSlice:
    """Input slice and local padding required for a target Conv output range."""

    slice_start: int
    slice_end: int
    pad_top: int
    pad_bottom: int
    x0: int
    x1: int


def partition_ranges(total: int, splits: int) -> list[tuple[int, int]]:
    """Partition `[0, total)` into `splits` non-empty contiguous ranges."""
    if total is None or total <= 0:
        raise ValueError(f"total must be > 0; got {total}")
    if splits <= 0:
        raise ValueError(f"splits must be > 0; got {splits}")
    if splits > total:
        raise ValueError(
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


def receptive_field(kernel: int, dilation: int) -> int:
    return (kernel - 1) * dilation + 1


def conv_input_slice_for_output(
    y0: int,
    y1: int,
    stride: int,
    dilation: int,
    kernel: int,
    pad_top: int,
    h_in: int,
) -> ConvInputSlice:
    """Compute the input span and local padding for producing output rows `[y0, y1)`."""
    if y1 <= y0:
        raise ValueError(f"invalid output range [{y0},{y1})")

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
    h_in: int,
    kernel: int,
    stride: int,
    dilation: int,
    pad_top: int,
    pad_bottom: int,
) -> int:
    rf = receptive_field(kernel, dilation)
    return ((h_in + pad_top + pad_bottom - rf) // stride) + 1
