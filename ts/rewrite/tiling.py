from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ConvInputSlice:
    slice_start: int
    slice_end: int
    pad_top: int
    pad_bottom: int
    x0: int
    x1: int


def _partition_ranges(total: int, tile_count: int) -> List[Tuple[int, int]]:
    assert total is not None and total > 0, f"total must be > 0; got {total}"
    assert tile_count > 0, f"tile_count must be > 0; got {tile_count}"
    assert tile_count <= total, (
        f"cannot split dimension {total} into {tile_count} non-empty tiles"
    )

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


def _receptive_field(kernel: int, dilation: int) -> int:
    return (kernel - 1) * dilation + 1


def _conv_input_slice_for_output(
    y0: int,
    y1: int,
    stride: int,
    dilation: int,
    kernel: int,
    pad_top: int,
    h_in: int,
) -> ConvInputSlice:
    assert y1 > y0, f"empty or invalid Conv output range [{y0},{y1}) is not supported"
    assert h_in >= 0, f"invalid input height {h_in}"

    rf = _receptive_field(kernel, dilation)
    x0 = y0 * stride - pad_top
    x1 = (y1 - 1) * stride - pad_top + rf

    # Clamp both bounds into [0, h_in] to avoid inverted ranges when output
    # tiles map fully outside the original tensor and rely only on padding.
    slice_start = min(max(0, x0), h_in)
    slice_end = min(max(0, x1), h_in)

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


def _conv_output_height(
    h_in: int,
    kernel: int,
    stride: int,
    dilation: int,
    pad_top: int,
    pad_bottom: int,
) -> int:
    rf = _receptive_field(kernel, dilation)
    return ((h_in + pad_top + pad_bottom - rf) // stride) + 1
