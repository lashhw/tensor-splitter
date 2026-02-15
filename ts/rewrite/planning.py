from collections import namedtuple

from .conv import _conv_input_slice_for_output_2d, _parse_conv_spec
from .tensor import _tensor_height, _tensor_width

_StagePlan = namedtuple(
    "_StagePlan",
    ["stage_regions", "conv_slices_by_stage", "tile_ids"],
)

_WriterSlot = namedtuple(
    "_WriterSlot",
    ["key", "start_index", "limit"],
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


def _partition_spatial_regions(height, width, tile_count):
    if type(tile_count) is int:
        tile_count = (tile_count, 1)
    assert isinstance(tile_count, (list, tuple)) and len(tile_count) == 2
    tiles_h, tiles_w = tile_count
    h_ranges = _partition_ranges(height, tiles_h)
    w_ranges = _partition_ranges(width, tiles_w)

    tile_ids = []
    regions = []
    for split_h, (h0, h1) in enumerate(h_ranges):
        for split_w, (w0, w1) in enumerate(w_ranges):
            tile_ids.append((split_h, split_w))
            regions.append((h0, h1, w0, w1))
    return tile_ids, regions


def _plan_stage_ranges(group_info, tile_count):
    """
    Build per-stage required spatial regions with backward propagation from group output.

    stage_regions[i] is the required region list for the tensor entering node i.
    stage_regions[-1] is the required region list for group output (used for final concat).
    """
    stage_count = len(group_info.nodes) + 1
    stage_regions = [[] for _ in range(stage_count)]
    tile_ids, stage_regions[-1] = _partition_spatial_regions(
        _tensor_height(group_info.exit_tensor),
        _tensor_width(group_info.exit_tensor),
        tile_count,
    )

    conv_slices_by_stage = [None for _ in range(stage_count - 1)]

    for stage_idx in range(stage_count - 2, -1, -1):
        node = group_info.nodes[stage_idx]
        out_regions = stage_regions[stage_idx + 1]
        main_idx = group_info.main_input_indices[stage_idx]

        if node.op != "Conv":
            stage_regions[stage_idx] = list(out_regions)
            continue

        spec = _parse_conv_spec(node)
        h_in = _tensor_height(node.inputs[main_idx])
        w_in = _tensor_width(node.inputs[main_idx])

        in_regions = []
        conv_slices = []
        for y0, y1, x0, x1 in out_regions:
            slice_info = _conv_input_slice_for_output_2d(
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                spec=spec,
                h_in=h_in,
                w_in=w_in,
            )
            in_regions.append((
                slice_info.h_start,
                slice_info.h_end,
                slice_info.w_start,
                slice_info.w_end,
            ))
            conv_slices.append(slice_info)

        stage_regions[stage_idx] = in_regions
        conv_slices_by_stage[stage_idx] = conv_slices

    return _StagePlan(
        stage_regions=stage_regions,
        conv_slices_by_stage=conv_slices_by_stage,
        tile_ids=tile_ids,
    )


def _build_ordered_node_writer(schedule, first_orig_index):
    slot_by_key = {}
    slot_cursor = {}
    slots = []
    total_rewritten = 0

    for key in schedule:
        orig_index, _ = key
        capacity = 2 if orig_index == first_orig_index else 1
        slot = _WriterSlot(
            key=key,
            start_index=total_rewritten,
            limit=total_rewritten + capacity,
        )

        slot_by_key[key] = slot
        slot_cursor[key] = slot.start_index
        slots.append(slot)
        total_rewritten += capacity

    ordered_nodes = [None for _ in range(total_rewritten)]

    def place_node(orig_index, split_hw, node):
        key = (orig_index, split_hw)
        slot = slot_by_key[key]
        write_idx = slot_cursor[key]
        assert write_idx < slot.limit, f"execution_order slot {key} has too many rewritten nodes"
        ordered_nodes[write_idx] = node
        slot_cursor[key] = write_idx + 1

    def finalize_nodes(concat_node, extra_nodes=None):
        for slot in slots:
            assert slot_cursor[slot.key] == slot.limit, (
                f"execution_order slot {slot.key} has missing rewritten nodes"
            )
        assert all(node is not None for node in ordered_nodes), "internal error: unfilled ordered node slot"
        if extra_nodes is None:
            extra_nodes = []
        assert all(node is not None for node in extra_nodes), "internal error: unfilled extra node slot"
        return ordered_nodes + list(extra_nodes) + [concat_node]

    return place_node, finalize_nodes
