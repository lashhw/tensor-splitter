from collections import namedtuple

from .conv import _conv_input_slice_for_output, _parse_conv_spec
from .tensor import _tensor_height

StagePlan = namedtuple(
    "StagePlan",
    ["stage_ranges", "conv_slices_by_stage", "conv_specs_by_stage"],
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


def _plan_stage_ranges(group_info, tile_count):
    """
    Build per-stage required ranges with backward propagation from group output.

    stage_ranges[i] is the required range list for the tensor entering node i.
    stage_ranges[-1] is the required range list for group output (used for final concat).
    """
    stage_count = len(group_info.nodes) + 1
    stage_ranges = [[] for _ in range(stage_count)]
    stage_ranges[-1] = _partition_ranges(_tensor_height(group_info.exit_tensor), tile_count)

    conv_slices_by_stage = [None for _ in range(stage_count - 1)]
    conv_specs_by_stage = [None for _ in range(stage_count - 1)]

    for stage_idx in range(stage_count - 2, -1, -1):
        node = group_info.nodes[stage_idx]
        out_ranges = stage_ranges[stage_idx + 1]
        main_idx = group_info.main_input_indices[stage_idx]

        if node.op != "Conv":
            h_in = _tensor_height(node.inputs[main_idx])
            h_out = _tensor_height(node.outputs[0])
            assert h_in == h_out, (
                f"node {node.name or node.op} changes height ({h_in}->{h_out}) but is not Conv"
            )
            stage_ranges[stage_idx] = list(out_ranges)
            continue

        spec = _parse_conv_spec(node)
        h_in = _tensor_height(node.inputs[main_idx])

        in_ranges = []
        conv_slices = []
        for y0, y1 in out_ranges:
            slice_info = _conv_input_slice_for_output(y0, y1, spec, h_in)
            in_ranges.append((slice_info.slice_start, slice_info.slice_end))
            conv_slices.append(slice_info)

        stage_ranges[stage_idx] = in_ranges
        conv_slices_by_stage[stage_idx] = conv_slices
        conv_specs_by_stage[stage_idx] = spec

    return StagePlan(
        stage_ranges=stage_ranges,
        conv_slices_by_stage=conv_slices_by_stage,
        conv_specs_by_stage=conv_specs_by_stage,
    )


def _slot_offsets(slot_sizes):
    offsets = []
    running = 0
    for size in slot_sizes:
        offsets.append(running)
        running += size
    return offsets, running


def _build_ordered_node_writer(schedule, first_orig_index):
    schedule_pos = {pair: idx for idx, pair in enumerate(schedule)}
    slot_sizes = [2 if orig_index == first_orig_index else 1 for orig_index, _ in schedule]
    slot_offsets, total_rewritten = _slot_offsets(slot_sizes)
    slot_limits = [offset + size for offset, size in zip(slot_offsets, slot_sizes)]
    slot_cursor = list(slot_offsets)
    ordered_nodes = [None for _ in range(total_rewritten + 1)]

    def place_node(orig_index, tile_id, node):
        key = (orig_index, tile_id)
        assert key in schedule_pos, f"missing execution_order entry for {key}"
        slot_idx = schedule_pos[key]
        write_idx = slot_cursor[slot_idx]
        assert write_idx < slot_limits[slot_idx], f"execution_order slot {key} has too many rewritten nodes"
        ordered_nodes[write_idx] = node
        slot_cursor[slot_idx] += 1

    def finalize_nodes(concat_node):
        for slot_idx, key in enumerate(schedule):
            assert slot_cursor[slot_idx] == slot_limits[slot_idx], (
                f"execution_order slot {key} has missing rewritten nodes"
            )

        ordered_nodes[-1] = concat_node
        assert all(node is not None for node in ordered_nodes), "internal error: unfilled ordered node slot"
        return ordered_nodes

    return place_node, finalize_nodes
