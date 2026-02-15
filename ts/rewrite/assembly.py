from .catalog import _ensure_supported_op
from .conv import _conv_params
from .lowering import _build_tiled_op
from .tensor import _make_concat, _make_slice, _tensor_height
from .tiling import _conv_input_slice_for_output, _partition_ranges
from .ordering import _ensure_toposorted


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
    conv_base_pads_by_stage = [None for _ in range(stage_count - 1)]

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

        kernel_shape, strides, pads = _conv_params(node)
        k_h = kernel_shape[0]
        s_h = strides[0]
        pad_top = pads[0]
        h_in = _tensor_height(node.inputs[main_idx])

        in_ranges = []
        conv_slices = []
        for y0, y1 in out_ranges:
            slice_info = _conv_input_slice_for_output(y0, y1, s_h, k_h, pad_top, h_in)
            in_ranges.append((slice_info.slice_start, slice_info.slice_end))
            conv_slices.append(slice_info)
        stage_ranges[stage_idx] = in_ranges
        conv_slices_by_stage[stage_idx] = conv_slices
        conv_base_pads_by_stage[stage_idx] = list(pads)

    return stage_ranges, conv_slices_by_stage, conv_base_pads_by_stage


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


def _build_entry_tiles(entry, entry_ranges, first_orig_index, place_node):
    tiles = []
    for tile_id, (start, end) in enumerate(entry_ranges):
        tile, node = _make_slice(entry, start, end, 2, tile_id)
        tiles.append(tile)
        place_node(first_orig_index, tile_id, node)
    return tiles


def _build_group(group_info, group_cfg, node_index_map):
    for node in group_info.nodes:
        _ensure_supported_op(node)

    stage_ranges, conv_slices_by_stage, conv_base_pads_by_stage = _plan_stage_ranges(group_info, group_cfg.tile_count)
    first_orig_index = node_index_map[id(group_info.nodes[0])]
    place_node, finalize_nodes = _build_ordered_node_writer(group_cfg.execution_order, first_orig_index)
    tiles = _build_entry_tiles(group_info.entry_tensor, stage_ranges[0], first_orig_index, place_node)

    for stage_idx, node in enumerate(group_info.nodes):
        orig_index = node_index_map[id(node)]
        main_idx = group_info.main_input_indices[stage_idx]
        out_ranges = stage_ranges[stage_idx + 1]
        conv_slices = conv_slices_by_stage[stage_idx]
        conv_base_pads = conv_base_pads_by_stage[stage_idx]
        tiles, op_nodes = _build_tiled_op(node, tiles, out_ranges, main_idx, conv_slices, conv_base_pads)
        for tile_id, op_node in enumerate(op_nodes):
            place_node(orig_index, tile_id, op_node)

    concat_out, concat_node = _make_concat(
        tiles,
        2,
        group_info.exit_tensor.shape,
        group_info.exit_tensor.name,
    )
    ordered_nodes = finalize_nodes(concat_node)
    _ensure_toposorted(ordered_nodes)

    return ordered_nodes, concat_out
