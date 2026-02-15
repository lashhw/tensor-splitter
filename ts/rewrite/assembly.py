from .analysis import _ensure_supported_op, _ensure_toposorted
from .lowering import _build_entry_tiles, _build_group_concat, _build_stage_tiles
from .planning import _build_ordered_node_writer, _plan_stage_ranges


def _build_group(group_info, group_cfg, node_index_map):
    for node in group_info.nodes:
        _ensure_supported_op(node)

    stage_plan = _plan_stage_ranges(group_info, group_cfg.tile_count)
    first_orig_index = node_index_map[id(group_info.nodes[0])]
    place_node, finalize_nodes = _build_ordered_node_writer(group_cfg.execution_order, first_orig_index)

    tiles, entry_nodes = _build_entry_tiles(group_info.entry_tensor, stage_plan.stage_ranges[0], axis=2)
    for tile_id, entry_node in enumerate(entry_nodes):
        place_node(first_orig_index, tile_id, entry_node)

    for stage_idx, node in enumerate(group_info.nodes):
        orig_index = node_index_map[id(node)]
        main_input_idx = group_info.main_input_indices[stage_idx]
        out_ranges = stage_plan.stage_ranges[stage_idx + 1]
        conv_slices = stage_plan.conv_slices_by_stage[stage_idx]
        tiles, op_nodes = _build_stage_tiles(node, tiles, out_ranges, main_input_idx, conv_slices)
        for tile_id, op_node in enumerate(op_nodes):
            place_node(orig_index, tile_id, op_node)

    concat_out, concat_node = _build_group_concat(tiles, axis=2, output_tensor=group_info.exit_tensor)
    ordered_nodes = finalize_nodes(concat_node)
    _ensure_toposorted(ordered_nodes)

    return ordered_nodes, concat_out


def _apply_group(graph, orig_nodes, group_info, group_cfg, new_nodes, concat_out):
    node_a = orig_nodes[group_cfg.node_range[0]]
    node_b = orig_nodes[group_cfg.node_range[1]]
    start_pos = next(i for i, node in enumerate(graph.nodes) if node is node_a)
    end_pos = next(i for i, node in enumerate(graph.nodes) if node is node_b)

    for consumer in list(group_info.exit_tensor.outputs):
        for idx, inp in enumerate(consumer.inputs):
            if inp is group_info.exit_tensor:
                consumer.inputs[idx] = concat_out

    for idx, out in enumerate(graph.outputs):
        if out is group_info.exit_tensor:
            graph.outputs[idx] = concat_out

    for node in group_info.nodes:
        node.inputs = []
        node.outputs = []

    graph.nodes = graph.nodes[:start_pos] + new_nodes + graph.nodes[end_pos + 1 :]
