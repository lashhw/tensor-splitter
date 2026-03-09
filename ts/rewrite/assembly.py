from .analysis import _ensure_toposorted
from .lowering import _build_entry_tiles, _build_group_concat, _build_tile_crop, _build_tiled_node
from .planning import _plan_node_ranges


def _group_name_scope(group_info):
    start, end = group_info.node_range
    return f"g{start}_{end}"


def _get_tile_from_source(range_plan, tiles_by_local_index, entry_tiles, source, split_pos):
    if source.kind == "entry":
        tile = entry_tiles[split_pos]
        produced_range = range_plan.entry_ranges[split_pos]
    else:
        producer_tiles = tiles_by_local_index[source.producer_local_index]
        tile = producer_tiles[split_pos]
        produced_range = range_plan.output_ranges_by_node[source.producer_local_index][split_pos]
    return tile, produced_range


def _crop_tile_if_needed(tile, produced_range, required_range, split_id, name_prefix, name_scope):
    if produced_range == required_range:
        return tile, []

    cropped_tile, crop_node = _build_tile_crop(
        tile=tile,
        produced_range=produced_range,
        required_range=required_range,
        split_id=split_id,
        name_prefix=name_prefix,
        name_scope=name_scope,
    )
    return cropped_tile, [crop_node]


def _build_tiled_step(
    group_info,
    range_plan,
    tiles_by_local_index,
    entry_tiles,
    local_index,
    split_id,
    split_pos,
    name_scope,
):
    node_spec = group_info.node_specs[local_index]
    node = node_spec.node
    demanded_ranges = range_plan.input_ranges_by_node[local_index]
    step_nodes = []
    input_tensors_by_index = {}

    for input_index, source in node_spec.input_sources.items():
        source_tile, produced_range = _get_tile_from_source(
            range_plan=range_plan,
            tiles_by_local_index=tiles_by_local_index,
            entry_tiles=entry_tiles,
            source=source,
            split_pos=split_pos,
        )
        assert source_tile is not None, (
            f"execution_order violates dependencies: node index {group_info.node_range[0] + local_index} "
            f"split {split_id} uses an input before producer is built"
        )

        if input_index not in demanded_ranges:
            input_tensors_by_index[input_index] = source_tile
            continue

        prepared_tile, prep_nodes = _crop_tile_if_needed(
            tile=source_tile,
            produced_range=produced_range,
            required_range=demanded_ranges[input_index][split_pos],
            split_id=split_id,
            name_prefix=f"{node.name or node.op}_l{local_index}_in{input_index}",
            name_scope=name_scope,
        )
        step_nodes.extend(prep_nodes)
        input_tensors_by_index[input_index] = prepared_tile

    out_range = range_plan.output_ranges_by_node[local_index][split_pos]
    spatial_slice = None
    if node.op in {"Conv", "AveragePool"}:
        spatial_slice = range_plan.spatial_slices_by_node[local_index][split_pos]

    output_tile, tiled_node = _build_tiled_node(
        node=node,
        split_id=split_id,
        input_tensors_by_index=input_tensors_by_index,
        out_range=out_range,
        spatial_slice=spatial_slice,
        name_scope=name_scope,
    )
    step_nodes.append(tiled_node)
    return output_tile, step_nodes


def _group_outputs_to_stitch(group_info):
    sink_local_index = len(group_info.nodes) - 1
    outputs = [(sink_local_index, group_info.exit_tensor)]
    outputs.extend((spec.local_index, spec.output_tensor) for spec in group_info.boundary_output_specs)
    return outputs


def _build_stitched_output(
    local_index,
    output_tensor,
    produced_tiles,
    produced_ranges,
    range_plan,
    tile_count,
    stitch_nodes,
    name_scope,
):
    stitch_tiles = []
    stitch_ranges = range_plan.stitch_ranges_by_local_index[local_index]
    for split_pos, split_id in enumerate(range_plan.split_keys):
        stitched_tile, prep_nodes = _crop_tile_if_needed(
            tile=produced_tiles[split_pos],
            produced_range=produced_ranges[split_pos],
            required_range=stitch_ranges[split_pos],
            split_id=split_id,
            name_prefix=f"{output_tensor.name}_stitch_l{local_index}",
            name_scope=name_scope,
        )
        stitch_nodes.extend(prep_nodes)
        stitch_tiles.append(stitched_tile)

    stitched_out, concat_nodes = _build_group_concat(
        tiles=stitch_tiles,
        output_tensor=output_tensor,
        split_keys=range_plan.split_keys,
        tile_count=tile_count,
        name_scope=name_scope,
    )
    stitch_nodes.extend(concat_nodes)
    return stitched_out


def _build_group(group_info):
    range_plan = _plan_node_ranges(group_info)
    split_pos_by_key = {split_key: split_pos for split_pos, split_key in enumerate(range_plan.split_keys)}
    group_start, _ = group_info.node_range
    name_scope = _group_name_scope(group_info)

    entry_tiles, entry_split_nodes = _build_entry_tiles(
        group_info.entry_tensor,
        range_plan.entry_ranges,
        name_scope=name_scope,
    )

    tiles_by_local_index = [[None for _ in range(len(range_plan.split_keys))] for _ in group_info.nodes]
    body_nodes = []

    for orig_index, split_id in group_info.execution_order:
        assert split_id in split_pos_by_key, f"invalid split id {split_id} in execution_order"
        split_pos = split_pos_by_key[split_id]
        local_index = orig_index - group_start
        assert 0 <= local_index < len(group_info.nodes), (
            f"node index {orig_index} in execution_order is outside group range {group_info.node_range}"
        )

        output_tile, step_nodes = _build_tiled_step(
            group_info=group_info,
            range_plan=range_plan,
            tiles_by_local_index=tiles_by_local_index,
            entry_tiles=entry_tiles,
            local_index=local_index,
            split_id=split_id,
            split_pos=split_pos,
            name_scope=name_scope,
        )
        tiles_by_local_index[local_index][split_pos] = output_tile
        body_nodes.extend(step_nodes)

    for local_index, node_tiles in enumerate(tiles_by_local_index):
        assert all(tile is not None for tile in node_tiles), (
            f"execution_order has missing rewritten nodes for node index {group_info.node_range[0] + local_index}"
        )

    stitched_outputs_by_tensor_id = {}
    stitch_nodes = []
    for local_index, output_tensor in _group_outputs_to_stitch(group_info):
        stitched_out = _build_stitched_output(
            local_index=local_index,
            output_tensor=output_tensor,
            produced_tiles=tiles_by_local_index[local_index],
            produced_ranges=range_plan.output_ranges_by_node[local_index],
            range_plan=range_plan,
            tile_count=group_info.tile_count,
            stitch_nodes=stitch_nodes,
            name_scope=name_scope,
        )
        stitched_outputs_by_tensor_id[id(output_tensor)] = stitched_out

    ordered_nodes = entry_split_nodes + body_nodes + stitch_nodes
    _ensure_toposorted(ordered_nodes)
    return ordered_nodes, stitched_outputs_by_tensor_id


def _apply_group(graph, orig_nodes, group_info, new_nodes, stitched_outputs_by_tensor_id):
    node_a = orig_nodes[group_info.node_range[0]]
    node_b = orig_nodes[group_info.node_range[1]]
    start_pos = next(i for i, node in enumerate(graph.nodes) if node is node_a)
    end_pos = next(i for i, node in enumerate(graph.nodes) if node is node_b)
    group_node_ids = {id(node) for node in group_info.nodes}

    stitched_outputs = [group_info.exit_tensor]
    stitched_outputs.extend(spec.output_tensor for spec in group_info.boundary_output_specs)
    for output_tensor in stitched_outputs:
        stitched_out = stitched_outputs_by_tensor_id[id(output_tensor)]
        for consumer in list(output_tensor.outputs):
            if id(consumer) in group_node_ids:
                continue
            for idx, inp in enumerate(consumer.inputs):
                if inp is output_tensor:
                    consumer.inputs[idx] = stitched_out

        for idx, out in enumerate(graph.outputs):
            if out is output_tensor:
                graph.outputs[idx] = stitched_out

    for node in group_info.nodes:
        node.inputs = []
        node.outputs = []

    graph.nodes = graph.nodes[:start_pos] + new_nodes + graph.nodes[end_pos + 1 :]
