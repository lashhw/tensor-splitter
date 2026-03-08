from .analysis import _ensure_toposorted
from .lowering import _build_entry_tiles, _build_group_concat, _build_tile_crop, _build_tiled_node
from .planning import _plan_node_ranges


def _get_tile_from_source(range_plan, tiles_by_local_index, entry_tiles_by_key, source, split_pos):
    if source.kind == "entry":
        entry_tiles = entry_tiles_by_key[source.entry_key]
        tile = entry_tiles[split_pos]
        produced_range = range_plan.entry_ranges_by_key[source.entry_key][split_pos]
    else:
        producer_tiles = tiles_by_local_index[source.producer_local_index]
        tile = producer_tiles[split_pos]
        produced_range = range_plan.output_ranges_by_node[source.producer_local_index][split_pos]
    return tile, produced_range


def _crop_tile_if_needed(tile, produced_range, required_range, split_id, name_prefix, scheduled_nodes):
    if produced_range == required_range:
        return tile

    cropped_tile, crop_node = _build_tile_crop(
        tile=tile,
        produced_range=produced_range,
        required_range=required_range,
        split_id=split_id,
        name_prefix=name_prefix,
    )
    scheduled_nodes.append(crop_node)
    return cropped_tile


def _iter_stitch_targets(group_info):
    sink_local_index = len(group_info.nodes) - 1
    yield sink_local_index, group_info.exit_tensor
    for spec in group_info.boundary_output_specs:
        yield spec.local_index, spec.output_tensor


def _iter_stitched_output_tensors(group_info):
    yield group_info.exit_tensor
    for spec in group_info.boundary_output_specs:
        yield spec.output_tensor


def _build_stitched_output(
    local_index,
    output_tensor,
    produced_tiles,
    produced_ranges,
    stitch_ranges,
    split_keys,
    tile_count,
):
    stitch_tiles = []
    stitch_nodes = []
    for split_pos, split_id in enumerate(split_keys):
        stitched_tile = _crop_tile_if_needed(
            tile=produced_tiles[split_pos],
            produced_range=produced_ranges[split_pos],
            required_range=stitch_ranges[split_pos],
            split_id=split_id,
            name_prefix=f"{output_tensor.name}_stitch_l{local_index}",
            scheduled_nodes=stitch_nodes,
        )
        stitch_tiles.append(stitched_tile)

    stitched_out, concat_nodes = _build_group_concat(
        stitch_tiles,
        axis=2,
        output_tensor=output_tensor,
        split_keys=split_keys,
        tile_count=tile_count,
    )
    stitch_nodes.extend(concat_nodes)
    return stitched_out, stitch_nodes


def _build_group(group_info, group_cfg):
    range_plan = _plan_node_ranges(group_info, group_cfg.tile_count)
    split_pos_by_key = {split_key: split_pos for split_pos, split_key in enumerate(range_plan.split_keys)}
    group_start, _ = group_info.node_range

    entry_tiles_by_key = {}
    entry_slice_nodes = []
    for entry_key, entry_tensor in group_info.entry_tensors.items():
        entry_ranges = range_plan.entry_ranges_by_key[entry_key]
        entry_tiles, entry_nodes = _build_entry_tiles(entry_tensor, entry_ranges)
        entry_tiles_by_key[entry_key] = entry_tiles
        entry_slice_nodes.extend(entry_nodes)

    tiles_by_local_index = [[None for _ in range(len(range_plan.split_keys))] for _ in group_info.nodes]
    scheduled_nodes = []
    built_node_count = 0

    for orig_index, split_id in group_cfg.execution_order:
        assert split_id in split_pos_by_key, f"invalid split id {split_id} in execution_order"
        split_pos = split_pos_by_key[split_id]
        local_index = orig_index - group_start
        assert 0 <= local_index < len(group_info.nodes), (
            f"node index {orig_index} in execution_order is outside group range {group_info.node_range}"
        )

        node_spec = group_info.node_specs[local_index]
        node = node_spec.node
        input_tensors_by_index = {}
        demanded_ranges = range_plan.input_ranges_by_node[local_index]

        for input_index, source in node_spec.input_sources.items():
            source_tile, produced_range = _get_tile_from_source(
                range_plan=range_plan,
                tiles_by_local_index=tiles_by_local_index,
                entry_tiles_by_key=entry_tiles_by_key,
                source=source,
                split_pos=split_pos,
            )
            assert source_tile is not None, (
                f"execution_order violates dependencies: node index {orig_index} split {split_id} "
                f"uses an input before producer is built"
            )

            if input_index not in demanded_ranges:
                input_tensors_by_index[input_index] = source_tile
                continue

            required_range = demanded_ranges[input_index][split_pos]
            input_tensors_by_index[input_index] = _crop_tile_if_needed(
                tile=source_tile,
                produced_range=produced_range,
                required_range=required_range,
                split_id=split_id,
                name_prefix=f"{node.name or node.op}_in{input_index}",
                scheduled_nodes=scheduled_nodes,
            )

        out_range = range_plan.output_ranges_by_node[local_index][split_pos]
        spatial_slice = None
        if node.op in {"Conv", "AveragePool"}:
            spatial_slice = range_plan.spatial_slices_by_node[local_index][split_pos]

        output_tile, new_node = _build_tiled_node(
            node=node,
            split_id=split_id,
            input_tensors_by_index=input_tensors_by_index,
            out_range=out_range,
            spatial_slice=spatial_slice,
        )
        tiles_by_local_index[local_index][split_pos] = output_tile
        scheduled_nodes.append(new_node)
        built_node_count += 1

    expected_built_count = len(group_cfg.execution_order)
    assert built_node_count == expected_built_count, (
        f"internal error: expected {expected_built_count} rewritten op nodes, got {built_node_count}"
    )

    for local_index, node_tiles in enumerate(tiles_by_local_index):
        assert all(tile is not None for tile in node_tiles), (
            f"execution_order has missing rewritten nodes for node index {group_info.node_range[0] + local_index}"
        )

    stitched_outputs_by_tensor_id = {}
    stitch_nodes = []
    seen_local_indices = set()
    for local_index, output_tensor in _iter_stitch_targets(group_info):
        if local_index in seen_local_indices:
            continue
        seen_local_indices.add(local_index)

        stitched_out, stitched_nodes = _build_stitched_output(
            local_index=local_index,
            output_tensor=output_tensor,
            produced_tiles=tiles_by_local_index[local_index],
            produced_ranges=range_plan.output_ranges_by_node[local_index],
            stitch_ranges=range_plan.stitch_ranges_by_local_index[local_index],
            split_keys=range_plan.split_keys,
            tile_count=group_cfg.tile_count,
        )
        stitched_outputs_by_tensor_id[id(output_tensor)] = stitched_out
        stitch_nodes.extend(stitched_nodes)

    ordered_nodes = entry_slice_nodes + scheduled_nodes + stitch_nodes
    _ensure_toposorted(ordered_nodes)
    return ordered_nodes, stitched_outputs_by_tensor_id


def _apply_group(graph, orig_nodes, group_info, group_cfg, new_nodes, stitched_outputs_by_tensor_id):
    node_a = orig_nodes[group_cfg.node_range[0]]
    node_b = orig_nodes[group_cfg.node_range[1]]
    start_pos = next(i for i, node in enumerate(graph.nodes) if node is node_a)
    end_pos = next(i for i, node in enumerate(graph.nodes) if node is node_b)
    group_node_ids = {id(node) for node in group_info.nodes}

    seen_tensor_ids = set()
    for output_tensor in _iter_stitched_output_tensors(group_info):
        tensor_id = id(output_tensor)
        if tensor_id in seen_tensor_ids:
            continue
        seen_tensor_ids.add(tensor_id)

        stitched_out = stitched_outputs_by_tensor_id[tensor_id]
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
