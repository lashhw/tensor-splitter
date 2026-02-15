from __future__ import annotations

from typing import Dict, List

import onnx_graphsurgeon as gs

from ..config import GroupConfig
from .catalog import ensure_supported_op
from .conv import conv_params
from .lowering import build_tiled_op
from .naming import NameScope
from .tensor import make_concat, make_slice, tensor_height
from .tiling import conv_input_slice_for_output, partition_ranges
from .types import ConvSlice, GroupAnalysis, GroupBuild, GroupPlan, HeightRange, StagePlan, TileBlock


def plan_group(group: GroupAnalysis, tile_count: int) -> GroupPlan:
    output_ranges = partition_ranges(tensor_height(group.exit_tensor), tile_count)
    stage_plans_reversed: List[StagePlan] = []
    downstream_ranges = output_ranges

    for stage_idx in range(len(group.nodes) - 1, -1, -1):
        node = group.nodes[stage_idx]
        main_idx = group.main_input_indices[stage_idx]

        if node.op != "Conv":
            input_height = tensor_height(node.inputs[main_idx])
            output_height = tensor_height(node.outputs[0])
            assert input_height == output_height, (
                f"node {node.name or node.op} changes height ({input_height}->{output_height}) but is not Conv"
            )
            stage_input_ranges = list(downstream_ranges)
            stage_plans_reversed.append(
                StagePlan(
                    input_ranges=stage_input_ranges,
                    output_ranges=list(downstream_ranges),
                )
            )
            downstream_ranges = stage_input_ranges
            continue

        params = conv_params(node)
        input_height = tensor_height(node.inputs[main_idx])
        in_ranges: List[HeightRange] = []
        conv_slices: List[ConvSlice] = []
        for y0, y1 in downstream_ranges:
            slice_info = conv_input_slice_for_output(
                y0,
                y1,
                params.strides[0],
                params.kernel_shape[0],
                params.pads[0],
                input_height,
            )
            in_ranges.append((slice_info.slice_start, slice_info.slice_end))
            conv_slices.append(slice_info)

        stage_plans_reversed.append(
            StagePlan(
                input_ranges=in_ranges,
                output_ranges=list(downstream_ranges),
                conv_slices=conv_slices,
                conv_base_pads=list(params.pads),
            )
        )
        downstream_ranges = in_ranges

    stage_plans = list(reversed(stage_plans_reversed))
    assert stage_plans, "group must contain at least one node"
    return GroupPlan(
        entry_ranges=stage_plans[0].input_ranges,
        stage_plans=stage_plans,
        output_ranges=output_ranges,
    )


def _build_entry_tiles(
    name_scope: NameScope,
    entry: gs.Variable,
    entry_ranges: List[HeightRange],
) -> tuple[List[gs.Variable], List[gs.Node]]:
    tiles = []
    nodes = []
    for start, end in entry_ranges:
        tile, node = make_slice(name_scope, entry, start, end, 2)
        tiles.append(tile)
        nodes.append(node)
    return tiles, nodes


def build_group_tiles(
    name_scope: NameScope,
    group: GroupAnalysis,
    group_cfg: GroupConfig,
    node_index_map: Dict[int, int],
) -> GroupBuild:
    for node in group.nodes:
        ensure_supported_op(node)

    plan = plan_group(group, group_cfg.tile_count)
    tiles, nodes = _build_entry_tiles(name_scope, group.entry_tensor, plan.entry_ranges)

    first_orig_index = node_index_map[id(group.nodes[0])]
    blocks = [
        TileBlock(orig_index=first_orig_index, tile_id=tile_id, node=slice_node)
        for tile_id, slice_node in enumerate(nodes)
    ]

    for stage_plan, node, main_idx in zip(plan.stage_plans, group.nodes, group.main_input_indices):
        orig_index = node_index_map[id(node)]
        stage_build = build_tiled_op(
            name_scope=name_scope,
            node=node,
            orig_index=orig_index,
            tiles=tiles,
            out_ranges=stage_plan.output_ranges,
            main_idx=main_idx,
            conv_slices=stage_plan.conv_slices,
            conv_base_pads=stage_plan.conv_base_pads,
        )
        tiles = stage_build.output_tiles
        nodes.extend(stage_build.nodes)
        blocks.extend(stage_build.blocks)

    concat_out, concat_node = make_concat(name_scope, tiles, 2, group.exit_tensor.shape)
    nodes.append(concat_node)
    return GroupBuild(
        nodes=nodes,
        blocks=blocks,
        concat_output=concat_out,
        concat_node=concat_node,
    )
