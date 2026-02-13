from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import onnx_graphsurgeon as gs

from ..config import GroupConfig
from .analysis import GroupInfo
from .catalog import TileBlock
from .conv import _conv_params
from .lowering import _build_tiled_op, _ensure_supported_op
from .naming import NameScope
from .tensor import _make_concat, _make_slice, _tensor_height
from .tiling import _conv_input_slice_for_output, _partition_ranges


def _plan_stage_ranges(
    group_info: GroupInfo,
    tile_count: int,
) -> List[List[Tuple[int, int]]]:
    """
    Build per-stage required ranges with backward propagation from group output.

    stage_ranges[i] is the required range list for the tensor entering node i.
    stage_ranges[-1] is the required range list for group output (used for final concat).
    """
    stage_count = len(group_info.nodes) + 1
    stage_ranges: List[List[Tuple[int, int]]] = [[] for _ in range(stage_count)]
    stage_ranges[-1] = _partition_ranges(_tensor_height(group_info.exit_tensor), tile_count)

    for stage_idx in range(stage_count - 2, -1, -1):
        node = group_info.nodes[stage_idx]
        out_ranges = stage_ranges[stage_idx + 1]
        main_idx = group_info.main_input_index[id(node)]

        if node.op != "Conv":
            h_in = _tensor_height(node.inputs[main_idx])
            h_out = _tensor_height(node.outputs[0])
            if h_in != h_out:
                raise RuntimeError(
                    f"node {node.name or node.op} changes height ({h_in}->{h_out}) but is not Conv"
                )
            stage_ranges[stage_idx] = list(out_ranges)
            continue

        kernel_shape, strides, dilations, pads = _conv_params(node)
        k_h = kernel_shape[0]
        s_h = strides[0]
        d_h = dilations[0]
        pad_top = pads[0]
        h_in = _tensor_height(node.inputs[main_idx])

        in_ranges: List[Tuple[int, int]] = []
        for y0, y1 in out_ranges:
            slice_info = _conv_input_slice_for_output(y0, y1, s_h, d_h, k_h, pad_top, h_in)
            in_ranges.append((slice_info.slice_start, slice_info.slice_end))
        stage_ranges[stage_idx] = in_ranges

    return stage_ranges


def _build_entry_tiles(
    name_scope: NameScope,
    entry: gs.Variable,
    entry_ranges: Sequence[Tuple[int, int]],
) -> Tuple[List[gs.Variable], List[gs.Node]]:
    nodes: List[gs.Node] = []
    h_in = _tensor_height(entry)
    tiles = []
    for tile_id, (start, end) in enumerate(entry_ranges):
        if start < 0 or end < 0 or start > end or end > h_in:
            raise RuntimeError(
                f"invalid entry range for tile {tile_id}: [{start},{end}) with input height {h_in}"
            )
        tile, tile_node = _make_slice(name_scope, entry, start, end, 2)
        nodes.append(tile_node)
        tiles.append(tile)
    return tiles, nodes


def _build_group_tiles(
    name_scope: NameScope,
    group_info: GroupInfo,
    group_cfg: GroupConfig,
    node_index_map: Dict[int, int],
) -> Tuple[List[gs.Variable], List[gs.Node], List[TileBlock]]:
    for node in group_info.nodes:
        _ensure_supported_op(node)
    stage_ranges = _plan_stage_ranges(group_info, group_cfg.tile_count)
    tiles, nodes = _build_entry_tiles(name_scope, group_info.entry_tensor, stage_ranges[0])

    blocks = []
    for stage_idx, node in enumerate(group_info.nodes):
        orig_index = node_index_map[id(node)]
        main_idx = group_info.main_input_index[id(node)]
        in_ranges = stage_ranges[stage_idx]
        out_ranges = stage_ranges[stage_idx + 1]
        tiles, op_blocks = _build_tiled_op(
            name_scope,
            node,
            orig_index,
            tiles,
            in_ranges,
            out_ranges,
            nodes,
            main_idx,
        )
        blocks.extend(op_blocks)

    return tiles, nodes, blocks


def _build_group_output(
    name_scope: NameScope,
    tiles: Sequence[gs.Variable],
    shape_hint: Sequence[Any] | None,
    nodes: List[gs.Node],
) -> gs.Variable:
    out, concat_node = _make_concat(name_scope, tiles, axis=2, shape_hint=shape_hint)
    nodes.append(concat_node)
    return out
