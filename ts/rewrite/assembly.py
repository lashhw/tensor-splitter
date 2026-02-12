from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import onnx_graphsurgeon as gs

from ..config import GroupConfig
from .analysis import GroupInfo
from .catalog import TileBlock
from .lowering import _build_tiled_op, _ensure_supported_op
from .naming import NameScope
from .tensor import _make_concat, _make_slice, _tensor_height
from .tiling import _partition_ranges


def _build_entry_tiles(
    name_scope: NameScope,
    entry: gs.Variable,
    tile_count: int,
    nodes: List[gs.Node],
) -> Tuple[List[gs.Variable], List[Tuple[int, int]]]:
    h_in = _tensor_height(entry)
    ranges = _partition_ranges(h_in, tile_count)
    tiles = []
    for start, end in ranges:
        tile = _make_slice(name_scope, entry, start, end, 2, nodes)
        tiles.append(tile)
    return tiles, ranges


def _build_group_tiles(
    name_scope: NameScope,
    group_info: GroupInfo,
    group_cfg: GroupConfig,
    node_index_map: Dict[int, int],
) -> Tuple[List[gs.Variable], List[gs.Node], List[TileBlock]]:
    nodes = []
    blocks = []

    for node in group_info.nodes:
        _ensure_supported_op(node)

    tiles, ranges = _build_entry_tiles(name_scope, group_info.entry_tensor, group_cfg.tile_count, nodes)

    for node in group_info.nodes:
        orig_index = node_index_map[id(node)]
        main_idx = group_info.main_input_index[id(node)]
        tiles, ranges, op_blocks = _build_tiled_op(
            name_scope,
            node,
            orig_index,
            tiles,
            ranges,
            group_cfg.tile_count,
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
    return _make_concat(name_scope, tiles, axis=2, nodes=nodes, shape_hint=shape_hint)
