from __future__ import annotations

from typing import Dict, List, Tuple

import onnx_graphsurgeon as gs

from .catalog import TileBlock


def _build_execution_order_map(
    blocks: List[TileBlock],
    schedule: List[Tuple[int, int]],
) -> Dict[int, int]:
    schedule_pos = {pair: idx for idx, pair in enumerate(schedule)}
    order_map = {}
    for block in blocks:
        key = (block.orig_index, block.tile_id)
        assert key in schedule_pos, f"execution_order is missing entry {key}"
        block.assign_order(order_map, schedule_pos[key])
    return order_map


def _order_by_execution_order(nodes: List[gs.Node], order_map: Dict[int, int]) -> List[gs.Node]:
    indexed = list(enumerate(nodes))
    scheduled = [
        (order_map[id(node)], orig_pos, node)
        for orig_pos, node in indexed
        if id(node) in order_map
    ]
    scheduled.sort(key=lambda item: (item[0], item[1]))
    scheduled_iter = iter(node for _, _, node in scheduled)

    ordered = []
    for _, node in indexed:
        if id(node) in order_map:
            ordered.append(next(scheduled_iter))
        else:
            ordered.append(node)

    try:
        _ensure_toposorted(ordered)
    except AssertionError:
        assert False, "execution_order is not topologically valid for rewritten graph"
    return ordered


def _ensure_toposorted(nodes: List[gs.Node]) -> None:
    index = {id(node): node_idx for node_idx, node in enumerate(nodes)}
    for node_idx, node in enumerate(nodes):
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_idx = index.get(id(producer))
                if producer_idx is None:
                    continue
                assert producer_idx <= node_idx, "graph nodes are not topologically sorted"
