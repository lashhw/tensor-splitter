from __future__ import annotations

from typing import Dict, List, Tuple

import onnx_graphsurgeon as gs

from .types import TileBlock


def ensure_topologically_sorted(nodes: List[gs.Node]) -> None:
    index = {id(node): node_idx for node_idx, node in enumerate(nodes)}
    for node_idx, node in enumerate(nodes):
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_idx = index.get(id(producer))
                if producer_idx is None:
                    continue
                assert producer_idx <= node_idx, "graph nodes are not topologically sorted"


def build_execution_order_map(
    blocks: List[TileBlock],
    schedule: List[Tuple[int, int]],
    final_node: gs.Node,
) -> Dict[int, int]:
    schedule_pos = {pair: idx for idx, pair in enumerate(schedule)}
    order_map = {}
    for block in blocks:
        key = (block.orig_index, block.tile_id)
        assert key in schedule_pos, f"missing schedule entry for block {key}"
        order_map[id(block.node)] = schedule_pos[key]
    order_map[id(final_node)] = len(schedule)
    return order_map


def order_by_execution_order(nodes: List[gs.Node], order_map: Dict[int, int]) -> List[gs.Node]:
    indexed = list(enumerate(nodes))
    scheduled = [
        (order_map[id(node)], orig_pos, node)
        for orig_pos, node in indexed
    ]
    scheduled.sort(key=lambda item: (item[0], item[1]))
    ordered = [node for _, _, node in scheduled]

    ensure_topologically_sorted(ordered)
    return ordered
