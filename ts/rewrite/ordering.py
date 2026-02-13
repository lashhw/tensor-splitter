from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import onnx_graphsurgeon as gs

from .catalog import TileBlock


def _build_execution_order_map(
    blocks: Sequence[TileBlock],
    schedule: Sequence[Tuple[int, int]],
) -> Dict[int, int]:
    schedule_pos = {pair: idx for idx, pair in enumerate(schedule)}
    block_pairs = {(block.orig_index, block.tile_id) for block in blocks}
    schedule_pairs = set(schedule_pos)
    if block_pairs != schedule_pairs:
        missing = sorted(block_pairs - schedule_pairs)
        extra = sorted(schedule_pairs - block_pairs)
        assert False, (
            "execution_order does not match rewritten block set. "
            f"Missing: {missing}, Extra: {extra}"
        )

    order_map = {}
    for block in blocks:
        order = schedule_pos.get((block.orig_index, block.tile_id))
        assert order is not None, (
            "execution_order is missing entry for rewritten block "
            f"{(block.orig_index, block.tile_id)}"
        )
        block.assign_order(order_map, order)
    return order_map


def _order_by_execution_order(nodes: Sequence[gs.Node], order_map: Dict[int, int]) -> List[gs.Node]:
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


def _ensure_toposorted(nodes: Sequence[gs.Node]) -> None:
    index = {id(node): node_idx for node_idx, node in enumerate(nodes)}
    for node_idx, node in enumerate(nodes):
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_idx = index.get(id(producer))
                if producer_idx is None:
                    continue
                assert producer_idx <= node_idx, "graph nodes are not topologically sorted"
