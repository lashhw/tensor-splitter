from __future__ import annotations

from typing import Dict, List, Sequence

import onnx_graphsurgeon as gs


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
    except ValueError as exc:
        raise ValueError("execution_order is not topologically valid for rewritten graph") from exc
    return ordered


def _ensure_toposorted(nodes: Sequence[gs.Node]) -> None:
    index = {id(node): node_idx for node_idx, node in enumerate(nodes)}
    for node_idx, node in enumerate(nodes):
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_idx = index.get(id(producer))
                if producer_idx is None:
                    continue
                if producer_idx > node_idx:
                    raise ValueError("graph nodes are not topologically sorted")


def _replace_tensor_consumers(graph: gs.Graph, old: gs.Tensor, new: gs.Tensor) -> None:
    for consumer in list(old.outputs):
        for idx, inp in enumerate(consumer.inputs):
            if inp is old:
                consumer.inputs[idx] = new

    for idx, out in enumerate(graph.outputs):
        if out is old:
            graph.outputs[idx] = new
