from __future__ import annotations

import heapq
from typing import Dict, List, Sequence

import onnx_graphsurgeon as gs


def _toposort_with_priority(nodes: Sequence[gs.Node], priority: Dict[int, int]) -> List[gs.Node]:
    node_ids = [id(node) for node in nodes]
    node_set = set(node_ids)
    node_by_id = {id(node): node for node in nodes}
    adj = {node_id: [] for node_id in node_ids}
    indeg = {node_id: 0 for node_id in node_ids}

    for node_id, node in zip(node_ids, nodes):
        for inp in node.inputs:
            for prod in inp.inputs:
                prod_id = id(prod)
                if prod_id in node_set:
                    adj[prod_id].append(node_id)
                    indeg[node_id] += 1

    order = {node_id: idx for idx, node_id in enumerate(node_ids)}
    heap = []
    for node_id, deg in indeg.items():
        if deg == 0:
            heapq.heappush(heap, (priority.get(node_id, 10**9), order[node_id], node_id))

    result = []
    while heap:
        _, _, node_id = heapq.heappop(heap)
        result.append(node_by_id[node_id])
        for nxt in adj[node_id]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, (priority.get(nxt, 10**9), order[nxt], nxt))

    if len(result) != len(nodes):
        raise RuntimeError("cycle detected while ordering rewritten nodes")
    return result


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
