from __future__ import annotations

from typing import List

import onnx_graphsurgeon as gs

from .types import GroupAnalysis


def _replace_tensor_uses(graph: gs.Graph, old: gs.Variable, new: gs.Variable) -> None:
    for consumer in list(old.outputs):
        for idx, inp in enumerate(consumer.inputs):
            if inp is old:
                consumer.inputs[idx] = new

    for idx, out in enumerate(graph.outputs):
        if out is old:
            graph.outputs[idx] = new


def apply_group_rewrite(
    graph: gs.Graph,
    original_nodes: List[gs.Node],
    group: GroupAnalysis,
    replacement_nodes: List[gs.Node],
    replacement_output: gs.Variable,
) -> None:
    start_node = original_nodes[group.node_range[0]]
    end_node = original_nodes[group.node_range[1]]
    start_pos = next(i for i, node in enumerate(graph.nodes) if node is start_node)
    end_pos = next(i for i, node in enumerate(graph.nodes) if node is end_node)

    _replace_tensor_uses(graph, group.exit_tensor, replacement_output)

    for node in group.nodes:
        node.inputs = []
        node.outputs = []

    graph.nodes = graph.nodes[:start_pos] + replacement_nodes + graph.nodes[end_pos + 1 :]
