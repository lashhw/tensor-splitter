from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import onnx_graphsurgeon as gs


@dataclass(frozen=True)
class GroupInfo:
    node_range: Tuple[int, int]
    nodes: Sequence[gs.Node]
    entry_tensor: gs.Variable
    exit_tensor: gs.Variable
    main_input_index: Dict[int, int]


def _is_constant(tensor: gs.Tensor) -> bool:
    return isinstance(tensor, gs.Constant)


def _node_label(node: gs.Node) -> str:
    return node.name or node.op


def _producers_in_group(tensor: gs.Tensor, group_node_ids: set[int]) -> list[gs.Node]:
    return [producer for producer in tensor.inputs if id(producer) in group_node_ids]


def _has_previous_producer(producers: Sequence[gs.Node], previous: gs.Node) -> bool:
    previous_id = id(previous)
    return any(id(producer) == previous_id for producer in producers)


def _analyze_group(nodes: Sequence[gs.Node], node_range: Tuple[int, int]) -> GroupInfo:
    a, b = node_range
    if a < 0 or b >= len(nodes) or b < a:
        raise ValueError(f"invalid group node_range {node_range} for graph with {len(nodes)} nodes")

    group_nodes = nodes[a : b + 1]
    group_node_ids = {id(node) for node in group_nodes}
    main_input_index = {}

    for node in group_nodes:
        if len(node.outputs) != 1:
            raise ValueError(f"node {_node_label(node)} must have a single output")

    first = group_nodes[0]
    entry_candidates = []
    for idx, inp in enumerate(first.inputs):
        if _is_constant(inp):
            continue
        entry_candidates.append((idx, inp))

    if len(entry_candidates) != 1:
        raise ValueError(
            "group entry node must have exactly one non-constant input from outside the group"
        )

    main_input_index[id(first)] = entry_candidates[0][0]
    entry_tensor = entry_candidates[0][1]

    for prev, node in zip(group_nodes[:-1], group_nodes[1:]):
        main_idx = None

        for idx, inp in enumerate(node.inputs):
            if _is_constant(inp):
                continue

            producers = list(inp.inputs)
            if len(producers) != 1:
                raise ValueError(
                    f"node {_node_label(node)} input {inp.name} must have exactly one producer; "
                    f"found {len(producers)}"
                )

            group_producers = _producers_in_group(inp, group_node_ids)
            if not group_producers:
                raise ValueError(
                    f"node {_node_label(node)} has non-constant input {inp.name} that is produced "
                    "outside the group; all non-constant inputs must come from nodes in the group"
                )

            if main_idx is not None:
                raise ValueError(
                    f"node {_node_label(node)} must have exactly one data input from within group"
                )
            main_idx = idx

            if not _has_previous_producer(group_producers, prev):
                raise ValueError(
                    f"node {_node_label(node)} data input must come from previous node in group"
                )

        if main_idx is None:
            raise ValueError(
                f"node {_node_label(node)} must have exactly one data input from within group"
            )
        main_input_index[id(node)] = main_idx

    for node, nxt in zip(group_nodes[:-1], group_nodes[1:]):
        out_tensor = node.outputs[0]
        for consumer in out_tensor.outputs:
            if consumer is nxt:
                continue
            if id(consumer) in group_node_ids:
                raise ValueError(
                    f"node {_node_label(node)} output is consumed by another node in group"
                )
            raise ValueError(
                f"node {_node_label(node)} output is consumed outside the group; rewrite requires a linear chain"
            )

    exit_tensor = group_nodes[-1].outputs[0]
    return GroupInfo(
        node_range=node_range,
        nodes=group_nodes,
        entry_tensor=entry_tensor,
        exit_tensor=exit_tensor,
        main_input_index=main_input_index,
    )
