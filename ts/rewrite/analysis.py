from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import onnx_graphsurgeon as gs


@dataclass(frozen=True)
class GroupInfo:
    node_range: Tuple[int, int]
    nodes: Sequence[gs.Node]
    entry_tensor: gs.Variable
    exit_tensor: gs.Variable
    main_input_indices: List[int]


def _is_constant(tensor: gs.Tensor) -> bool:
    return isinstance(tensor, gs.Constant)


def _node_label(node: gs.Node) -> str:
    return node.name or node.op


def _analyze_group(nodes: Sequence[gs.Node], node_range: Tuple[int, int]) -> GroupInfo:
    a, b = node_range
    assert not (a < 0 or b >= len(nodes) or b < a), (
        f"invalid group node_range {node_range} for graph with {len(nodes)} nodes"
    )

    group_nodes = nodes[a : b + 1]
    main_input_indices = []

    first = group_nodes[0]
    entry_candidates = []
    for idx, inp in enumerate(first.inputs):
        if _is_constant(inp):
            continue
        entry_candidates.append((idx, inp))

    assert len(entry_candidates) == 1, (
        "group entry node must have exactly one non-constant input from outside the group"
    )

    main_input_indices.append(entry_candidates[0][0])
    entry_tensor = entry_candidates[0][1]

    for prev, node in zip(group_nodes[:-1], group_nodes[1:]):
        main_idx = None

        for idx, inp in enumerate(node.inputs):
            if _is_constant(inp):
                continue

            producers = list(inp.inputs)
            assert len(producers) == 1, (
                f"node {_node_label(node)} input {inp.name} must have exactly one producer; "
                f"found {len(producers)}"
            )
            producer = producers[0]

            if main_idx is not None:
                assert False, (
                    f"node {_node_label(node)} must have exactly one data input from within group"
                )
            main_idx = idx

            assert id(producer) == id(prev), (
                f"node {_node_label(node)} data input must come from previous node in group"
            )

        if main_idx is None:
            assert False, f"node {_node_label(node)} must have exactly one data input from within group"
        main_input_indices.append(main_idx)

    for node in group_nodes:
        assert len(node.outputs) == 1, f"node {_node_label(node)} must have a single output"
    exit_tensor = group_nodes[-1].outputs[0]

    for node, nxt in zip(group_nodes[:-1], group_nodes[1:]):
        out_tensor = node.outputs[0]
        for consumer in out_tensor.outputs:
            assert consumer is nxt, (
                f"node {_node_label(node)} output has extra consumer {_node_label(consumer)}; "
                "rewrite requires a linear chain where each node feeds only the next node"
            )

    return GroupInfo(
        node_range=node_range,
        nodes=group_nodes,
        entry_tensor=entry_tensor,
        exit_tensor=exit_tensor,
        main_input_indices=main_input_indices,
    )
