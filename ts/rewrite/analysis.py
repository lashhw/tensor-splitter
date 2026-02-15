from __future__ import annotations

from typing import List, Tuple

import onnx_graphsurgeon as gs

from .tensor import is_constant
from .types import GroupAnalysis


def _entry_tensor(first_node: gs.Node) -> Tuple[int, gs.Variable]:
    entry_candidates = [
        (idx, inp)
        for idx, inp in enumerate(first_node.inputs)
        if not is_constant(inp)
    ]
    assert len(entry_candidates) == 1, (
        "group entry node must have exactly one non-constant input from outside the group"
    )

    main_idx, entry = entry_candidates[0]
    assert isinstance(entry, gs.Variable), (
        f"group entry tensor must be a Variable; got {type(entry).__name__}"
    )
    return main_idx, entry


def _main_input_from_previous(node: gs.Node, prev: gs.Node) -> int:
    main_idx = None
    for idx, inp in enumerate(node.inputs):
        if is_constant(inp):
            continue

        producers = list(inp.inputs)
        assert len(producers) == 1, (
            f"node {node.name} input {inp.name} must have exactly one producer; "
            f"found {len(producers)}"
        )
        producer = producers[0]

        assert main_idx is None, (
            f"node {node.name} must have exactly one data input from within group"
        )
        assert producer is prev, (
            f"node {node.name} data input must come from previous node in group"
        )
        main_idx = idx

    assert main_idx is not None, (
        f"node {node.name} must have exactly one data input from within group"
    )
    return main_idx


def _validate_linear_outputs(group_nodes: List[gs.Node]) -> None:
    for node in group_nodes:
        assert len(node.outputs) == 1, f"node {node.name} must have a single output"

    for node, nxt in zip(group_nodes[:-1], group_nodes[1:]):
        for consumer in node.outputs[0].outputs:
            assert consumer is nxt, (
                f"node {node.name} output has extra consumer {consumer.name}; "
                "rewrite requires a linear chain where each node feeds only the next node"
            )


def analyze_group(nodes: List[gs.Node], node_range: Tuple[int, int]) -> GroupAnalysis:
    a, b = node_range
    assert not (a < 0 or b >= len(nodes) or b < a), (
        f"invalid group node_range {node_range} for graph with {len(nodes)} nodes"
    )

    group_nodes = nodes[a : b + 1]
    entry_main_idx, entry_tensor = _entry_tensor(group_nodes[0])
    main_input_indices = [entry_main_idx]
    for prev, node in zip(group_nodes[:-1], group_nodes[1:]):
        main_input_indices.append(_main_input_from_previous(node, prev))

    _validate_linear_outputs(group_nodes)
    exit_tensor = group_nodes[-1].outputs[0]
    assert isinstance(exit_tensor, gs.Variable), (
        f"group exit tensor must be a Variable; got {type(exit_tensor).__name__}"
    )

    return GroupAnalysis(
        node_range=node_range,
        nodes=group_nodes,
        entry_tensor=entry_tensor,
        exit_tensor=exit_tensor,
        main_input_indices=main_input_indices,
    )
