from dataclasses import dataclass

import onnx_graphsurgeon as gs


@dataclass(frozen=True)
class GroupInfo:
    """Static analysis output for a linear node range selected for rewriting."""

    indices: tuple[int, int]
    nodes: list[gs.Node]
    entry_tensor: gs.Tensor
    exit_tensor: gs.Tensor
    main_input_index: dict[int, int]


def _is_constant(tensor: gs.Tensor) -> bool:
    return isinstance(tensor, gs.Constant)


def _node_label(node: gs.Node) -> str:
    return node.name or node.op


def _validate_single_outputs(group_nodes: list[gs.Node]) -> None:
    for node in group_nodes:
        if len(node.outputs) != 1:
            raise ValueError(f"node {_node_label(node)} must have a single output")


def _resolve_entry(first: gs.Node) -> tuple[int, gs.Tensor]:
    entry_candidates = []
    for idx, inp in enumerate(first.inputs):
        if _is_constant(inp):
            continue
        entry_candidates.append((idx, inp))

    if len(entry_candidates) != 1:
        raise ValueError(
            "group entry node must have exactly one non-constant input from outside the group"
        )
    return entry_candidates[0]


def _validate_node_data_flow(
    group_nodes: list[gs.Node],
    group_node_ids: set[int],
    main_input_index: dict[int, int],
) -> None:
    for prev, node in zip(group_nodes[:-1], group_nodes[1:]):
        main_idx = None
        for idx, inp in enumerate(node.inputs):
            if _is_constant(inp):
                continue
            producers = [p for p in inp.inputs if id(p) in group_node_ids]
            if producers:
                if main_idx is not None:
                    raise ValueError(
                        f"node {_node_label(node)} must have exactly one data input from within group"
                    )
                if prev not in producers:
                    raise ValueError(
                        f"node {_node_label(node)} data input must come from previous node in group"
                    )
                main_idx = idx
            else:
                raise ValueError(
                    f"node {_node_label(node)} has unsupported external variable input {inp.name}"
                )

        if main_idx is None:
            raise ValueError(
                f"node {_node_label(node)} must have exactly one data input from within group"
            )
        main_input_index[id(node)] = main_idx


def _validate_intermediate_consumers(group_nodes: list[gs.Node], group_node_ids: set[int]) -> None:
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
                f"node {_node_label(node)} output is consumed outside the group; v1 requires linear chain"
            )


def analyze_group(nodes: list[gs.Node], indices: tuple[int, int]) -> GroupInfo:
    """Validate a contiguous node range and derive rewrite metadata."""
    a, b = indices
    if a < 0 or b >= len(nodes) or b < a:
        raise ValueError(f"invalid group indices {indices} for graph with {len(nodes)} nodes")

    group_nodes = nodes[a : b + 1]
    group_node_ids = {id(node) for node in group_nodes}
    main_input_index: dict[int, int] = {}

    _validate_single_outputs(group_nodes)

    entry_input_index, entry_tensor = _resolve_entry(group_nodes[0])
    main_input_index[id(group_nodes[0])] = entry_input_index

    _validate_node_data_flow(group_nodes, group_node_ids, main_input_index)
    _validate_intermediate_consumers(group_nodes, group_node_ids)

    return GroupInfo(
        indices=indices,
        nodes=group_nodes,
        entry_tensor=entry_tensor,
        exit_tensor=group_nodes[-1].outputs[0],
        main_input_index=main_input_index,
    )
