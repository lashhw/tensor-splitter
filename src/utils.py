import onnx_graphsurgeon as gs


class GroupInfo:
    def __init__(self, indices, nodes, entry_tensor, exit_tensor, main_input_index):
        self.indices = indices
        self.nodes = nodes
        self.entry_tensor = entry_tensor
        self.exit_tensor = exit_tensor
        self.main_input_index = main_input_index


def _is_constant(tensor):
    return isinstance(tensor, gs.Constant)


def _node_label(node):
    return node.name or node.op


def analyze_group(nodes, indices):
    a, b = indices
    if a < 0 or b >= len(nodes) or b < a:
        raise ValueError(f"invalid group indices {indices} for graph with {len(nodes)} nodes")

    group_nodes = nodes[a : b + 1]
    group_set = set(group_nodes)
    main_input_index = {}

    # Validate single-output constraint and linear chain
    for node in group_nodes:
        if len(node.outputs) != 1:
            raise ValueError(
                f"node {_node_label(node)} must have a single output"
            )

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
    main_input_index[first] = entry_candidates[0][0]
    entry_tensor = entry_candidates[0][1]

    # Validate remaining nodes
    for prev, node in zip(group_nodes[:-1], group_nodes[1:]):
        main_idx = None
        for idx, inp in enumerate(node.inputs):
            if _is_constant(inp):
                continue
            producers = [p for p in inp.inputs if p in group_set]
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
                # Disallow external variable inputs
                raise ValueError(
                    f"node {_node_label(node)} has unsupported external variable input {inp.name}"
                )

        if main_idx is None:
            raise ValueError(
                f"node {_node_label(node)} must have exactly one data input from within group"
            )
        main_input_index[node] = main_idx

    # Validate intermediate outputs are only consumed by next node
    for node, nxt in zip(group_nodes[:-1], group_nodes[1:]):
        out_tensor = node.outputs[0]
        for consumer in out_tensor.outputs:
            if consumer is nxt:
                continue
            if consumer in group_set:
                raise ValueError(
                    f"node {_node_label(node)} output is consumed by another node in group"
                )
            raise ValueError(
                f"node {_node_label(node)} output is consumed outside the group; v1 requires linear chain"
            )

    exit_tensor = group_nodes[-1].outputs[0]
    return GroupInfo(
        indices=indices,
        nodes=group_nodes,
        entry_tensor=entry_tensor,
        exit_tensor=exit_tensor,
        main_input_index=main_input_index,
    )
