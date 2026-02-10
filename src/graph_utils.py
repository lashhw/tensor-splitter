from . import gs


class GroupInfo:
    def __init__(self, indices, nodes, entry_tensor, exit_tensor, main_input_index):
        self.indices = indices
        self.nodes = nodes
        self.entry_tensor = entry_tensor
        self.exit_tensor = exit_tensor
        self.main_input_index = main_input_index


def _is_constant(tensor):
    return isinstance(tensor, gs.Constant)


def index_nodes(nodes):
    return {node: idx for idx, node in enumerate(nodes)}


def analyze_group(nodes, indices):
    a, b = indices
    if a < 0 or b >= len(nodes) or b < a:
        raise ValueError(f"invalid group indices {indices} for graph with {len(nodes)} nodes")

    group_nodes = nodes[a : b + 1]
    group_set = set(group_nodes)
    main_input_index = {}

    # Validate single-output constraint and linear chain.
    for node in group_nodes:
        if len(node.outputs) != 1:
            raise ValueError(
                f"node {node.name or node.op} must have a single output for v1 tiling"
            )

    first = group_nodes[0]
    var_inputs = [inp for inp in first.inputs if not _is_constant(inp)]
    entry_candidates = []
    for idx, inp in enumerate(first.inputs):
        if _is_constant(inp):
            continue
        producers = [p for p in inp.inputs if p in group_set]
        if producers:
            raise ValueError(
                f"group entry node has input produced within group: {inp.name}"
            )
        entry_candidates.append((idx, inp))

    if len(entry_candidates) != 1:
        raise ValueError(
            "group entry node must have exactly one non-constant input from outside the group"
        )
    main_input_index[first] = entry_candidates[0][0]
    entry_tensor = entry_candidates[0][1]

    # Validate remaining nodes.
    for prev, node in zip(group_nodes[:-1], group_nodes[1:]):
        var_inputs = [inp for inp in node.inputs if not _is_constant(inp)]
        from_group = []
        for idx, inp in enumerate(node.inputs):
            if _is_constant(inp):
                continue
            producers = [p for p in inp.inputs if p in group_set]
            if producers:
                from_group.append((idx, inp, producers))

        if len(from_group) != 1:
            raise ValueError(
                f"node {node.name or node.op} must have exactly one data input from within group"
            )
        idx, main_inp, producers = from_group[0]
        if prev not in producers:
            raise ValueError(
                f"node {node.name or node.op} data input must come from previous node in group"
            )
        main_input_index[node] = idx

        for inp in var_inputs:
            if inp is main_inp:
                continue
            # Disallow external variable inputs for v1.
            if not _is_constant(inp):
                raise ValueError(
                    f"node {node.name or node.op} has unsupported external variable input {inp.name}"
                )

    # Validate intermediate outputs are only consumed by next node.
    for node, nxt in zip(group_nodes[:-1], group_nodes[1:]):
        out_tensor = node.outputs[0]
        for consumer in out_tensor.outputs:
            if consumer is nxt:
                continue
            if consumer in group_set:
                raise ValueError(
                    f"node {node.name or node.op} output is consumed by another node in group"
                )
            raise ValueError(
                f"node {node.name or node.op} output is consumed outside the group; v1 requires linear chain"
            )

    exit_tensor = group_nodes[-1].outputs[0]
    return GroupInfo(
        indices=indices,
        nodes=group_nodes,
        entry_tensor=entry_tensor,
        exit_tensor=exit_tensor,
        main_input_index=main_input_index,
    )
