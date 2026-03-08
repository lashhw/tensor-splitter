from collections import namedtuple

from .tensor import _is_constant

SUPPORTED_NON_CONV_OPS = {
    "Relu",
    "BatchNormalization",
    "Add",
    "Concat",
    "AveragePool",
    "Constant",
    "Reshape",
    "Transpose",
}
SUPPORTED_GROUP_OPS = SUPPORTED_NON_CONV_OPS | {"Conv"}

_InputSource = namedtuple(
    "_InputSource",
    ["input_index", "kind", "producer_local_index", "entry_key"],
)

_NodeSpec = namedtuple(
    "_NodeSpec",
    ["node", "local_index", "data_input_indices", "input_sources"],
)

_BoundaryOutputSpec = namedtuple(
    "_BoundaryOutputSpec",
    ["local_index", "output_tensor"],
)

_GroupInfo = namedtuple(
    "_GroupInfo",
    [
        "node_range",
        "nodes",
        "entry_tensors",
        "exit_tensor",
        "boundary_output_specs",
        "node_specs",
        "node_spec_by_id",
    ],
)


def _ensure_toposorted(nodes):
    index = {id(node): node_idx for node_idx, node in enumerate(nodes)}
    for node_idx, node in enumerate(nodes):
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_idx = index.get(id(producer))
                if producer_idx is None:
                    continue
                assert producer_idx < node_idx, "graph nodes are not topologically sorted"


def _ensure_supported_op(node):
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op {node.op} for tiled rewrite"
    if node.op == "Concat":
        axis = node.attrs.get("axis")
        assert axis is not None, f"Concat node {node.name} must define axis"
        assert axis == 1, f"Concat axis must be 1 for tiled rewrite; got {axis}"
    if node.op == "Transpose":
        perm = node.attrs.get("perm")
        assert isinstance(perm, list) and len(perm) >= 2, (
            f"Transpose node {node.name} must define perm with rank >= 2"
        )
        # Keep H/W as the trailing axes so tile ranges remain spatially aligned.
        rank = len(perm)
        assert perm[-2] == rank - 2 and perm[-1] == rank - 1, (
            f"Transpose node {node.name} perm must keep trailing H/W axes unchanged; got {perm}"
        )


def _collect_node_specs(group_nodes):
    local_index_by_id = {id(node): local_index for local_index, node in enumerate(group_nodes)}
    node_specs = []
    external_tensors = {}
    internal_edges = []
    join_local_indices = []

    for local_index, node in enumerate(group_nodes):
        _ensure_supported_op(node)
        assert len(node.outputs) == 1, f"node {node.name} must have a single output"

        data_input_indices = []
        input_sources = {}
        internal_input_count = 0

        for input_index, tensor in enumerate(node.inputs):
            producers = list(tensor.inputs)
            if _is_constant(tensor):
                continue

            assert len(producers) <= 1, (
                f"node {node.name} input {tensor.name} must have at most one producer; "
                f"found {len(producers)}"
            )

            if producers and producers[0].op == "Constant":
                producer_local_index = local_index_by_id.get(id(producers[0]))
                if producer_local_index is not None:
                    input_sources[input_index] = _InputSource(
                        input_index=input_index,
                        kind="node",
                        producer_local_index=producer_local_index,
                        entry_key=None,
                    )
                continue

            data_input_indices.append(input_index)
            producer_local_index = None
            if producers:
                producer_local_index = local_index_by_id.get(id(producers[0]))

            if producer_local_index is None:
                entry_key = id(tensor)
                input_sources[input_index] = _InputSource(
                    input_index=input_index,
                    kind="entry",
                    producer_local_index=None,
                    entry_key=entry_key,
                )
                external_tensors[entry_key] = tensor
            else:
                input_sources[input_index] = _InputSource(
                    input_index=input_index,
                    kind="node",
                    producer_local_index=producer_local_index,
                    entry_key=None,
                )
                internal_edges.append((producer_local_index, local_index))
                internal_input_count += 1

        if node.op == "Constant":
            assert not data_input_indices, (
                f"Constant node {node.name} must not have non-constant data inputs"
            )
        else:
            assert data_input_indices, f"node {node.name} must have at least one non-constant data input"

        if internal_input_count >= 2:
            join_local_indices.append(local_index)

        node_specs.append(
            _NodeSpec(
                node=node,
                local_index=local_index,
                data_input_indices=tuple(data_input_indices),
                input_sources=input_sources,
            )
        )

    assert external_tensors, "group must have at least one external non-constant entry tensor"

    in_degree = [0] * len(group_nodes)
    out_edges = [[] for _ in group_nodes]
    in_edges = [[] for _ in group_nodes]
    for src, dst in internal_edges:
        in_degree[dst] += 1
        out_edges[src].append(dst)
        in_edges[dst].append(src)

    flow_local_indices = {spec.local_index for spec in node_specs if spec.data_input_indices}
    assert flow_local_indices, "group must contain at least one dataflow node"

    source_local_indices = [idx for idx in flow_local_indices if in_degree[idx] == 0]
    assert len(source_local_indices) == 1, (
        "group must have a single source node for tiled rewrite"
    )

    sink_local_indices = [
        idx for idx in flow_local_indices if not any(dst in flow_local_indices for dst in out_edges[idx])
    ]
    assert len(sink_local_indices) == 1, (
        "group must have exactly one sink node in internal dataflow"
    )
    sink_local_index = sink_local_indices[0]
    assert sink_local_index == len(group_nodes) - 1, (
        "group sink node must be the last node in node_range"
    )

    join_local_indices = [idx for idx in join_local_indices if idx in flow_local_indices]
    if not join_local_indices:
        join_local_index = sink_local_index
    else:
        assert len(join_local_indices) == 1, (
            "group may have at most one join node with multiple internal data inputs"
        )
        join_local_index = join_local_indices[0]
        assert join_local_index == len(group_nodes) - 1, (
            "join node must be the last node in node_range"
        )

    source_local_index = source_local_indices[0]
    reachable_from_source = set()
    stack = [source_local_index]
    while stack:
        cur = stack.pop()
        if cur in reachable_from_source:
            continue
        reachable_from_source.add(cur)
        stack.extend(dst for dst in out_edges[cur] if dst in flow_local_indices)

    can_reach_join = set()
    stack = [join_local_index]
    while stack:
        cur = stack.pop()
        if cur in can_reach_join:
            continue
        can_reach_join.add(cur)
        stack.extend(src for src in in_edges[cur] if src in flow_local_indices)

    for spec in node_specs:
        if spec.local_index not in flow_local_indices:
            continue
        assert spec.local_index in reachable_from_source, (
            f"node {spec.node.name} is not reachable from group source node"
        )
        assert spec.local_index in can_reach_join, (
            f"node {spec.node.name} does not feed the group join node"
        )

    join_node = group_nodes[join_local_index]
    group_node_ids = {id(node) for node in group_nodes}
    boundary_output_specs = []
    for spec in node_specs:
        out_tensor = spec.node.outputs[0]
        external_consumers = []
        for consumer in out_tensor.outputs:
            if id(consumer) in group_node_ids:
                continue

            external_consumers.append(consumer)

        if not external_consumers or id(spec.node) == id(join_node):
            continue

        assert spec.local_index in flow_local_indices, (
            f"node {spec.node.name} output leaves the group but is not in the tiled dataflow"
        )
        boundary_output_specs.append(
            _BoundaryOutputSpec(
                local_index=spec.local_index,
                output_tensor=out_tensor,
            )
        )

    node_spec_by_id = {id(spec.node): spec for spec in node_specs}
    return external_tensors, boundary_output_specs, node_specs, node_spec_by_id


def _analyze_group(orig_nodes, node_range):
    a, b = node_range
    assert a >= 0 and b < len(orig_nodes) and a <= b

    group_nodes = orig_nodes[a : b + 1]
    entry_tensors, boundary_output_specs, node_specs, node_spec_by_id = _collect_node_specs(group_nodes)

    return _GroupInfo(
        node_range=node_range,
        nodes=group_nodes,
        entry_tensors=entry_tensors,
        exit_tensor=group_nodes[-1].outputs[0],
        boundary_output_specs=tuple(boundary_output_specs),
        node_specs=node_specs,
        node_spec_by_id=node_spec_by_id,
    )
