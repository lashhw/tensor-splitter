from collections import namedtuple

import onnx_graphsurgeon as gs

SUPPORTED_GROUP_OPS = {"Conv", "Relu", "Add", "Concat", "AveragePool", "Reshape"}

_InputSource = namedtuple(
    "_InputSource",
    ["kind", "producer_local_index"],
)

_NodeSpec = namedtuple(
    "_NodeSpec",
    ["node", "local_index", "input_sources"],
)

_BoundaryOutputSpec = namedtuple(
    "_BoundaryOutputSpec",
    ["local_index", "output_tensor"],
)

_GroupInfo = namedtuple(
    "_GroupInfo",
    [
        "node_range",
        "tile_count",
        "execution_order",
        "nodes",
        "entry_tensor",
        "exit_tensor",
        "boundary_output_specs",
        "node_specs",
    ],
)


def _ensure_toposorted(nodes):
    index = {id(node): node_index for node_index, node in enumerate(nodes)}
    for node_index, node in enumerate(nodes):
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_index = index.get(id(producer))
                if producer_index is None:
                    continue
                assert producer_index < node_index, "graph nodes are not topologically sorted"


def _ensure_supported_op(node):
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op in split group: {node.op}"
    if node.op == "Concat":
        assert node.attrs["axis"] == 1, "Concat in split group must use axis=1"


def _build_adjacency(node_count, internal_edges):
    out_edges = [[] for _ in range(node_count)]
    in_edges = [[] for _ in range(node_count)]
    for src, dst in internal_edges:
        out_edges[src].append(dst)
        in_edges[dst].append(src)
    return in_edges, out_edges


def _reachable_indices(seed_index, edges):
    visited = set()
    stack = [seed_index]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(edges[current])
    return visited


def _validate_group_topology(node_specs, in_edges, out_edges):
    source_local_indices = [index for index, incoming in enumerate(in_edges) if not incoming]
    assert len(source_local_indices) == 1, "group must have exactly one source node"
    source_local_index = source_local_indices[0]
    assert source_local_index == 0, "group source node must be the first node in node_range"

    sink_local_indices = [index for index, outgoing in enumerate(out_edges) if not outgoing]
    assert len(sink_local_indices) == 1, "group must have exactly one sink node"
    sink_local_index = sink_local_indices[0]
    assert sink_local_index == len(node_specs) - 1, "group sink node must be the last node in node_range"

    join_local_indices = [index for index, incoming in enumerate(in_edges) if len(incoming) >= 2]
    if join_local_indices:
        assert len(join_local_indices) == 1, "group may have at most one join node"
        assert join_local_indices[0] == sink_local_index, "join node must be the sink node"

    reachable_from_source = _reachable_indices(source_local_index, out_edges)
    reachable_to_sink = _reachable_indices(sink_local_index, in_edges)
    for spec in node_specs:
        assert spec.local_index in reachable_from_source, (
            f"node {spec.node.name} is not reachable from group source node"
        )
        assert spec.local_index in reachable_to_sink, (
            f"node {spec.node.name} does not feed the group sink node"
        )

    return sink_local_index


def _collect_boundary_outputs(group_nodes, node_specs, sink_local_index):
    boundary_output_specs = []
    group_node_ids = {id(node) for node in group_nodes}

    for spec in node_specs:
        if spec.local_index == sink_local_index:
            continue

        output_tensor = spec.node.outputs[0]
        has_external_consumer = any(id(consumer) not in group_node_ids for consumer in output_tensor.outputs)
        if not has_external_consumer:
            continue

        boundary_output_specs.append(
            _BoundaryOutputSpec(
                local_index=spec.local_index,
                output_tensor=output_tensor,
            )
        )

    return boundary_output_specs


def _collect_node_specs(group_nodes):
    local_index_by_id = {id(node): local_index for local_index, node in enumerate(group_nodes)}

    entry_tensor = None
    internal_edges = []
    node_specs = []

    for local_index, node in enumerate(group_nodes):
        _ensure_supported_op(node)
        assert len(node.outputs) == 1, f"node {node.name} must have exactly one output"

        input_sources = {}

        for input_index, tensor in enumerate(node.inputs):
            if isinstance(tensor, gs.Constant):
                continue

            producers = list(tensor.inputs)
            assert len(producers) <= 1, f"tensor {tensor.name} must have at most one producer"

            producer_local_index = None
            if producers:
                producer_local_index = local_index_by_id.get(id(producers[0]))

            if producer_local_index is None:
                assert entry_tensor is None
                entry_tensor = tensor
                input_sources[input_index] = _InputSource(
                    kind="entry",
                    producer_local_index=None,
                )
            else:
                input_sources[input_index] = _InputSource(
                    kind="node",
                    producer_local_index=producer_local_index,
                )
                internal_edges.append((producer_local_index, local_index))

        assert input_sources, f"node {node.name} must have at least one data input"

        node_specs.append(
            _NodeSpec(
                node=node,
                local_index=local_index,
                input_sources=input_sources,
            )
        )

    assert entry_tensor is not None, "group must have exactly one external entry tensor"

    in_edges, out_edges = _build_adjacency(len(group_nodes), internal_edges)
    sink_local_index = _validate_group_topology(node_specs, in_edges, out_edges)
    boundary_output_specs = _collect_boundary_outputs(group_nodes, node_specs, sink_local_index)
    return entry_tensor, boundary_output_specs, node_specs


def _analyze_group(orig_nodes, group_cfg):
    _ensure_toposorted(orig_nodes)

    start, end = group_cfg.node_range
    assert 0 <= start <= end < len(orig_nodes), f"invalid node_range {group_cfg.node_range}"

    group_nodes = orig_nodes[start : end + 1]
    entry_tensor, boundary_output_specs, node_specs = _collect_node_specs(group_nodes)
    exit_tensor = group_nodes[-1].outputs[0]

    return _GroupInfo(
        node_range=group_cfg.node_range,
        tile_count=group_cfg.tile_count,
        execution_order=group_cfg.execution_order,
        nodes=group_nodes,
        entry_tensor=entry_tensor,
        exit_tensor=exit_tensor,
        boundary_output_specs=boundary_output_specs,
        node_specs=node_specs,
    )
