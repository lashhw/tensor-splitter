from __future__ import annotations

import onnx
import onnx_graphsurgeon as gs

from .group_chain import _analyze_group
from .naming import NameScope
from .op_rewriters import _apply_schedule_priority, _build_group_output, _build_group_tiles
from .scheduling import _ensure_toposorted, _replace_tensor_consumers, _toposort_with_priority


def _rewrite_group(
    group_info,
    group_cfg,
    node_index_map,
    name_scope: NameScope,
):
    tiles, nodes, blocks = _build_group_tiles(
        name_scope,
        group_info,
        group_cfg,
        node_index_map,
    )

    concat_out = _build_group_output(
        name_scope,
        tiles,
        shape_hint=group_info.exit_tensor.shape,
        nodes=nodes,
    )

    priority = _apply_schedule_priority(blocks, group_cfg.execution_order)
    ordered_nodes = _toposort_with_priority(nodes, priority)
    return ordered_nodes, concat_out


def rewrite_model(model, groups):
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)
    orig_nodes = list(graph.nodes)
    _ensure_toposorted(orig_nodes)

    groups_sorted = sorted(groups, key=lambda g: g.node_range[0])
    node_index_map = {id(node): idx for idx, node in enumerate(orig_nodes)}
    name_scope = NameScope.from_existing(graph.tensors().keys())

    for group_cfg in groups_sorted:
        group_info = _analyze_group(orig_nodes, group_cfg.node_range)
        new_nodes, concat_out = _rewrite_group(
            group_info,
            group_cfg,
            node_index_map,
            name_scope,
        )

        _replace_tensor_consumers(graph, group_info.exit_tensor, concat_out)

        for node in group_info.nodes:
            node.inputs = []
            node.outputs = []

        node_a = orig_nodes[group_cfg.node_range[0]]
        node_b = orig_nodes[group_cfg.node_range[1]]
        start_pos = graph.nodes.index(node_a)
        end_pos = graph.nodes.index(node_b)
        if end_pos < start_pos:
            raise RuntimeError("group nodes are not contiguous in current graph")

        graph.nodes = graph.nodes[:start_pos] + new_nodes + graph.nodes[end_pos + 1 :]

    graph.cleanup(remove_unused_graph_inputs=False)

    out_model = gs.export_onnx(graph)
    out_model = onnx.shape_inference.infer_shapes(out_model)
    onnx.checker.check_model(out_model)
    return out_model
