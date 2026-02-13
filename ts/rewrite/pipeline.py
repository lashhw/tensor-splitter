from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import onnx
import onnx_graphsurgeon as gs

from ..config import GroupConfig
from .analysis import GroupInfo, _analyze_group
from .assembly import _build_group_output, _build_group_tiles
from .naming import NameScope
from .ordering import _build_execution_order_map, _ensure_toposorted, _order_by_execution_order


def _rewrite_group(
    group_info: GroupInfo,
    group_cfg: GroupConfig,
    node_index_map: Dict[int, int],
    name_scope: NameScope,
) -> Tuple[List[gs.Node], gs.Variable]:
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

    order_map = _build_execution_order_map(blocks, group_cfg.execution_order)
    ordered_nodes = _order_by_execution_order(nodes, order_map)
    return ordered_nodes, concat_out


def _apply_group(
    graph: gs.Graph,
    orig_nodes: List[gs.Node],
    group_info: GroupInfo,
    group_cfg: GroupConfig,
    new_nodes: List[gs.Node],
    concat_out: gs.Variable,
) -> None:
    for consumer in list(group_info.exit_tensor.outputs):
        for idx, inp in enumerate(consumer.inputs):
            if inp is group_info.exit_tensor:
                consumer.inputs[idx] = concat_out

    for idx, out in enumerate(graph.outputs):
        if out is group_info.exit_tensor:
            graph.outputs[idx] = concat_out

    for node in group_info.nodes:
        node.inputs = []
        node.outputs = []

    node_a = orig_nodes[group_cfg.node_range[0]]
    node_b = orig_nodes[group_cfg.node_range[1]]
    start_pos = graph.nodes.index(node_a)
    end_pos = graph.nodes.index(node_b)

    graph.nodes = graph.nodes[:start_pos] + new_nodes + graph.nodes[end_pos + 1 :]


def rewrite_model(model: onnx.ModelProto, groups: Sequence[GroupConfig]) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)
    orig_nodes = list(graph.nodes)
    _ensure_toposorted(orig_nodes)

    groups_sorted = sorted(groups, key=lambda g: g.node_range[0])
    node_index_map = {id(node): idx for idx, node in enumerate(orig_nodes)}
    name_scope = NameScope.from_existing(graph.tensors().keys())

    for group_cfg in groups_sorted:
        group_info = _analyze_group(orig_nodes, group_cfg.node_range)
        new_nodes, concat_out = _rewrite_group(group_info, group_cfg, node_index_map, name_scope)
        _apply_group(graph, orig_nodes, group_info, group_cfg, new_nodes, concat_out)

    out_model = gs.export_onnx(graph)
    out_model = onnx.shape_inference.infer_shapes(out_model)
    onnx.checker.check_model(out_model)

    return out_model
