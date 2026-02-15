from __future__ import annotations

from typing import Dict, List

import onnx
import onnx_graphsurgeon as gs

from ..config import GroupConfig
from .analysis import analyze_group
from .apply import apply_group_rewrite
from .assembly import build_group_tiles
from .naming import NameScope
from .ordering import build_execution_order_map, ensure_topologically_sorted, order_by_execution_order
from .types import GroupAnalysis

_TARGET_OPSET = 11


def rewrite_group(
    group: GroupAnalysis,
    group_cfg: GroupConfig,
    node_index_map: Dict[int, int],
    name_scope: NameScope,
) -> tuple[List[gs.Node], gs.Variable]:
    build = build_group_tiles(name_scope, group, group_cfg, node_index_map)
    order_map = build_execution_order_map(build.blocks, group_cfg.execution_order, build.concat_node)
    ordered_nodes = order_by_execution_order(build.nodes, order_map)
    return ordered_nodes, build.concat_output


def rewrite_model(model: onnx.ModelProto, groups: List[GroupConfig]) -> onnx.ModelProto:
    model = onnx.version_converter.convert_version(model, _TARGET_OPSET)
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)
    orig_nodes = list(graph.nodes)
    ensure_topologically_sorted(orig_nodes)

    groups_sorted = sorted(groups, key=lambda g: g.node_range[0])
    node_index_map = {id(node): idx for idx, node in enumerate(orig_nodes)}
    name_scope = NameScope.from_existing(set(graph.tensors().keys()))

    for group_cfg in groups_sorted:
        group = analyze_group(orig_nodes, group_cfg.node_range)
        new_nodes, concat_out = rewrite_group(group, group_cfg, node_index_map, name_scope)
        apply_group_rewrite(graph, orig_nodes, group, new_nodes, concat_out)

    out_model = gs.export_onnx(graph)
    out_model = onnx.shape_inference.infer_shapes(out_model)
    onnx.checker.check_model(out_model)

    return out_model
