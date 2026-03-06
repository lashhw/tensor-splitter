import onnx
import onnx_graphsurgeon as gs

from .analysis import _analyze_group, _ensure_toposorted
from .assembly import _apply_group, _build_group

_TARGET_OPSET = 11


def rewrite_model(model, groups):
    model = onnx.version_converter.convert_version(model, _TARGET_OPSET)
    model = onnx.shape_inference.infer_shapes(model)

    graph = gs.import_onnx(model)
    orig_nodes = list(graph.nodes)
    _ensure_toposorted(orig_nodes)

    groups_sorted = sorted(groups, key=lambda g: g.node_range[0])
    node_index_map = {id(node): idx for idx, node in enumerate(orig_nodes)}

    for group_cfg in groups_sorted:
        group_info = _analyze_group(orig_nodes, group_cfg.node_range)
        new_nodes, concat_out = _build_group(group_info, group_cfg, node_index_map)
        _apply_group(graph, orig_nodes, group_info, group_cfg, new_nodes, concat_out)

    out_model = gs.export_onnx(graph)
    out_model = onnx.shape_inference.infer_shapes(out_model)
    onnx.checker.check_model(out_model)

    return out_model
