import onnx
import onnx_graphsurgeon as gs

from .analysis import _analyze_group
from .assembly import _build_group
from .ordering import _ensure_toposorted

_TARGET_OPSET = 11


def _apply_group(graph, orig_nodes, group_info, group_cfg, new_nodes, concat_out):
    node_a = orig_nodes[group_cfg.node_range[0]]
    node_b = orig_nodes[group_cfg.node_range[1]]
    start_pos = next(i for i, node in enumerate(graph.nodes) if node is node_a)
    end_pos = next(i for i, node in enumerate(graph.nodes) if node is node_b)

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

    graph.nodes = graph.nodes[:start_pos] + new_nodes + graph.nodes[end_pos + 1 :]


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
