import onnx
import onnx_graphsurgeon as gs

from .analysis import _analyze_group, _ensure_toposorted
from .assembly import _apply_group, _build_group

_TARGET_OPSET = 11


def _remove_constant_nodes(graph):
    constant_nodes = [node for node in graph.nodes if node.op == "Constant"]
    failed_constants = []

    for node in constant_nodes:
        node_name = node.name or "<unnamed>"

        if len(node.outputs) != 1:
            failed_constants.append(
                f"{node_name}: expected 1 output, got {len(node.outputs)}"
            )
            continue

        value_attr = node.attrs.get("value")
        if not isinstance(value_attr, gs.Constant):
            failed_constants.append(
                f"{node_name}: unsupported Constant attrs {list(node.attrs.keys())}"
            )
            continue

    if failed_constants:
        raise AssertionError(
            "Failed to remove all Constant nodes before splitting: " + "; ".join(failed_constants)
        )

    for node in constant_nodes:
        output_tensor = node.outputs[0]
        value_attr = node.attrs["value"]
        output_tensor.to_constant(values=value_attr.values)
        output_tensor.inputs.clear()
        node.outputs = []

    if constant_nodes:
        graph.nodes = [node for node in graph.nodes if node.op != "Constant"]

    return len(constant_nodes)


def rewrite_model(model, groups):
    model = onnx.version_converter.convert_version(model, _TARGET_OPSET)
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)

    total_constant_count = _remove_constant_nodes(graph)
    if total_constant_count:
        print(f"Found {total_constant_count} Constant node(s); removed them before splitting.")

    orig_nodes = list(graph.nodes)
    _ensure_toposorted(orig_nodes)

    groups_sorted = sorted(groups, key=lambda g: g.node_range[0])
    node_index_map = {id(node): idx for idx, node in enumerate(orig_nodes)}

    for group_cfg in groups_sorted:
        group_info = _analyze_group(orig_nodes, group_cfg.node_range)
        new_nodes, stitched_outputs_by_tensor_id = _build_group(group_info, group_cfg, node_index_map)
        _apply_group(graph, orig_nodes, group_info, group_cfg, new_nodes, stitched_outputs_by_tensor_id)

    out_model = gs.export_onnx(graph)
    out_model = onnx.shape_inference.infer_shapes(out_model)
    onnx.checker.check_model(out_model)

    return out_model
