from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import onnx
import onnx_graphsurgeon as gs

from src.config import GroupConfig
from src.group_analysis import GroupInfo, analyze_group
from src.rewrite.ops import (
    BINARY_OPS,
    HEIGHT_AXIS,
    UNARY_CONST_OPS,
    UNARY_OPS,
    NameScope,
    TileBlock,
    _build_binary_tiles,
    _build_conv_tiles,
    _build_entry_tiles,
    _build_unary_const_tiles,
    _build_unary_tiles,
    _ensure_supported_op,
    _make_concat,
)


# ---- Graph utilities ----

def _toposort_with_priority(nodes: List[gs.Node], priority: Dict[gs.Node, int]) -> List[gs.Node]:
    node_set = set(nodes)
    adj = {node: [] for node in nodes}
    indeg = {node: 0 for node in nodes}

    for node in nodes:
        for inp in node.inputs:
            for prod in inp.inputs:
                if prod in node_set:
                    adj[prod].append(node)
                    indeg[node] += 1

    import heapq

    order = {node: idx for idx, node in enumerate(nodes)}
    heap = []
    for node, deg in indeg.items():
        if deg == 0:
            heapq.heappush(heap, (priority.get(node, 10**9), order[node], node))

    result = []
    while heap:
        _, _, node = heapq.heappop(heap)
        result.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, (priority.get(nxt, 10**9), order[nxt], nxt))

    if len(result) != len(nodes):
        raise RuntimeError("cycle detected while ordering rewritten nodes")
    return result


def _ensure_toposorted(nodes: Sequence[gs.Node]) -> None:
    index = {node: idx for idx, node in enumerate(nodes)}
    for node in nodes:
        node_idx = index[node]
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_idx = index.get(producer)
                if producer_idx is None:
                    continue
                if producer_idx > node_idx:
                    raise ValueError("graph nodes are not topologically sorted")


def _replace_tensor_consumers(graph: gs.Graph, old: gs.Tensor, new: gs.Tensor) -> None:
    for consumer in list(old.outputs):
        for idx, inp in enumerate(consumer.inputs):
            if inp is old:
                consumer.inputs[idx] = new
    for idx, out in enumerate(graph.outputs):
        if out is old:
            graph.outputs[idx] = new


def _apply_schedule_priority(
    blocks: Sequence[TileBlock],
    schedule: Sequence[Tuple[int, int]],
) -> Dict[gs.Node, int]:
    schedule_pos = {pair: idx for idx, pair in enumerate(schedule)}
    priority: Dict[gs.Node, int] = {}
    for block in blocks:
        order = schedule_pos.get((block.orig_index, block.tile_id))
        if order is None:
            continue
        block.assign_priority(priority, order)
    return priority


# ---- Rewrite entry points ----

def rewrite_group(
    graph: gs.Graph,
    group_info: GroupInfo,
    group_cfg: GroupConfig,
    node_index_map: Dict[gs.Node, int],
    name_scope: NameScope,
) -> Tuple[List[gs.Node], gs.Tensor]:
    """Rewrite a single contiguous chain group into tiled subgraphs."""
    nodes: List[gs.Node] = []
    blocks: List[TileBlock] = []

    for node in group_info.nodes:
        _ensure_supported_op(node)

    tiles, ranges = _build_entry_tiles(name_scope, group_info.entry_tensor, group_cfg.splits, nodes)

    for node in group_info.nodes:
        orig_index = node_index_map[node]
        main_idx = group_info.main_input_index[node]

        if node.op == "Conv":
            tiles, ranges, conv_blocks = _build_conv_tiles(
                name_scope,
                node,
                orig_index,
                tiles,
                ranges,
                group_cfg.splits,
                nodes,
            )
            blocks.extend(conv_blocks)
        elif node.op in UNARY_OPS:
            tiles, op_blocks = _build_unary_tiles(name_scope, node, orig_index, tiles, nodes)
            blocks.extend(op_blocks)
        elif node.op in UNARY_CONST_OPS:
            tiles, op_blocks = _build_unary_const_tiles(
                name_scope, node, orig_index, tiles, nodes, main_idx
            )
            blocks.extend(op_blocks)
        elif node.op in BINARY_OPS:
            tiles, op_blocks = _build_binary_tiles(
                name_scope, node, orig_index, tiles, nodes, main_idx
            )
            blocks.extend(op_blocks)
        else:
            raise RuntimeError(f"unsupported op {node.op}")

    concat_out = _make_concat(
        name_scope,
        tiles,
        axis=HEIGHT_AXIS,
        nodes=nodes,
        shape_hint=group_info.exit_tensor.shape,
    )

    priority = _apply_schedule_priority(blocks, group_cfg.schedule)
    ordered_nodes = _toposort_with_priority(nodes, priority)

    return ordered_nodes, concat_out


def rewrite_model(
    model: onnx.ModelProto,
    groups: Sequence[GroupConfig],
) -> onnx.ModelProto:
    """Rewrite the model by tiling each configured group."""
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)
    orig_nodes = list(graph.nodes)
    _ensure_toposorted(orig_nodes)
    node_index_map = {node: idx for idx, node in enumerate(orig_nodes)}
    name_scope = NameScope(graph.tensors().keys())

    groups_sorted = sorted(groups, key=lambda g: g.indices[0])
    for group_cfg in groups_sorted:
        group_info = analyze_group(orig_nodes, group_cfg.indices)
        new_nodes, concat_out = rewrite_group(
            graph, group_info, group_cfg, node_index_map, name_scope
        )

        _replace_tensor_consumers(graph, group_info.exit_tensor, concat_out)

        for node in group_info.nodes:
            node.inputs = []
            node.outputs = []

        # Replace nodes in graph list.
        node_a = orig_nodes[group_cfg.indices[0]]
        node_b = orig_nodes[group_cfg.indices[1]]
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
