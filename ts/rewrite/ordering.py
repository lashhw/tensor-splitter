def _ensure_toposorted(nodes):
    index = {id(node): node_idx for node_idx, node in enumerate(nodes)}
    for node_idx, node in enumerate(nodes):
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_idx = index.get(id(producer))
                if producer_idx is None:
                    continue
                assert producer_idx <= node_idx, "graph nodes are not topologically sorted"


def _build_execution_order_map(
    blocks,
    schedule,
    final_node,
):
    schedule_pos = {pair: idx for idx, pair in enumerate(schedule)}
    order_map = {}
    for block in blocks:
        key = (block.orig_index, block.tile_id)
        block.assign_order(order_map, schedule_pos[key])
    order_map[id(final_node)] = len(schedule)
    return order_map


def _order_by_execution_order(nodes, order_map):
    indexed = list(enumerate(nodes))
    scheduled = [
        (order_map[id(node)], orig_pos, node)
        for orig_pos, node in indexed
    ]
    scheduled.sort(key=lambda item: (item[0], item[1]))
    ordered = [node for _, _, node in scheduled]

    _ensure_toposorted(ordered)
    return ordered
