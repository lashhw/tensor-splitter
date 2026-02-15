SUPPORTED_NON_CONV_OPS = {"Relu", "BatchNormalization"}
SUPPORTED_GROUP_OPS = SUPPORTED_NON_CONV_OPS | {"Conv"}


class TileBlock:
    """A scheduled work unit for one original op and one split tile."""

    def __init__(self, orig_index, tile_id, node):
        self.orig_index = orig_index
        self.tile_id = tile_id
        self.node = node

    def assign_order(self, order_map, order):
        order_map[id(self.node)] = order


def _ensure_supported_op(node):
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op {node.op} for tiled rewrite"
