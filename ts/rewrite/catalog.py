SUPPORTED_NON_CONV_OPS = {"Relu", "BatchNormalization"}
SUPPORTED_GROUP_OPS = SUPPORTED_NON_CONV_OPS | {"Conv"}


def _ensure_supported_op(node):
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op {node.op} for tiled rewrite"
