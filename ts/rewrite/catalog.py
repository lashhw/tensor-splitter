from __future__ import annotations

import onnx_graphsurgeon as gs

SUPPORTED_NON_CONV_OPS = frozenset({"Relu", "BatchNormalization"})
SUPPORTED_GROUP_OPS = SUPPORTED_NON_CONV_OPS | {"Conv"}


def ensure_supported_op(node: gs.Node) -> None:
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op {node.op} for tiled rewrite"
