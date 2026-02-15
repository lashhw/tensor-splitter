from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import onnx_graphsurgeon as gs

SUPPORTED_NON_CONV_OPS = {"Relu", "BatchNormalization"}
SUPPORTED_GROUP_OPS = SUPPORTED_NON_CONV_OPS | {"Conv"}


@dataclass
class TileBlock:
    """A scheduled work unit for one original op and one split tile."""

    orig_index: int
    tile_id: int
    node: gs.Node

    def assign_order(self, order_map: Dict[int, int], order: int) -> None:
        order_map[id(self.node)] = order


def _ensure_supported_op(node: gs.Node) -> None:
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op {node.op} for tiled rewrite"
