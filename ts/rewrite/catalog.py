from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import onnx_graphsurgeon as gs


@dataclass
class TileBlock:
    """A scheduled work unit for one original op and one split tile."""

    orig_index: int
    tile_id: int
    nodes: List[gs.Node] = field(default_factory=list)

    def assign_order(self, order_map: Dict[int, int], order: int) -> None:
        for node in self.nodes:
            order_map[id(node)] = order


UNARY_OPS = {"Relu"}
UNARY_CONST_OPS = {
    "BatchNormalization",
}
