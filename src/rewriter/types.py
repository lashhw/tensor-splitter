from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

import onnx_graphsurgeon as gs


@dataclass
class NameScope:
    """Generates graph-unique names for rewritten tensors and nodes."""

    existing: Set[str]
    counter: int = 0

    @classmethod
    def from_existing(cls, existing: Iterable[str]) -> "NameScope":
        return cls(set(existing))

    def make(self, base: str) -> str:
        base = base.replace(":", "_")
        name = f"{base}_{self.counter}"
        self.counter += 1
        while name in self.existing:
            name = f"{base}_{self.counter}"
            self.counter += 1
        self.existing.add(name)
        return name


@dataclass
class TileBlock:
    """A scheduled work unit for one original op and one split tile."""

    orig_index: int
    tile_id: int
    nodes: List[gs.Node] = field(default_factory=list)

    def assign_priority(self, priority: Dict[int, int], order: int) -> None:
        for node in self.nodes:
            priority[id(node)] = order


UNARY_OPS = {
    "Relu",
    "Sigmoid",
    "Tanh",
    "Identity",
}
UNARY_CONST_OPS = {
    "Clip",
    "BatchNormalization",
}
BINARY_OPS = {"Add", "Mul", "Sub", "Div"}
