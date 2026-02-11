from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set


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
