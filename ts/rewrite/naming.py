from __future__ import annotations

from dataclasses import dataclass
from typing import Set


@dataclass
class NameScope:
    existing: Set[str]
    counter: int = 0

    @classmethod
    def from_existing(cls, existing: Set[str]) -> "NameScope":
        return cls(set(existing))

    def make(self, base: str) -> str:
        name = f"{base}_{self.counter}"
        self.counter += 1
        while name in self.existing:
            name = f"{base}_{self.counter}"
            self.counter += 1
        self.existing.add(name)
        return name
