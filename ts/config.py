from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

IndexPair = Tuple[int, int]
ScheduleEntry = Tuple[int, int]


@dataclass(frozen=True)
class GroupConfig:
    node_range: IndexPair
    tile_count: int
    execution_order: List[ScheduleEntry]


def _to_tuple_pair(value: object, field: str) -> IndexPair:
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field} must be a tuple/list of length 2; got {value!r}")
    try:
        return int(value[0]), int(value[1])
    except Exception as exc:
        raise ValueError(f"{field} must contain integers; got {value!r}") from exc


def _normalize_execution_order(execution_order: Iterable[object]) -> List[ScheduleEntry]:
    return [_to_tuple_pair(entry, "execution_order entry") for entry in execution_order]


def _validate_group(group: GroupConfig) -> None:
    a, b = group.node_range
    if a < 0 or b < 0 or b < a:
        raise ValueError(f"node_range must be non-negative and a <= b; got {group.node_range}")
    if group.tile_count <= 0:
        raise ValueError(f"tile_count must be > 0; got {group.tile_count}")

    expected = [(i, s) for i in range(a, b + 1) for s in range(group.tile_count)]
    if len(group.execution_order) != len(expected):
        raise ValueError(
            "execution_order length mismatch: expected "
            f"{len(expected)} entries, got {len(group.execution_order)}"
        )

    expected_set = set(expected)
    execution_order_set = set(group.execution_order)
    if expected_set != execution_order_set:
        missing = sorted(expected_set - execution_order_set)
        extra = sorted(execution_order_set - expected_set)
        raise ValueError(
            "execution_order entries must cover each (node_index, split_id) exactly once. "
            f"Missing: {missing}, Extra: {extra}"
        )


def _validate_ranges(groups: Sequence[GroupConfig]) -> None:
    ranges_sorted = sorted(group.node_range for group in groups)
    for (a0, b0), (a1, b1) in zip(ranges_sorted, ranges_sorted[1:]):
        if a1 <= b0:
            raise ValueError(f"group ranges overlap or touch: {(a0, b0)} and {(a1, b1)}")


def parse_config(path: str | Path) -> List[GroupConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("config must be a list of group entries")

    groups = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"group {idx} must be a dict-like entry")
        if (
            "node_range" not in entry
            or "tile_count" not in entry
            or "execution_order" not in entry
        ):
            raise ValueError(
                f"group {idx} missing required keys: node_range, tile_count, execution_order"
            )

        group = GroupConfig(
            node_range=_to_tuple_pair(entry["node_range"], "node_range"),
            tile_count=int(entry["tile_count"]),
            execution_order=_normalize_execution_order(entry["execution_order"]),
        )
        _validate_group(group)
        groups.append(group)

    _validate_ranges(groups)
    return groups
