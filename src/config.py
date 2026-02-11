from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

IndexPair = Tuple[int, int]
ScheduleEntry = Tuple[int, int]


@dataclass(frozen=True)
class GroupConfig:
    indices: IndexPair
    splits: int
    schedule: List[ScheduleEntry]


def _to_tuple_pair(value, field: str) -> IndexPair:
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field} must be a tuple/list of length 2; got {value!r}")
    try:
        return int(value[0]), int(value[1])
    except Exception as exc:
        raise ValueError(f"{field} must contain integers; got {value!r}") from exc


def _normalize_schedule(schedule: Iterable) -> List[ScheduleEntry]:
    return [_to_tuple_pair(entry, "schedule entry") for entry in schedule]


def _validate_group(group: GroupConfig) -> None:
    a, b = group.indices
    if a < 0 or b < 0 or b < a:
        raise ValueError(f"indices must be non-negative and a<=b; got {group.indices}")
    if group.splits <= 0:
        raise ValueError(f"splits must be > 0; got {group.splits}")

    expected = [(i, s) for i in range(a, b + 1) for s in range(group.splits)]
    if len(group.schedule) != len(expected):
        raise ValueError(
            "schedule length mismatch: expected "
            f"{len(expected)} entries, got {len(group.schedule)}"
        )

    expected_set = set(expected)
    schedule_set = set(group.schedule)
    if expected_set != schedule_set:
        missing = sorted(expected_set - schedule_set)
        extra = sorted(schedule_set - expected_set)
        raise ValueError(
            "schedule entries must cover each (node_index, split_id) exactly once. "
            f"Missing: {missing}, Extra: {extra}"
        )


def _validate_ranges(groups: Sequence[GroupConfig]) -> None:
    ranges_sorted = sorted(group.indices for group in groups)
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
        if "indices" not in entry or "splits" not in entry or "schedule" not in entry:
            raise ValueError(
                f"group {idx} missing required keys: indices, splits, schedule"
            )

        group = GroupConfig(
            indices=_to_tuple_pair(entry["indices"], "indices"),
            splits=int(entry["splits"]),
            schedule=_normalize_schedule(entry["schedule"]),
        )
        _validate_group(group)
        groups.append(group)

    _validate_ranges(groups)
    return groups
