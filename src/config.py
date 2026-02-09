from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple


@dataclass(frozen=True)
class GroupConfig:
    indices: Tuple[int, int]
    splits: int
    schedule: List[Tuple[int, int]]


class ConfigError(ValueError):
    pass


def _parse_python_literal(text: str, path: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:
        # Try to handle files that wrap the literal in an assignment.
        try:
            tree = ast.parse(text, filename=path, mode="exec")
        except Exception as exc:
            raise ConfigError(
                f"Failed to parse config {path}. Provide JSON or a Python literal."
            ) from exc

        for node in tree.body:
            if isinstance(node, ast.Assign):
                try:
                    return ast.literal_eval(node.value)
                except Exception:
                    continue
            if isinstance(node, ast.Expr):
                try:
                    return ast.literal_eval(node.value)
                except Exception:
                    continue

        raise ConfigError(
            f"Failed to parse config {path}. Provide JSON or a Python literal."
        )


def _load_raw_config(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Prefer JSON if it parses cleanly.
    try:
        return json.loads(text)
    except Exception:
        return _parse_python_literal(text, path)


def _to_tuple_pair(value: Any, field: str) -> Tuple[int, int]:
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ConfigError(f"{field} must be a tuple/list of length 2; got {value!r}")
    try:
        return int(value[0]), int(value[1])
    except Exception as exc:
        raise ConfigError(f"{field} must contain integers; got {value!r}") from exc


def _normalize_schedule(schedule: Iterable[Any]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for entry in schedule:
        out.append(_to_tuple_pair(entry, "schedule entry"))
    return out


def _validate_group(group: GroupConfig) -> None:
    a, b = group.indices
    if a < 0 or b < 0 or b < a:
        raise ConfigError(f"indices must be non-negative and a<=b; got {group.indices}")
    if group.splits <= 0:
        raise ConfigError(f"splits must be > 0; got {group.splits}")

    expected = [(i, s) for i in range(a, b + 1) for s in range(group.splits)]
    if len(group.schedule) != len(expected):
        raise ConfigError(
            "schedule length mismatch: expected "
            f"{len(expected)} entries, got {len(group.schedule)}"
        )
    expected_set = set(expected)
    schedule_set = set(group.schedule)
    if expected_set != schedule_set:
        missing = sorted(expected_set - schedule_set)
        extra = sorted(schedule_set - expected_set)
        raise ConfigError(
            "schedule entries must cover each (node_index, split_id) exactly once. "
            f"Missing: {missing}, Extra: {extra}"
        )


def parse_config(path: str) -> List[GroupConfig]:
    raw = _load_raw_config(path)
    if not isinstance(raw, list):
        raise ConfigError("config must be a list of group entries")

    groups: List[GroupConfig] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"group {idx} must be a dict-like entry")
        if "indices" not in entry or "splits" not in entry or "schedule" not in entry:
            raise ConfigError(
                f"group {idx} missing required keys: indices, splits, schedule"
            )
        indices = _to_tuple_pair(entry["indices"], "indices")
        splits = int(entry["splits"])
        schedule = _normalize_schedule(entry["schedule"])
        group = GroupConfig(indices=indices, splits=splits, schedule=schedule)
        _validate_group(group)
        groups.append(group)

    # Ensure groups do not overlap.
    ranges = []
    for group in groups:
        ranges.append(group.indices)
    ranges_sorted = sorted(ranges, key=lambda x: (x[0], x[1]))
    for (a0, b0), (a1, b1) in zip(ranges_sorted, ranges_sorted[1:]):
        if a1 <= b0:
            raise ConfigError(
                f"group ranges overlap or touch: {(a0, b0)} and {(a1, b1)}"
            )

    return groups


def config_to_jsonable(groups: List[GroupConfig]) -> List[dict]:
    out = []
    for group in groups:
        out.append(
            {
                "indices": [group.indices[0], group.indices[1]],
                "splits": group.splits,
                "schedule": [[i, s] for i, s in group.schedule],
            }
        )
    return out
