import json
from collections import namedtuple

GroupConfig = namedtuple(
    "_GroupConfig",
    ["node_range", "tile_count", "execution_order"]
)


def _to_int(value, field):
    assert type(value) is int, f"{field} must be an integer; got {value!r}"
    return value


def _to_tuple_pair(value, field):
    assert isinstance(value, (list, tuple)) and len(value) == 2, (
        f"{field} must be a tuple/list of length 2; got {value!r}"
    )
    return _to_int(value[0], f"{field}[0]"), _to_int(value[1], f"{field}[1]")


def _normalize_execution_order(execution_order):
    return [_to_tuple_pair(entry, "execution_order entry") for entry in execution_order]


def _validate_group(group):
    a, b = group.node_range
    assert not (a < 0 or b < 0 or b < a), (
        f"node_range must be non-negative and a <= b; got {group.node_range}"
    )
    assert group.tile_count > 0, f"tile_count must be > 0; got {group.tile_count}"

    expected = [(i, s) for i in range(a, b + 1) for s in range(group.tile_count)]
    assert len(group.execution_order) == len(expected), (
        "execution_order length mismatch: expected "
        f"{len(expected)} entries, got {len(group.execution_order)}"
    )

    expected_set = set(expected)
    execution_order_set = set(group.execution_order)
    if expected_set != execution_order_set:
        missing = sorted(expected_set - execution_order_set)
        extra = sorted(execution_order_set - expected_set)
        assert False, (
            "execution_order entries must cover each (node_index, split_id) exactly once. "
            f"Missing: {missing}, Extra: {extra}"
        )


def _validate_ranges(groups):
    ranges_sorted = sorted(group.node_range for group in groups)
    for (a0, b0), (a1, b1) in zip(ranges_sorted[:-1], ranges_sorted[1:]):
        assert a1 > b0, f"group ranges overlap or touch: {(a0, b0)} and {(a1, b1)}"


def parse_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    assert isinstance(raw, list), "config must be a list of group entries"

    groups = []
    for idx, entry in enumerate(raw):
        assert isinstance(entry, dict), f"group {idx} must be a dict-like entry"
        assert (
            "node_range" in entry
            and "tile_count" in entry
            and "execution_order" in entry
        ), f"group {idx} missing required keys: node_range, tile_count, execution_order"

        group = GroupConfig(
            node_range=_to_tuple_pair(entry["node_range"], "node_range"),
            tile_count=_to_int(entry["tile_count"], "tile_count"),
            execution_order=_normalize_execution_order(entry["execution_order"]),
        )
        _validate_group(group)
        groups.append(group)

    _validate_ranges(groups)
    return groups
