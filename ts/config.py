import json
from collections import namedtuple

_GroupConfig = namedtuple(
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


def _normalize_tile_count(tile_count):
    assert isinstance(tile_count, (list, tuple)) and len(tile_count) == 2, (
        "tile_count must be a tuple/list [split_count_h, split_count_w]; "
        f"got {tile_count!r}"
    )
    split_count_h = _to_int(tile_count[0], "tile_count[0]")
    split_count_w = _to_int(tile_count[1], "tile_count[1]")
    assert split_count_h > 0 and split_count_w > 0, (
        "tile_count tuple values must be > 0; "
        f"got ({split_count_h}, {split_count_w})"
    )
    return split_count_h, split_count_w


def _normalize_execution_order(execution_order):
    normalized = []
    for entry in execution_order:
        assert isinstance(entry, (list, tuple)) and len(entry) == 2, (
            "execution_order entry must be [node_index, [split_id_h, split_id_w]]; "
            f"got {entry!r}"
        )
        node_index = _to_int(entry[0], "execution_order entry[0]")

        split_id = entry[1]
        assert isinstance(split_id, (list, tuple)) and len(split_id) == 2, (
            "execution_order split id must be [split_id_h, split_id_w]; "
            f"got {split_id!r}"
        )
        split_id_h = _to_int(split_id[0], "execution_order entry[1][0]")
        split_id_w = _to_int(split_id[1], "execution_order entry[1][1]")
        normalized.append((node_index, (split_id_h, split_id_w)))
    return normalized


def _validate_group(group):
    a, b = group.node_range
    assert not (a < 0 or b < 0 or b < a), (
        f"node_range must be non-negative and a <= b; got {group.node_range}"
    )
    split_count_h, split_count_w = group.tile_count
    assert split_count_h > 0 and split_count_w > 0, (
        "tile_count tuple values must be > 0; "
        f"got ({split_count_h}, {split_count_w})"
    )
    expected = [
        (i, (s_h, s_w))
        for i in range(a, b + 1)
        for s_h in range(split_count_h)
        for s_w in range(split_count_w)
    ]
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

    raw_groups = []
    for idx, entry in enumerate(raw):
        assert isinstance(entry, dict), f"group {idx} must be a dict-like entry"
        assert (
            "node_range" in entry
            and "tile_count" in entry
            and "execution_order" in entry
        ), f"group {idx} missing required keys: node_range, tile_count, execution_order"

        raw_groups.append(
            _GroupConfig(
                node_range=entry["node_range"],
                tile_count=entry["tile_count"],
                execution_order=entry["execution_order"],
            )
        )

    return _normalize_groups(raw_groups)


def _normalize_groups(groups):
    normalized_groups = []
    for idx, group in enumerate(groups):
        assert hasattr(group, "node_range"), f"group {idx} missing node_range"
        assert hasattr(group, "tile_count"), f"group {idx} missing tile_count"
        assert hasattr(group, "execution_order"), f"group {idx} missing execution_order"

        tile_count = _normalize_tile_count(group.tile_count)
        normalized_group = _GroupConfig(
            node_range=_to_tuple_pair(group.node_range, "node_range"),
            tile_count=tile_count,
            execution_order=_normalize_execution_order(group.execution_order),
        )
        _validate_group(normalized_group)
        normalized_groups.append(normalized_group)

    _validate_ranges(normalized_groups)
    return normalized_groups
