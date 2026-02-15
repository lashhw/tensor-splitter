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


def _to_tile_count(value):
    if type(value) is int:
        count = _to_int(value, "tile_count")
        assert count > 0, f"tile_count must be > 0; got {count}"
        return (count, 1), True

    assert isinstance(value, (list, tuple)) and len(value) == 2, (
        "tile_count must be an integer or a tuple/list of length 2; "
        f"got {value!r}"
    )
    tiles_h = _to_int(value[0], "tile_count[0]")
    tiles_w = _to_int(value[1], "tile_count[1]")
    assert tiles_h > 0 and tiles_w > 0, (
        f"tile_count tuple entries must be > 0; got {(tiles_h, tiles_w)}"
    )
    return (tiles_h, tiles_w), False


def _normalize_execution_entry(entry, allow_legacy_split_id):
    assert isinstance(entry, (list, tuple)) and len(entry) == 2, (
        f"execution_order entry must be a tuple/list of length 2; got {entry!r}"
    )
    node_index = _to_int(entry[0], "execution_order entry[0]")
    split = entry[1]

    if isinstance(split, (list, tuple)):
        assert len(split) == 2, (
            "execution_order entry[1] must be an integer or a tuple/list of length 2; "
            f"got {split!r}"
        )
        split_h = _to_int(split[0], "execution_order entry[1][0]")
        split_w = _to_int(split[1], "execution_order entry[1][1]")
        return node_index, (split_h, split_w)

    if allow_legacy_split_id:
        split_h = _to_int(split, "execution_order entry[1]")
        return node_index, (split_h, 0)

    assert False, (
        "execution_order entry[1] must be [split_id_h, split_id_w] when tile_count "
        f"is a 2-tuple; got {split!r}"
    )


def _normalize_execution_order(execution_order, allow_legacy_split_id):
    assert isinstance(execution_order, list), (
        f"execution_order must be a list; got {execution_order!r}"
    )
    return [
        _normalize_execution_entry(entry, allow_legacy_split_id)
        for entry in execution_order
    ]


def _validate_group(group):
    a, b = group.node_range
    assert not (a < 0 or b < 0 or b < a), (
        f"node_range must be non-negative and a <= b; got {group.node_range}"
    )
    tiles_h, tiles_w = group.tile_count
    assert tiles_h > 0 and tiles_w > 0, (
        f"tile_count entries must be > 0; got {group.tile_count}"
    )

    expected = [
        (node_index, (split_h, split_w))
        for node_index in range(a, b + 1)
        for split_h in range(tiles_h)
        for split_w in range(tiles_w)
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
            "execution_order entries must cover each "
            "(node_index, (split_id_h, split_id_w)) exactly once. "
            f"Missing: {missing}, Extra: {extra}"
        )


def _validate_ranges(groups):
    ranges_sorted = sorted(group.node_range for group in groups)
    for (a0, b0), (a1, b1) in zip(ranges_sorted[:-1], ranges_sorted[1:]):
        assert a1 > b0, f"group ranges overlap or touch: {(a0, b0)} and {(a1, b1)}"


def normalize_group_config(group):
    tile_count, tile_count_was_int = _to_tile_count(group.tile_count)
    normalized = GroupConfig(
        node_range=_to_tuple_pair(group.node_range, "node_range"),
        tile_count=tile_count,
        execution_order=_normalize_execution_order(
            list(group.execution_order),
            allow_legacy_split_id=tile_count_was_int,
        ),
    )
    _validate_group(normalized)
    return normalized


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

        group = normalize_group_config(GroupConfig(
            node_range=_to_tuple_pair(entry["node_range"], "node_range"),
            tile_count=entry["tile_count"],
            execution_order=entry["execution_order"],
        ))
        groups.append(group)

    _validate_ranges(groups)
    return groups
