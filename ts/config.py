import json
from collections import namedtuple

_GroupConfig = namedtuple(
    "_GroupConfig",
    ["node_range", "tile_count", "execution_order"]
)


def _to_int(value):
    assert type(value) is int
    return value


def _to_tuple_pair(value):
    assert isinstance(value, (list, tuple)) and len(value) == 2
    return _to_int(value[0]), _to_int(value[1])


def _normalize_execution_order(execution_order):
    normalized = []

    for entry in execution_order:
        assert isinstance(entry, (list, tuple)) and len(entry) == 2
        node_index = _to_int(entry[0])
        split_id = entry[1]

        assert isinstance(split_id, (list, tuple)) and len(split_id) == 2
        split_id_h = _to_int(split_id[0])
        split_id_w = _to_int(split_id[1])

        normalized.append((node_index, (split_id_h, split_id_w)))

    return normalized


def _validate_group(group):
    a, b = group.node_range
    assert a >= 0 and b >= a

    split_count_h, split_count_w = group.tile_count
    assert split_count_h > 0 and split_count_w > 0

    expected = [
        (i, (s_h, s_w))
        for i in range(a, b + 1)
        for s_h in range(split_count_h)
        for s_w in range(split_count_w)
    ]
    assert len(group.execution_order) == len(expected)

    expected_set = set(expected)
    execution_order_set = set(group.execution_order)
    assert expected_set == execution_order_set


def _validate_ranges(groups):
    ranges_sorted = sorted(group.node_range for group in groups)
    for (a0, b0), (a1, b1) in zip(ranges_sorted[:-1], ranges_sorted[1:]):
        assert a1 > b0, f"group ranges overlap or touch: {(a0, b0)} and {(a1, b1)}"


def parse_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    assert isinstance(raw, list)

    normalized_groups = []
    for entry in raw:
        assert isinstance(entry, dict)
        assert "node_range" in entry and "tile_count" in entry and "execution_order" in entry
        normalized_group = _GroupConfig(
            node_range=_to_tuple_pair(entry["node_range"]),
            tile_count=_to_tuple_pair(entry["tile_count"]),
            execution_order=_normalize_execution_order(entry["execution_order"]),
        )
        _validate_group(normalized_group)
        normalized_groups.append(normalized_group)

    _validate_ranges(normalized_groups)
    return normalized_groups
