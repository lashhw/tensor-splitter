from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.config import parse_config


def _write_config(tmp_path, value):
    path = tmp_path / "config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f)
    return path


def test_parse_config_rejects_integer_tile_count(tmp_path):
    path = _write_config(
        tmp_path,
        [
            {
                "node_range": [0, 1],
                "tile_count": 2,
                "execution_order": [[0, 0], [0, 1], [1, 0], [1, 1]],
            }
        ],
    )

    with pytest.raises(AssertionError, match=r"tile_count must be a tuple/list"):
        parse_config(path)


def test_parse_config_accepts_tuple_tile_count_and_nested_execution_order(tmp_path):
    path = _write_config(
        tmp_path,
        [
            {
                "node_range": [0, 1],
                "tile_count": [2, 2],
                "execution_order": [
                    [0, [0, 0]],
                    [0, [0, 1]],
                    [0, [1, 0]],
                    [0, [1, 1]],
                    [1, [0, 0]],
                    [1, [0, 1]],
                    [1, [1, 0]],
                    [1, [1, 1]],
                ],
            }
        ],
    )

    groups = parse_config(path)
    assert len(groups) == 1
    assert groups[0].tile_count == (2, 2)
    assert groups[0].execution_order == [
        (0, (0, 0)),
        (0, (0, 1)),
        (0, (1, 0)),
        (0, (1, 1)),
        (1, (0, 0)),
        (1, (0, 1)),
        (1, (1, 0)),
        (1, (1, 1)),
    ]


def test_parse_config_rejects_flat_execution_order_for_tuple_tile_count(tmp_path):
    path = _write_config(
        tmp_path,
        [
            {
                "node_range": [0, 0],
                "tile_count": [2, 2],
                "execution_order": [[0, 0], [0, 1], [0, 2], [0, 3]],
            }
        ],
    )

    with pytest.raises(AssertionError, match=r"execution_order split id must be"):
        parse_config(path)
