from pathlib import Path
import sys
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.config import GroupConfig, normalize_group_config, parse_config


def test_parse_config_accepts_integer_tile_count_and_legacy_execution_order(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "node_range": [0, 1],
                    "tile_count": 2,
                    "execution_order": [[0, 0], [0, 1], [1, 0], [1, 1]],
                }
            ]
        ),
        encoding="utf-8",
    )

    groups = parse_config(str(config_path))
    assert len(groups) == 1
    assert groups[0].tile_count == (2, 1)
    assert groups[0].execution_order == [
        (0, (0, 0)),
        (0, (1, 0)),
        (1, (0, 0)),
        (1, (1, 0)),
    ]


def test_parse_config_accepts_2d_tile_count_and_nested_execution_order(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "node_range": [0, 0],
                    "tile_count": [2, 2],
                    "execution_order": [
                        [0, [0, 0]],
                        [0, [0, 1]],
                        [0, [1, 0]],
                        [0, [1, 1]],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    groups = parse_config(str(config_path))
    assert len(groups) == 1
    assert groups[0].tile_count == (2, 2)
    assert groups[0].execution_order == [
        (0, (0, 0)),
        (0, (0, 1)),
        (0, (1, 0)),
        (0, (1, 1)),
    ]


def test_parse_config_rejects_legacy_execution_ids_for_2d_tile_count(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "node_range": [0, 0],
                    "tile_count": [2, 2],
                    "execution_order": [[0, 0], [0, 1], [0, 2], [0, 3]],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match=r"must be \[split_id_h, split_id_w\]"):
        parse_config(str(config_path))


def test_normalize_group_config_accepts_legacy_tuple_entries_for_integer_tile_count():
    group = GroupConfig(
        node_range=(0, 0),
        tile_count=2,
        execution_order=[(0, 0), (0, 1)],
    )
    normalized = normalize_group_config(group)

    assert normalized.tile_count == (2, 1)
    assert normalized.execution_order == [(0, (0, 0)), (0, (1, 0))]
