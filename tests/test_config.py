import json

import pytest

from src.config import GroupConfig, parse_config


def test_group_config_equality_semantics():
    a = GroupConfig(indices=(1, 2), splits=2, schedule=[(1, 0), (1, 1), (2, 0), (2, 1)])
    b = GroupConfig(indices=(1, 2), splits=2, schedule=[(1, 0), (1, 1), (2, 0), (2, 1)])
    assert a == b


def test_parse_config_rejects_touching_ranges(tmp_path):
    payload = [
        {
            "indices": [0, 1],
            "splits": 2,
            "schedule": [[0, 0], [0, 1], [1, 0], [1, 1]],
        },
        {
            "indices": [1, 1],
            "splits": 2,
            "schedule": [[1, 0], [1, 1]],
        },
    ]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap or touch"):
        parse_config(config_path)
