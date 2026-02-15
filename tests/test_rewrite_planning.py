from pathlib import Path
import sys

import numpy as np
import onnx_graphsurgeon as gs
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.rewrite.analysis import _GroupInfo
from ts.rewrite.planning import _build_ordered_node_writer, _plan_stage_ranges


def _make_group_info():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])

    w0 = gs.Constant("w0", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    y0 = gs.Variable("y0", dtype=np.float32, shape=[1, 4, 8, 8])
    conv0 = gs.Node(
        op="Conv",
        inputs=[inp, w0],
        outputs=[y0],
        attrs={
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
            "dilations": [1, 1],
        },
    )

    y1 = gs.Variable("y1", dtype=np.float32, shape=[1, 4, 8, 8])
    relu = gs.Node(op="Relu", inputs=[y0], outputs=[y1])

    w2 = gs.Constant("w2", values=np.random.randn(5, 4, 3, 3).astype(np.float32))
    y2 = gs.Variable("y2", dtype=np.float32, shape=[1, 5, 4, 8])
    conv2 = gs.Node(
        op="Conv",
        inputs=[y1, w2],
        outputs=[y2],
        attrs={
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [2, 1],
            "dilations": [1, 1],
        },
    )

    return _GroupInfo(
        node_range=(0, 2),
        nodes=[conv0, relu, conv2],
        entry_tensor=inp,
        exit_tensor=y2,
        main_input_indices=[0, 0, 0],
    )


def test_plan_stage_ranges_backpropagates_and_tracks_conv_metadata():
    group_info = _make_group_info()

    stage_plan = _plan_stage_ranges(group_info, tile_count=2)

    assert stage_plan.stage_ranges[-1] == [(0, 2), (2, 4)]
    assert stage_plan.stage_ranges[2] == [(0, 4), (3, 8)]
    assert stage_plan.stage_ranges[1] == [(0, 4), (3, 8)]
    assert stage_plan.stage_ranges[0] == [(0, 5), (2, 8)]

    assert stage_plan.conv_slices_by_stage[0] is not None
    assert stage_plan.conv_slices_by_stage[1] is None
    assert stage_plan.conv_slices_by_stage[2] is not None


def test_writer_allocates_two_positions_for_first_stage_slots():
    schedule = [(10, 0), (10, 1), (11, 0), (11, 1)]
    place_node, finalize_nodes = _build_ordered_node_writer(schedule, first_orig_index=10)

    place_node(10, 0, "entry_0")
    place_node(10, 0, "op0_0")
    place_node(10, 1, "entry_1")
    place_node(10, 1, "op0_1")
    place_node(11, 0, "op1_0")
    place_node(11, 1, "op1_1")

    ordered_nodes = finalize_nodes("concat")
    assert ordered_nodes == ["entry_0", "op0_0", "entry_1", "op0_1", "op1_0", "op1_1", "concat"]


def test_writer_rejects_unknown_schedule_key():
    schedule = [(0, 0)]
    place_node, _ = _build_ordered_node_writer(schedule, first_orig_index=0)

    with pytest.raises(AssertionError, match=r"missing execution_order entry for \(1, 0\)"):
        place_node(1, 0, "node")


def test_writer_rejects_slot_overfill():
    schedule = [(0, 0)]
    place_node, _ = _build_ordered_node_writer(schedule, first_orig_index=0)

    place_node(0, 0, "entry")
    place_node(0, 0, "op")
    with pytest.raises(AssertionError, match=r"execution_order slot \(0, 0\) has too many rewritten nodes"):
        place_node(0, 0, "extra")


def test_writer_rejects_finalize_with_missing_nodes():
    schedule = [(0, 0), (1, 0)]
    place_node, finalize_nodes = _build_ordered_node_writer(schedule, first_orig_index=0)

    place_node(0, 0, "entry")
    place_node(0, 0, "op")
    with pytest.raises(AssertionError, match=r"execution_order slot \(1, 0\) has missing rewritten nodes"):
        finalize_nodes("concat")


def test_writer_appends_concat_last_and_has_no_gaps():
    schedule = [(0, 0), (1, 0)]
    place_node, finalize_nodes = _build_ordered_node_writer(schedule, first_orig_index=0)

    place_node(0, 0, "entry")
    place_node(0, 0, "op")
    place_node(1, 0, "next")
    ordered_nodes = finalize_nodes("concat")

    assert ordered_nodes[-1] == "concat"
    assert all(node is not None for node in ordered_nodes)
