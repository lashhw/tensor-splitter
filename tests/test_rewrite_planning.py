from pathlib import Path
import sys

import numpy as np
import onnx_graphsurgeon as gs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.rewrite.analysis import GroupInfo
from ts.rewrite.planning import _plan_stage_ranges


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

    return GroupInfo(
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
