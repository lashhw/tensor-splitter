from pathlib import Path
import sys

import numpy as np
import onnx_graphsurgeon as gs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.rewrite.conv import ConvSpec, _conv_input_slice_for_output, _parse_conv_spec


def _make_conv_node(attrs):
    x = gs.Variable("x", dtype=np.float32, shape=[1, 3, 8, 8])
    w = gs.Constant("w", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    y = gs.Variable("y", dtype=np.float32, shape=[1, 4, 8, 8])
    return gs.Node(op="Conv", inputs=[x, w], outputs=[y], attrs=attrs)


def test_parse_conv_spec_success():
    node = _make_conv_node(
        {
            "kernel_shape": [3, 3],
            "strides": [2, 1],
            "dilations": [1, 1],
            "pads": [1, 2, 3, 4],
        }
    )

    spec = _parse_conv_spec(node)

    assert isinstance(spec, ConvSpec)
    assert spec.kernel_shape == [3, 3]
    assert spec.strides == [2, 1]
    assert spec.pads == [1, 2, 3, 4]


def test_parse_conv_spec_rejects_auto_pad():
    node = _make_conv_node(
        {
            "kernel_shape": [3, 3],
            "strides": [1, 1],
            "dilations": [1, 1],
            "pads": [1, 1, 1, 1],
            "auto_pad": "NOTSET",
        }
    )

    try:
        _parse_conv_spec(node)
    except AssertionError as exc:
        assert "auto_pad" in str(exc)
    else:
        assert False, "_parse_conv_spec should reject Conv nodes with auto_pad attr"


def test_parse_conv_spec_rejects_non_unit_dilation():
    node = _make_conv_node(
        {
            "kernel_shape": [3, 3],
            "strides": [1, 1],
            "dilations": [2, 1],
            "pads": [1, 1, 1, 1],
        }
    )

    try:
        _parse_conv_spec(node)
    except AssertionError as exc:
        assert "Conv dilations" in str(exc)
    else:
        assert False, "_parse_conv_spec should reject Conv dilation other than [1, 1]"


def test_conv_input_slice_for_output_handles_edge_overlap():
    spec = ConvSpec(kernel_shape=[3, 3], strides=[2, 1], pads=[1, 0, 1, 0])

    top = _conv_input_slice_for_output(0, 2, spec, h_in=6)
    assert (top.slice_start, top.slice_end) == (0, 4)
    assert (top.pad_top, top.pad_bottom) == (1, 0)

    bottom = _conv_input_slice_for_output(2, 4, spec, h_in=6)
    assert (bottom.slice_start, bottom.slice_end) == (3, 6)
    assert (bottom.pad_top, bottom.pad_bottom) == (0, 2)
