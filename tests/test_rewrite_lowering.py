from pathlib import Path
import sys

import numpy as np
import onnx_graphsurgeon as gs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.rewrite.conv import _ConvInputSlice
from ts.rewrite.lowering import _build_entry_tiles, _build_group_concat, _build_stage_tiles


def _make_conv_node():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 3, 8, 8])
    w = gs.Constant("w", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    y = gs.Variable("y", dtype=np.float32, shape=[1, 4, 4, 8])
    return gs.Node(
        op="Conv",
        inputs=[x, w],
        outputs=[y],
        attrs={
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [2, 1],
            "dilations": [1, 1],
        },
    )


def test_build_entry_tiles_creates_slice_nodes_with_expected_shapes():
    entry = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])

    tiles, slice_nodes = _build_entry_tiles(entry, [(0, 3), (3, 8)], axis=2)

    assert len(tiles) == 2
    assert [node.op for node in slice_nodes] == ["Slice", "Slice"]
    assert tiles[0].shape == [1, 3, 3, 8]
    assert tiles[1].shape == [1, 3, 5, 8]
    assert int(slice_nodes[0].inputs[1].values[0]) == 0
    assert int(slice_nodes[0].inputs[2].values[0]) == 3
    assert int(slice_nodes[1].inputs[1].values[0]) == 3
    assert int(slice_nodes[1].inputs[2].values[0]) == 8


def test_build_stage_tiles_for_non_conv_propagates_tile_shapes():
    relu_in = gs.Variable("relu_in", dtype=np.float32, shape=[1, 4, 8, 8])
    relu_out = gs.Variable("relu_out", dtype=np.float32, shape=[1, 4, 8, 8])
    relu = gs.Node(op="Relu", inputs=[relu_in], outputs=[relu_out])
    tiles = [
        gs.Variable("t0", dtype=np.float32, shape=[1, 4, 3, 8]),
        gs.Variable("t1", dtype=np.float32, shape=[1, 4, 5, 8]),
    ]

    out_tiles, relu_nodes = _build_stage_tiles(
        relu,
        tiles,
        out_ranges=[(0, 3), (3, 8)],
        main_input_index=0,
    )

    assert [node.op for node in relu_nodes] == ["Relu", "Relu"]
    assert out_tiles[0].shape == [1, 4, 3, 8]
    assert out_tiles[1].shape == [1, 4, 5, 8]


def test_build_stage_tiles_for_conv_rewrites_height_pads():
    conv = _make_conv_node()
    tiles = [
        gs.Variable("t0", dtype=np.float32, shape=[1, 3, 4, 8]),
        gs.Variable("t1", dtype=np.float32, shape=[1, 3, 5, 8]),
    ]
    conv_slices = [
        _ConvInputSlice(slice_start=0, slice_end=4, pad_top=1, pad_bottom=0),
        _ConvInputSlice(slice_start=3, slice_end=8, pad_top=0, pad_bottom=2),
    ]

    out_tiles, conv_nodes = _build_stage_tiles(
        conv,
        tiles,
        out_ranges=[(0, 2), (2, 4)],
        main_input_index=0,
        conv_slices=conv_slices,
    )

    assert [node.op for node in conv_nodes] == ["Conv", "Conv"]
    assert conv_nodes[0].attrs["pads"] == [1, 1, 0, 1]
    assert conv_nodes[1].attrs["pads"] == [0, 1, 2, 1]
    assert out_tiles[0].shape == [1, 4, 2, 8]
    assert out_tiles[1].shape == [1, 4, 2, 8]


def test_build_group_concat_uses_requested_axis_and_output_tensor():
    tiles = [
        gs.Variable("y_split0", dtype=np.float32, shape=[1, 4, 2, 8]),
        gs.Variable("y_split1", dtype=np.float32, shape=[1, 4, 2, 8]),
    ]
    output_tensor = gs.Variable("y", dtype=np.float32, shape=[1, 4, 4, 8])

    out, concat_node = _build_group_concat(tiles, axis=2, output_tensor=output_tensor)

    assert concat_node.op == "Concat"
    assert concat_node.attrs["axis"] == 2
    assert out.name == "y"
    assert out.shape == [1, 4, 4, 8]
