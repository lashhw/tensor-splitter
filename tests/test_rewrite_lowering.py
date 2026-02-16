from pathlib import Path
import sys

import numpy as np
import onnx_graphsurgeon as gs
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.rewrite.conv import _ConvInputSlice, _ConvInputSlice2D
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

    tiles, slice_nodes = _build_entry_tiles(entry, [((0, 3), (0, 8)), ((3, 8), (0, 8))], axis=2)

    assert len(tiles) == 2
    assert [node.op for node in slice_nodes] == ["Slice", "Slice"]
    assert tiles[0].shape == [1, 3, 3, 8]
    assert tiles[1].shape == [1, 3, 5, 8]
    assert list(slice_nodes[0].inputs[1].values) == [0, 0]
    assert list(slice_nodes[0].inputs[2].values) == [3, 8]
    assert list(slice_nodes[1].inputs[1].values) == [3, 0]
    assert list(slice_nodes[1].inputs[2].values) == [8, 8]


def test_build_entry_tiles_creates_2d_slice_nodes_with_expected_shapes():
    entry = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])

    tiles, slice_nodes = _build_entry_tiles(
        entry,
        [((0, 3), (0, 4)), ((0, 3), (4, 8)), ((3, 8), (0, 4)), ((3, 8), (4, 8))],
        axis=2,
    )

    assert len(tiles) == 4
    assert [node.op for node in slice_nodes] == ["Slice", "Slice", "Slice", "Slice"]
    assert tiles[0].shape == [1, 3, 3, 4]
    assert tiles[2].shape == [1, 3, 5, 4]
    assert list(slice_nodes[0].inputs[1].values) == [0, 0]
    assert list(slice_nodes[0].inputs[2].values) == [3, 4]
    assert list(slice_nodes[0].inputs[3].values) == [2, 3]


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
        _ConvInputSlice2D(
            height=_ConvInputSlice(slice_start=0, slice_end=4, pad_top=1, pad_bottom=0),
            width=_ConvInputSlice(slice_start=0, slice_end=8, pad_top=1, pad_bottom=1),
        ),
        _ConvInputSlice2D(
            height=_ConvInputSlice(slice_start=3, slice_end=8, pad_top=0, pad_bottom=2),
            width=_ConvInputSlice(slice_start=0, slice_end=8, pad_top=1, pad_bottom=1),
        ),
    ]

    out_tiles, conv_nodes = _build_stage_tiles(
        conv,
        tiles,
        out_ranges=[((0, 2), (0, 8)), ((2, 4), (0, 8))],
        main_input_index=0,
        conv_slices=conv_slices,
    )

    assert [node.op for node in conv_nodes] == ["Conv", "Conv"]
    assert conv_nodes[0].attrs["pads"] == [1, 1, 0, 1]
    assert conv_nodes[1].attrs["pads"] == [0, 1, 2, 1]
    assert out_tiles[0].shape == [1, 4, 2, 8]
    assert out_tiles[1].shape == [1, 4, 2, 8]


def test_build_stage_tiles_for_conv_rewrites_height_and_width_pads():
    conv = _make_conv_node()
    tiles = [
        gs.Variable("t0", dtype=np.float32, shape=[1, 3, 4, 5]),
        gs.Variable("t1", dtype=np.float32, shape=[1, 3, 4, 5]),
    ]
    conv_slices = [
        _ConvInputSlice2D(
            height=_ConvInputSlice(slice_start=0, slice_end=4, pad_top=1, pad_bottom=0),
            width=_ConvInputSlice(slice_start=0, slice_end=5, pad_top=1, pad_bottom=0),
        ),
        _ConvInputSlice2D(
            height=_ConvInputSlice(slice_start=0, slice_end=4, pad_top=1, pad_bottom=0),
            width=_ConvInputSlice(slice_start=3, slice_end=8, pad_top=0, pad_bottom=1),
        ),
    ]

    out_tiles, conv_nodes = _build_stage_tiles(
        conv,
        tiles,
        out_ranges=[((0, 2), (0, 4)), ((0, 2), (4, 8))],
        main_input_index=0,
        conv_slices=conv_slices,
    )

    assert [node.op for node in conv_nodes] == ["Conv", "Conv"]
    assert conv_nodes[0].attrs["pads"] == [1, 1, 0, 0]
    assert conv_nodes[1].attrs["pads"] == [1, 0, 0, 1]
    assert out_tiles[0].shape == [1, 4, 2, 4]
    assert out_tiles[1].shape == [1, 4, 2, 4]


def test_build_group_concat_uses_requested_axis_and_output_tensor():
    tiles = [
        gs.Variable("y_split0", dtype=np.float32, shape=[1, 4, 2, 8]),
        gs.Variable("y_split1", dtype=np.float32, shape=[1, 4, 2, 8]),
    ]
    output_tensor = gs.Variable("y", dtype=np.float32, shape=[1, 4, 4, 8])

    out, concat_nodes = _build_group_concat(
        tiles,
        axis=2,
        output_tensor=output_tensor,
        split_keys=[(0, 0), (1, 0)],
        tile_count=(2, 1),
    )

    assert len(concat_nodes) == 1
    assert concat_nodes[0].op == "Concat"
    assert concat_nodes[0].attrs["axis"] == 2
    assert out.name == "y"
    assert out.dtype == np.float32
    assert out.shape == [1, 4, 4, 8]


def test_build_group_concat_for_2d_tiles_uses_row_then_height_concat():
    tiles = [
        gs.Variable("y_split00", dtype=np.float32, shape=[1, 4, 2, 4]),
        gs.Variable("y_split01", dtype=np.float32, shape=[1, 4, 2, 4]),
        gs.Variable("y_split10", dtype=np.float32, shape=[1, 4, 2, 4]),
        gs.Variable("y_split11", dtype=np.float32, shape=[1, 4, 2, 4]),
    ]
    output_tensor = gs.Variable("y", dtype=np.float32, shape=[1, 4, 4, 8])

    out, concat_nodes = _build_group_concat(
        tiles,
        axis=2,
        output_tensor=output_tensor,
        split_keys=[(0, 0), (0, 1), (1, 0), (1, 1)],
        tile_count=(2, 2),
    )

    assert len(concat_nodes) == 3
    assert concat_nodes[0].op == "Concat"
    assert concat_nodes[0].attrs["axis"] == 3
    assert concat_nodes[1].op == "Concat"
    assert concat_nodes[1].attrs["axis"] == 3
    assert concat_nodes[2].op == "Concat"
    assert concat_nodes[2].attrs["axis"] == 2
    assert out.name == "y"
    assert out.dtype == np.float32
    assert out.shape == [1, 4, 4, 8]


def test_build_group_concat_requires_split_keys_and_tiles_to_match():
    tiles = [gs.Variable("y_split00", dtype=np.float32, shape=[1, 4, 2, 8])]
    output_tensor = gs.Variable("y", dtype=np.float32, shape=[1, 4, 2, 8])

    with pytest.raises(AssertionError, match="split_keys and tiles must have the same length"):
        _build_group_concat(
            tiles,
            axis=2,
            output_tensor=output_tensor,
            split_keys=[],
            tile_count=(1, 1),
        )
