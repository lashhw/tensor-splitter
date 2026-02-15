from pathlib import Path
import sys

import numpy as np
import onnx
import onnx_graphsurgeon as gs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.config import GroupConfig
from ts.rewrite import rewrite_model


def _default_opset(model):
    return next(int(imp.version) for imp in model.opset_import if not imp.domain)


def _make_conv_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])
    weight = gs.Constant("W", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    bias = gs.Constant("B", values=np.random.randn(4).astype(np.float32))
    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 4, 8, 8])
    conv = gs.Node(
        op="Conv",
        inputs=[inp, weight, bias],
        outputs=[conv_out],
        attrs={
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
            "dilations": [1, 1],
        },
    )
    graph = gs.Graph(nodes=[conv], inputs=[inp], outputs=[conv_out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_padding_heavy_conv_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 2, 8, 5])
    weight = gs.Constant("W_pad", values=np.random.randn(3, 2, 1, 3).astype(np.float32))
    bias = gs.Constant("B_pad", values=np.random.randn(3).astype(np.float32))
    conv_out = gs.Variable("conv_out_pad", dtype=np.float32, shape=[1, 3, 6, 5])
    conv = gs.Node(
        op="Conv",
        inputs=[inp, weight, bias],
        outputs=[conv_out],
        attrs={
            "kernel_shape": [1, 3],
            "pads": [1, 1, 2, 1],
            "strides": [2, 1],
            "dilations": [1, 1],
        },
    )
    graph = gs.Graph(nodes=[conv], inputs=[inp], outputs=[conv_out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_auto_pad_notset_conv_model():
    inp = gs.Variable("input_auto_pad", dtype=np.float32, shape=[1, 3, 8, 8])
    weight = gs.Constant("W_auto_pad", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    bias = gs.Constant("B_auto_pad", values=np.random.randn(4).astype(np.float32))
    conv_out = gs.Variable("conv_out_auto_pad", dtype=np.float32, shape=[1, 4, 8, 8])
    conv = gs.Node(
        op="Conv",
        inputs=[inp, weight, bias],
        outputs=[conv_out],
        attrs={
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
            "dilations": [1, 1],
            "auto_pad": "NOTSET",
        },
    )
    graph = gs.Graph(nodes=[conv], inputs=[inp], outputs=[conv_out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def test_small_conv_rewrite_matches():
    model = _make_conv_model()
    groups = [GroupConfig(node_range=(0, 0), tile_count=(2, 1), execution_order=[(0, (0, 0)), (0, (1, 0))])]
    rewritten = rewrite_model(model, groups)

    import onnxruntime as ort

    sess_orig = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    out_orig = sess_orig.run(None, {"input": inp})[0]
    out_new = sess_new.run(None, {"input": inp})[0]

    np.testing.assert_allclose(out_orig, out_new, rtol=1e-5, atol=1e-6)


def test_small_conv_rewrite_matches_2d_tiles():
    model = _make_conv_model()
    groups = [
        GroupConfig(
            node_range=(0, 0),
            tile_count=(2, 2),
            execution_order=[(0, (0, 0)), (0, (0, 1)), (0, (1, 0)), (0, (1, 1))],
        )
    ]
    rewritten = rewrite_model(model, groups)

    import onnxruntime as ort

    sess_orig = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    out_orig = sess_orig.run(None, {"input": inp})[0]
    out_new = sess_new.run(None, {"input": inp})[0]

    np.testing.assert_allclose(out_orig, out_new, rtol=1e-5, atol=1e-6)


def test_entry_slices_follow_first_stage_schedule_order():
    model = _make_conv_model()
    groups = [GroupConfig(node_range=(0, 0), tile_count=(2, 1), execution_order=[(0, (1, 0)), (0, (0, 0))])]
    rewritten = rewrite_model(model, groups)
    graph = gs.import_onnx(rewritten)

    assert [node.op for node in graph.nodes] == ["Slice", "Conv", "Slice", "Conv", "Concat"]
    assert graph.nodes[1].inputs[0] is graph.nodes[0].outputs[0]
    assert graph.nodes[3].inputs[0] is graph.nodes[2].outputs[0]


def test_entry_slices_follow_first_stage_schedule_order_2d():
    model = _make_conv_model()
    groups = [
        GroupConfig(
            node_range=(0, 0),
            tile_count=(2, 2),
            execution_order=[(0, (1, 1)), (0, (0, 0)), (0, (1, 0)), (0, (0, 1))],
        )
    ]
    rewritten = rewrite_model(model, groups)
    graph = gs.import_onnx(rewritten)

    assert [node.op for node in graph.nodes[:8]] == ["Slice", "Conv"] * 4
    assert graph.nodes[1].inputs[0] is graph.nodes[0].outputs[0]
    assert graph.nodes[3].inputs[0] is graph.nodes[2].outputs[0]
    assert graph.nodes[5].inputs[0] is graph.nodes[4].outputs[0]
    assert graph.nodes[7].inputs[0] is graph.nodes[6].outputs[0]


def test_padding_only_tile_conv_rewrite_rejects():
    model = _make_padding_heavy_conv_model()
    groups = [
        GroupConfig(
            node_range=(0, 0),
            tile_count=(4, 1),
            execution_order=[(0, (0, 0)), (0, (1, 0)), (0, (2, 0)), (0, (3, 0))],
        )
    ]
    try:
        rewrite_model(model, groups)
    except AssertionError:
        pass
    else:
        assert False, "rewrite_model should reject groups that produce padding-only Conv tiles"


def test_rewrite_rejects_conv_with_auto_pad_attr():
    model = _make_auto_pad_notset_conv_model()
    groups = [GroupConfig(node_range=(0, 0), tile_count=(2, 1), execution_order=[(0, (0, 0)), (0, (1, 0))])]

    try:
        rewrite_model(model, groups)
    except AssertionError as exc:
        assert "auto_pad" in str(exc)
    else:
        assert False, "rewrite_model should reject Conv nodes with auto_pad in attrs"


def test_rewrite_upgrades_legacy_opset_model():
    model = _make_conv_model()
    legacy = onnx.version_converter.convert_version(model, 7)
    groups = [GroupConfig(node_range=(0, 0), tile_count=(2, 1), execution_order=[(0, (0, 0)), (0, (1, 0))])]
    rewritten = rewrite_model(legacy, groups)

    legacy_opset = _default_opset(legacy)
    rewritten_opset = _default_opset(rewritten)
    assert legacy_opset == 7
    assert rewritten_opset == 11

    import onnxruntime as ort

    sess_orig = ort.InferenceSession(legacy.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    out_orig = sess_orig.run(None, {"input": inp})[0]
    out_new = sess_new.run(None, {"input": inp})[0]

    np.testing.assert_allclose(out_orig, out_new, rtol=1e-5, atol=1e-6)


def test_rewrite_forces_target_opset_from_legacy_model():
    model = _make_conv_model()
    legacy = onnx.version_converter.convert_version(model, 7)
    groups = [GroupConfig(node_range=(0, 0), tile_count=(2, 1), execution_order=[(0, (0, 0)), (0, (1, 0))])]

    rewritten = rewrite_model(legacy, groups)

    assert _default_opset(legacy) == 7
    assert _default_opset(rewritten) == 11


def test_rewrite_forces_target_opset_from_newer_model():
    model = _make_conv_model()
    newer = onnx.version_converter.convert_version(model, 13)
    groups = [GroupConfig(node_range=(0, 0), tile_count=(2, 1), execution_order=[(0, (0, 0)), (0, (1, 0))])]

    rewritten = rewrite_model(newer, groups)

    assert _default_opset(newer) == 13
    assert _default_opset(rewritten) == 11


if __name__ == "__main__":
    test_small_conv_rewrite_matches()
    test_padding_only_tile_conv_rewrite_rejects()
    test_rewrite_rejects_conv_with_auto_pad_attr()
    test_rewrite_upgrades_legacy_opset_model()
    test_rewrite_forces_target_opset_from_legacy_model()
    test_rewrite_forces_target_opset_from_newer_model()
