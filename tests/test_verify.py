from pathlib import Path
import sys

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.verify import _resolve_numpy_dtype, verify_model


def _make_identity_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 1, 2, 2])
    out = gs.Variable("out", dtype=np.float32, shape=[1, 1, 2, 2])
    node = gs.Node(op="Identity", inputs=[inp], outputs=[out])
    graph = gs.Graph(nodes=[node], inputs=[inp], outputs=[out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_add_one_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 1, 2, 2])
    one = gs.Constant("one", values=np.ones((1, 1, 2, 2), dtype=np.float32))
    out = gs.Variable("out", dtype=np.float32, shape=[1, 1, 2, 2])
    node = gs.Node(op="Add", inputs=[inp, one], outputs=[out])
    graph = gs.Graph(nodes=[node], inputs=[inp], outputs=[out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_two_input_sub_model(swap_inputs=False):
    a = gs.Variable("a", dtype=np.float32, shape=[1, 1, 2, 2])
    b = gs.Variable("b", dtype=np.float32, shape=[1, 1, 2, 2])
    out = gs.Variable("out", dtype=np.float32, shape=[1, 1, 2, 2])
    inputs = [b, a] if swap_inputs else [a, b]
    node = gs.Node(op="Sub", inputs=inputs, outputs=[out])
    graph = gs.Graph(nodes=[node], inputs=[a, b], outputs=[out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_reshaped_output_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 1, 2, 2])
    shape = gs.Constant("shape", values=np.array([1, 4], dtype=np.int64))
    out = gs.Variable("out", dtype=np.float32, shape=[1, 4])
    node = gs.Node(op="Reshape", inputs=[inp, shape], outputs=[out])
    graph = gs.Graph(nodes=[node], inputs=[inp], outputs=[out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_two_output_identity_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 1, 2, 2])
    out0 = gs.Variable("out0", dtype=np.float32, shape=[1, 1, 2, 2])
    out1 = gs.Variable("out1", dtype=np.float32, shape=[1, 1, 2, 2])
    node0 = gs.Node(op="Identity", inputs=[inp], outputs=[out0])
    node1 = gs.Node(op="Identity", inputs=[inp], outputs=[out1])
    graph = gs.Graph(nodes=[node0, node1], inputs=[inp], outputs=[out0, out1])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_dynamic_identity_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 1, None, 2])
    out = gs.Variable("out", dtype=np.float32, shape=[1, 1, None, 2])
    node = gs.Node(op="Identity", inputs=[inp], outputs=[out])
    graph = gs.Graph(nodes=[node], inputs=[inp], outputs=[out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def test_verify_model_returns_true_for_equal_models():
    model = _make_identity_model()
    ok, diffs = verify_model(model, model)

    assert ok
    assert diffs["out"] == 0.0


def test_verify_model_returns_false_for_numerical_mismatch():
    original = _make_identity_model()
    rewritten = _make_add_one_model()

    ok, diffs = verify_model(original, rewritten)

    assert not ok
    assert diffs["out"] > 0.0


def test_verify_model_uses_distinct_random_inputs_per_input_tensor():
    original = _make_two_input_sub_model(swap_inputs=False)
    rewritten = _make_two_input_sub_model(swap_inputs=True)

    ok, diffs = verify_model(original, rewritten)

    assert not ok
    assert diffs["out"] > 0.0


def test_verify_model_returns_inf_diff_on_shape_mismatch():
    original = _make_identity_model()
    rewritten = _make_reshaped_output_model()

    ok, diffs = verify_model(original, rewritten)

    assert not ok
    assert diffs["out"] == float("inf")


def test_verify_model_raises_on_output_count_mismatch():
    original = _make_identity_model()
    rewritten = _make_two_output_identity_model()

    with pytest.raises(
        AssertionError,
        match="output count mismatch between original and rewritten models",
    ):
        verify_model(original, rewritten)


def test_verify_model_raises_for_dynamic_input_shape():
    model = _make_dynamic_identity_model()

    with pytest.raises(
        AssertionError,
        match="input input must have static shapes for verification",
    ):
        verify_model(model, model)


def test_resolve_numpy_dtype_raises_on_unsupported_dtype():
    value_info = onnx.helper.make_tensor_value_info(
        "bad", onnx.TensorProto.UNDEFINED, [1]
    )

    with pytest.raises(AssertionError, match="input bad has unsupported dtype 0"):
        _resolve_numpy_dtype("bad", value_info)
