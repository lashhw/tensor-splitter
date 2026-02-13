from pathlib import Path
import sys

import numpy as np
import onnx
import onnx_graphsurgeon as gs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts.verify import verify_models


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


def test_verify_models_returns_true_for_equal_models():
    model = _make_identity_model()
    ok, diffs = verify_models(model, model)

    assert ok
    assert diffs["out"] == 0.0


def test_verify_models_returns_false_for_numerical_mismatch():
    original = _make_identity_model()
    rewritten = _make_add_one_model()

    ok, diffs = verify_models(original, rewritten)

    assert not ok
    assert diffs["out"] > 0.0
