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


def _make_chain_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 10, 6])
    weight = gs.Constant("W", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    bias = gs.Constant("B", values=np.random.randn(4).astype(np.float32))
    add_const = gs.Constant("C", values=np.random.randn(1, 4, 1, 1).astype(np.float32))

    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 4, 10, 6])
    relu_out = gs.Variable("relu_out", dtype=np.float32, shape=[1, 4, 10, 6])
    add_out = gs.Variable("add_out", dtype=np.float32, shape=[1, 4, 10, 6])

    conv = gs.Node(
        op="Conv",
        inputs=[inp, weight, bias],
        outputs=[conv_out],
        attrs={"pads": [1, 1, 1, 1], "strides": [1, 1]},
    )
    relu = gs.Node(op="Relu", inputs=[conv_out], outputs=[relu_out])
    add = gs.Node(op="Add", inputs=[relu_out, add_const], outputs=[add_out])

    graph = gs.Graph(nodes=[conv, relu, add], inputs=[inp], outputs=[add_out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def test_chain_rewrite_matches():
    model = _make_chain_model()
    groups = [
        GroupConfig(
            node_range=(0, 2),
            tile_count=2,
            execution_order=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        )
    ]
    rewritten = rewrite_model(model, groups)

    import onnxruntime as ort

    sess_orig = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 3, 10, 6)).astype(np.float32)
    out_orig = sess_orig.run(None, {"input": inp})[0]
    out_new = sess_new.run(None, {"input": inp})[0]

    np.testing.assert_allclose(out_orig, out_new, rtol=1e-5, atol=1e-6)


def test_chain_rewrite_rejects_dependency_violating_execution_order():
    model = _make_chain_model()
    groups = [
        GroupConfig(
            node_range=(0, 2),
            tile_count=2,
            execution_order=[(1, 0), (0, 0), (2, 0), (1, 1), (0, 1), (2, 1)],
        )
    ]

    try:
        rewrite_model(model, groups)
    except ValueError as exc:
        assert "execution_order is not topologically valid" in str(exc)
    else:
        raise AssertionError("rewrite_model should reject dependency-violating execution_order")


def _make_add_with_height_constant_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 10, 6])
    add_const = gs.Constant("C_height", values=np.random.randn(1, 3, 10, 1).astype(np.float32))
    out = gs.Variable("out", dtype=np.float32, shape=[1, 3, 10, 6])
    add = gs.Node(op="Add", inputs=[inp, add_const], outputs=[out])
    graph = gs.Graph(nodes=[add], inputs=[inp], outputs=[out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_two_conv_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 10, 6])
    w1 = gs.Constant("W1", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    b1 = gs.Constant("B1", values=np.random.randn(4).astype(np.float32))
    w2 = gs.Constant("W2", values=np.random.randn(4, 4, 3, 3).astype(np.float32))
    b2 = gs.Constant("B2", values=np.random.randn(4).astype(np.float32))

    conv1_out = gs.Variable("conv1_out", dtype=np.float32, shape=[1, 4, 10, 6])
    conv2_out = gs.Variable("conv2_out", dtype=np.float32, shape=[1, 4, 10, 6])

    conv1 = gs.Node(
        op="Conv",
        inputs=[inp, w1, b1],
        outputs=[conv1_out],
        attrs={"pads": [1, 1, 1, 1], "strides": [1, 1]},
    )
    conv2 = gs.Node(
        op="Conv",
        inputs=[conv1_out, w2, b2],
        outputs=[conv2_out],
        attrs={"pads": [1, 1, 1, 1], "strides": [1, 1]},
    )

    graph = gs.Graph(nodes=[conv1, conv2], inputs=[inp], outputs=[conv2_out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def test_binary_op_rewrite_slices_height_dependent_constants():
    model = _make_add_with_height_constant_model()
    groups = [GroupConfig(node_range=(0, 0), tile_count=2, execution_order=[(0, 0), (0, 1)])]
    rewritten = rewrite_model(model, groups)

    import onnxruntime as ort

    sess_orig = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 3, 10, 6)).astype(np.float32)
    out_orig = sess_orig.run(None, {"input": inp})[0]
    out_new = sess_new.run(None, {"input": inp})[0]

    np.testing.assert_allclose(out_orig, out_new, rtol=1e-5, atol=1e-6)


def test_multi_conv_rewrite_matches_and_uses_only_final_concat():
    model = _make_two_conv_model()
    groups = [
        GroupConfig(
            node_range=(0, 1),
            tile_count=2,
            execution_order=[(0, 0), (0, 1), (1, 0), (1, 1)],
        )
    ]
    rewritten = rewrite_model(model, groups)

    import onnxruntime as ort

    sess_orig = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 3, 10, 6)).astype(np.float32)
    out_orig = sess_orig.run(None, {"input": inp})[0]
    out_new = sess_new.run(None, {"input": inp})[0]

    np.testing.assert_allclose(out_orig, out_new, rtol=1e-5, atol=1e-6)

    graph = gs.import_onnx(rewritten)
    concat_nodes = [node for node in graph.nodes if node.op == "Concat"]
    assert len(concat_nodes) == 1


if __name__ == "__main__":
    test_chain_rewrite_matches()
    test_chain_rewrite_rejects_dependency_violating_execution_order()
    test_binary_op_rewrite_slices_height_dependent_constants()
    test_multi_conv_rewrite_matches_and_uses_only_final_concat()
