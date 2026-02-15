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
    scale = gs.Constant("S", values=np.random.randn(4).astype(np.float32))
    bn_bias = gs.Constant("BB", values=np.random.randn(4).astype(np.float32))
    mean = gs.Constant("M", values=np.random.randn(4).astype(np.float32))
    var = gs.Constant("V", values=np.abs(np.random.randn(4)).astype(np.float32) + 1e-3)

    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 4, 10, 6])
    relu_out = gs.Variable("relu_out", dtype=np.float32, shape=[1, 4, 10, 6])
    bn_out = gs.Variable("bn_out", dtype=np.float32, shape=[1, 4, 10, 6])

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
    relu = gs.Node(op="Relu", inputs=[conv_out], outputs=[relu_out])
    bn = gs.Node(
        op="BatchNormalization",
        inputs=[relu_out, scale, bn_bias, mean, var],
        outputs=[bn_out],
        attrs={"epsilon": 1e-5},
    )

    graph = gs.Graph(nodes=[conv, relu, bn], inputs=[inp], outputs=[bn_out])
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
    except AssertionError as exc:
        assert "execution_order is not topologically valid" in str(exc)
    else:
        assert False, "rewrite_model should reject dependency-violating execution_order"


def _make_chain_with_add_constant_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 10, 6])
    weight = gs.Constant("W_bad", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    bias = gs.Constant("B_bad", values=np.random.randn(4).astype(np.float32))
    add_const = gs.Constant("C_bad", values=np.random.randn(1, 4, 1, 1).astype(np.float32))

    conv_out = gs.Variable("conv_bad_out", dtype=np.float32, shape=[1, 4, 10, 6])
    relu_out = gs.Variable("relu_bad_out", dtype=np.float32, shape=[1, 4, 10, 6])
    out = gs.Variable("add_bad_out", dtype=np.float32, shape=[1, 4, 10, 6])

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
    relu = gs.Node(op="Relu", inputs=[conv_out], outputs=[relu_out])
    add = gs.Node(op="Add", inputs=[relu_out, add_const], outputs=[out])

    graph = gs.Graph(nodes=[conv, relu, add], inputs=[inp], outputs=[out])
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
        attrs={
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
            "dilations": [1, 1],
        },
    )
    conv2 = gs.Node(
        op="Conv",
        inputs=[conv1_out, w2, b2],
        outputs=[conv2_out],
        attrs={
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
            "dilations": [1, 1],
        },
    )

    graph = gs.Graph(nodes=[conv1, conv2], inputs=[inp], outputs=[conv2_out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def _make_downsample_conv_chain_with_padding_model():
    rng = np.random.default_rng(1)
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 1, 8, 15])

    w0 = gs.Constant("W0", values=rng.standard_normal((3, 1, 1, 4)).astype(np.float32))
    b0 = gs.Constant("B0", values=rng.standard_normal(3).astype(np.float32))
    y0 = gs.Variable("y0", dtype=np.float32, shape=[1, 3, 6, 8])
    conv0 = gs.Node(
        op="Conv",
        inputs=[inp, w0, b0],
        outputs=[y0],
        attrs={
            "kernel_shape": [1, 4],
            "pads": [0, 0, 3, 3],
            "strides": [2, 2],
            "dilations": [1, 1],
        },
    )

    r0 = gs.Variable("r0", dtype=np.float32, shape=[1, 3, 6, 8])
    relu0 = gs.Node(op="Relu", inputs=[y0], outputs=[r0])

    w1 = gs.Constant("W1", values=rng.standard_normal((3, 3, 1, 4)).astype(np.float32))
    b1 = gs.Constant("B1", values=rng.standard_normal(3).astype(np.float32))
    y1 = gs.Variable("y1", dtype=np.float32, shape=[1, 3, 5, 4])
    conv1 = gs.Node(
        op="Conv",
        inputs=[r0, w1, b1],
        outputs=[y1],
        attrs={
            "kernel_shape": [1, 4],
            "pads": [0, 0, 3, 3],
            "strides": [2, 2],
            "dilations": [1, 1],
        },
    )

    graph = gs.Graph(nodes=[conv0, relu0, conv1], inputs=[inp], outputs=[y1])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def test_chain_rewrite_rejects_unsupported_add_op():
    model = _make_chain_with_add_constant_model()
    groups = [
        GroupConfig(
            node_range=(0, 2),
            tile_count=2,
            execution_order=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        )
    ]
    try:
        rewrite_model(model, groups)
    except AssertionError as exc:
        assert "unsupported op Add" in str(exc)
    else:
        assert False, "rewrite_model should reject Add in tiled groups"


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


def test_chain_rewrite_rejects_empty_intermediate_ranges():
    model = _make_downsample_conv_chain_with_padding_model()
    groups = [
        GroupConfig(
            node_range=(0, 2),
            tile_count=5,
            execution_order=[
                (node_idx, tile_id)
                for node_idx in range(3)
                for tile_id in range(5)
            ],
        )
    ]
    try:
        rewrite_model(model, groups)
    except AssertionError:
        pass
    else:
        assert False, "rewrite_model should reject groups that produce empty Conv intermediate ranges"


def _make_height_dilated_conv_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])
    weight = gs.Constant("W_dilated", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    bias = gs.Constant("B_dilated", values=np.random.randn(4).astype(np.float32))
    conv_out = gs.Variable("conv_dilated_out", dtype=np.float32, shape=[1, 4, 6, 8])
    conv = gs.Node(
        op="Conv",
        inputs=[inp, weight, bias],
        outputs=[conv_out],
        attrs={
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
            "dilations": [2, 1],
        },
    )
    graph = gs.Graph(nodes=[conv], inputs=[inp], outputs=[conv_out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def test_chain_rewrite_rejects_non_unit_conv_dilation():
    model = _make_height_dilated_conv_model()
    groups = [GroupConfig(node_range=(0, 0), tile_count=2, execution_order=[(0, 0), (0, 1)])]

    try:
        rewrite_model(model, groups)
    except AssertionError as exc:
        assert "Conv dilations" in str(exc)
    else:
        assert False, "rewrite_model should reject Conv dilation other than [1, 1]"


if __name__ == "__main__":
    test_chain_rewrite_matches()
    test_chain_rewrite_rejects_dependency_violating_execution_order()
    test_chain_rewrite_rejects_unsupported_add_op()
    test_multi_conv_rewrite_matches_and_uses_only_final_concat()
    test_chain_rewrite_rejects_empty_intermediate_ranges()
    test_chain_rewrite_rejects_non_unit_conv_dilation()
