import numpy as np
import onnx
from src import gs

from src.config import GroupConfig
from src.rewrite import rewrite_model


def _make_conv_model():
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])
    weight = gs.Constant("W", values=np.random.randn(4, 3, 3, 3).astype(np.float32))
    bias = gs.Constant("B", values=np.random.randn(4).astype(np.float32))
    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 4, 8, 8])
    conv = gs.Node(
        op="Conv",
        inputs=[inp, weight, bias],
        outputs=[conv_out],
        attrs={"pads": [1, 1, 1, 1], "strides": [1, 1]},
    )
    graph = gs.Graph(nodes=[conv], inputs=[inp], outputs=[conv_out])
    model = gs.export_onnx(graph)
    return onnx.shape_inference.infer_shapes(model)


def test_small_conv_rewrite_matches():
    model = _make_conv_model()
    groups = [GroupConfig(indices=(0, 0), splits=2, schedule=[(0, 0), (0, 1)])]
    rewritten = rewrite_model(model, groups)

    import onnxruntime as ort

    sess_orig = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    out_orig = sess_orig.run(None, {"input": inp})[0]
    out_new = sess_new.run(None, {"input": inp})[0]

    np.testing.assert_allclose(out_orig, out_new, rtol=1e-5, atol=1e-6)
