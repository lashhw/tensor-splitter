from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort

DEFAULT_VERIFY_RTOL = 1e-4
DEFAULT_VERIFY_ATOL = 1e-5


def _make_input_array(name: str, value_info: onnx.ValueInfoProto, rng: np.random.Generator) -> np.ndarray:
    tensor_type = value_info.type.tensor_type
    shape: list[int | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(None)
    assert not any(dim == 0 for dim in shape if dim is not None), (
        f"input {name} has invalid dimension 0"
    )
    assert not any(dim is None for dim in shape), (
        f"input {name} must have static shapes for verification"
    )

    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(tensor_type.elem_type)
    assert dtype is not None, f"input {name} has unsupported dtype {tensor_type.elem_type}"

    static_shape = tuple(int(dim) for dim in shape)
    data = rng.standard_normal(size=static_shape).astype(dtype)
    return data


def verify_model(
    original: onnx.ModelProto,
    rewritten: onnx.ModelProto,
    rtol: float = DEFAULT_VERIFY_RTOL,
    atol: float = DEFAULT_VERIFY_ATOL,
) -> Tuple[bool, Dict[str, float]]:
    sess_options = ort.SessionOptions()
    sess_original = ort.InferenceSession(original.SerializeToString(), sess_options, providers=["CPUExecutionProvider"])
    sess_rewritten = ort.InferenceSession(rewritten.SerializeToString(), sess_options, providers=["CPUExecutionProvider"])

    inputs: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    initializer_names = {init.name for init in original.graph.initializer}
    for inp in original.graph.input:
        if inp.name in initializer_names:
            continue
        inputs[inp.name] = _make_input_array(inp.name, inp, rng)

    orig_outs = sess_original.run(None, inputs)
    new_outs = sess_rewritten.run(None, inputs)

    assert len(orig_outs) == len(new_outs), (
        "output count mismatch between original and rewritten models"
    )

    diffs: Dict[str, float] = {}
    ok = True
    for idx, (orig, new) in enumerate(zip(orig_outs, new_outs)):
        name = original.graph.output[idx].name
        if orig.shape != new.shape:
            diffs[name] = float("inf")
            ok = False
            continue

        diff = float(np.max(np.abs(orig - new)))
        diffs[name] = diff
        if not np.allclose(orig, new, rtol=rtol, atol=atol):
            ok = False

    return ok, diffs
