from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx_graphsurgeon as gs


def _get_attr(node: gs.Node, name: str, default: Any = None) -> Any:
    if node.attrs is None:
        return default
    return node.attrs.get(name, default)


def _as_int_list(value: Any, length: Optional[int] = None) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        out = [int(v) for v in value]
    else:
        out = [int(value)]
    assert length is None or len(out) == length, f"expected list of length {length}, got {out}"
    return out


def _conv_params(node: gs.Node) -> Tuple[List[int], List[int], List[int], List[int]]:
    auto_pad = _get_attr(node, "auto_pad", "NOTSET")
    assert auto_pad in (None, "NOTSET", ""), f"Conv auto_pad {auto_pad} is not supported"

    strides = _as_int_list(_get_attr(node, "strides", [1, 1]), length=2)
    dilations = _as_int_list(_get_attr(node, "dilations", [1, 1]), length=2)
    pads = _as_int_list(_get_attr(node, "pads", [0, 0, 0, 0]), length=4)
    kernel_shape = _as_int_list(_get_attr(node, "kernel_shape", None))

    if kernel_shape is None:
        assert len(node.inputs) >= 2, "Conv node missing weight input for kernel_shape inference"
        weight = node.inputs[1]
        assert hasattr(weight, "shape") and weight.shape is not None, (
            "Conv weight has no shape for kernel_shape inference"
        )
        assert len(weight.shape) >= 4, "Conv weight has invalid shape for kernel_shape inference"
        kernel_shape = [int(weight.shape[-2]), int(weight.shape[-1])]

    assert len(kernel_shape) == 2, f"Only 2D Conv supported; got kernel_shape {kernel_shape}"

    return kernel_shape, strides, dilations, pads


def _conv_attrs_with_height_pad(node: gs.Node, pads: Sequence[int]) -> Dict[str, Any]:
    attrs = dict(node.attrs) if node.attrs else {}
    attrs["pads"] = pads
    if "auto_pad" in attrs:
        attrs["auto_pad"] = "NOTSET"
    return attrs
