from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx_graphsurgeon as gs


def _get_attr(node: gs.Node, name: str, default: Any = None) -> Any:
    if node.attrs is None:
        return default
    return node.attrs.get(name, default)


def _as_int_list(value: Any, *, name: str, length: Optional[int] = None) -> List[int]:
    assert value is not None, f"Conv attribute {name} is required"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    assert isinstance(value, (tuple, list)), f"Conv attribute {name} must be a list/tuple; got {value!r}"
    out = [int(v) for v in value]
    assert length is None or len(out) == length, f"Conv attribute {name} must have length {length}; got {out}"
    return out


def _conv_params(node: gs.Node) -> Tuple[List[int], List[int], List[int], List[int]]:
    auto_pad = _get_attr(node, "auto_pad", "NOTSET")
    assert auto_pad in (None, "NOTSET", ""), f"Conv auto_pad {auto_pad} is not supported"

    kernel_shape = _as_int_list(_get_attr(node, "kernel_shape"), name="kernel_shape", length=2)
    strides = _as_int_list(_get_attr(node, "strides"), name="strides", length=2)
    dilations = _as_int_list(_get_attr(node, "dilations"), name="dilations", length=2)
    pads = _as_int_list(_get_attr(node, "pads"), name="pads", length=4)

    return kernel_shape, strides, dilations, pads


def _conv_attrs_with_height_pad(node: gs.Node, pads: List[int]) -> Dict[str, Any]:
    attrs = dict(node.attrs) if node.attrs else {}
    attrs["pads"] = pads
    if "auto_pad" in attrs:
        attrs["auto_pad"] = "NOTSET"
    return attrs
