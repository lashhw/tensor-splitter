from __future__ import annotations

from typing import Any, Dict, List, Tuple

import onnx_graphsurgeon as gs


def _get_attr(node: gs.Node, name: str, default: Any = None) -> Any:
    if node.attrs is None:
        return default
    return node.attrs.get(name, default)


def _as_int_list(value: Any, *, name: str, length: int) -> List[int]:
    assert value is not None, f"Conv attribute {name} is required"
    assert isinstance(value, list), f"Conv attribute {name} must be a list; got {value!r}"
    assert len(value) == length, f"Conv attribute {name} must have length {length}; got {value}"
    return value


def _conv_params(node: gs.Node) -> Tuple[List[int], List[int], List[int]]:
    attrs = node.attrs or {}
    assert "auto_pad" not in attrs, "Conv auto_pad must be unset in attrs"

    kernel_shape = _as_int_list(_get_attr(node, "kernel_shape"), name="kernel_shape", length=2)
    strides = _as_int_list(_get_attr(node, "strides"), name="strides", length=2)
    dilations = _as_int_list(_get_attr(node, "dilations"), name="dilations", length=2)
    pads = _as_int_list(_get_attr(node, "pads"), name="pads", length=4)
    assert dilations == [1, 1], f"Conv dilations {dilations} are not supported; expected [1, 1]"

    return kernel_shape, strides, pads


def _conv_attrs_with_height_pad(node: gs.Node, pads: List[int]) -> Dict[str, Any]:
    attrs = dict(node.attrs) if node.attrs else {}
    attrs["pads"] = pads
    return attrs
