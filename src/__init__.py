"""Tensor splitter package initialization."""
from __future__ import annotations

import os
import sys

# Ensure local onnx_graphsurgeon is importable via the bundled TensorRT checkout.
try:
    import onnx_graphsurgeon as _gs  # type: ignore
except Exception:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    _GS_PATH = os.path.join(_ROOT, "TensorRT", "tools", "onnx-graphsurgeon")
    if _GS_PATH not in sys.path:
        sys.path.insert(0, _GS_PATH)
    import onnx_graphsurgeon as _gs  # type: ignore

gs = _gs

__all__ = ["gs"]
