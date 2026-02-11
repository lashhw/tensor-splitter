"""Public package surface for tensor-splitter."""

from .config import GroupConfig, parse_config
from .rewrite import rewrite_model
from .verify import verify_models

__all__ = ["GroupConfig", "parse_config", "rewrite_model", "verify_models"]
