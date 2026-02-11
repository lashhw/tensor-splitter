"""Compatibility shim for the public rewrite API.

The implementation now lives in ``src/rewriter`` so rewrite logic is split
into smaller, easier-to-follow modules.
"""

from .rewriter import rewrite_model

__all__ = ["rewrite_model"]
