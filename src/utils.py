"""Backwards-compatible re-exports.

Prefer importing from `src.group_analysis` in new code.
"""

from src.group_analysis import GroupInfo, analyze_group

__all__ = ["GroupInfo", "analyze_group"]
