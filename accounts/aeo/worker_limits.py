"""
Shared worker pool size for AEO parallel stages (Phase 1 execution, Phase 3 extraction).

Uses Django setting ``AEO_EXECUTION_MAX_WORKERS`` (default 20), clamped to 1–64.
"""

from __future__ import annotations

from django.conf import settings


def aeo_execution_max_workers() -> int:
    v = getattr(settings, "AEO_EXECUTION_MAX_WORKERS", 20)
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = 20
    return max(1, min(64, n))
