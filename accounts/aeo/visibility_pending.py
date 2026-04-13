"""
Structured breakdown for ``visibility_pending`` (matches ``_aeo_profile_visibility_pending`` logic).

Used by the prompt-coverage API for diagnostics and repair gating.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from django.utils import timezone as django_timezone

from accounts.aeo.prompt_scan_progress import monitored_prompt_keys_in_order

if TYPE_CHECKING:
    from accounts.models import BusinessProfile

_AEO_COVERAGE_PLATFORM_SET = frozenset({"openai", "gemini", "perplexity"})
_AEO_PENDING_PLATFORM_ORDER = ("openai", "perplexity", "gemini")


def aeo_visibility_pending_breakdown(profile: BusinessProfile) -> dict[str, Any]:
    """
    Return flags that explain why ``visibility_pending`` is true.

    Keys:
    - execution_inflight: a run is pending or running (Phase 1 batch).
    - latest_run_extractions_inflight: newest run is completed but Phase 3 extractions not finished.
    - snapshots_awaiting_extraction: a monitored prompt has a latest per-platform response with
      zero extraction snapshots (Phase 3 not yet applied to that snapshot).
    - visibility_pending: OR of the above (same as legacy ``_aeo_profile_visibility_pending``).
    """
    from accounts.models import AEOResponseSnapshot, AEOExecutionRun

    monitored = monitored_prompt_keys_in_order(profile.selected_aeo_prompts)
    if not monitored:
        return {
            "execution_inflight": False,
            "latest_run_extractions_inflight": False,
            "snapshots_awaiting_extraction": False,
            "visibility_pending": False,
        }

    execution_inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exists()

    latest_run_extractions_inflight = False
    latest_run = (
        AEOExecutionRun.objects.filter(profile=profile).order_by("-created_at", "-id").first()
    )
    if latest_run is not None and latest_run.status == AEOExecutionRun.STATUS_COMPLETED:
        if latest_run.extraction_status in (
            AEOExecutionRun.STAGE_PENDING,
            AEOExecutionRun.STAGE_RUNNING,
        ):
            latest_run_extractions_inflight = True

    responses = list(
        AEOResponseSnapshot.objects.filter(profile=profile)
        .order_by("-created_at", "-id")
        .prefetch_related("extraction_snapshots")
    )
    by_prompt: dict[str, list] = {}
    for resp in responses:
        key = (resp.prompt_text or "").strip()
        if not key:
            continue
        by_prompt.setdefault(key, []).append(resp)

    def _response_sort_key(x: Any) -> tuple:
        c = x.created_at
        if c is None:
            return (datetime.min.replace(tzinfo=timezone.utc), x.id)
        if django_timezone.is_naive(c):
            c = django_timezone.make_aware(c, timezone.utc)
        return (c, x.id)

    def latest_snapshot_per_platform(rows: list) -> dict[str, Any]:
        best: dict[str, Any] = {}
        for r in sorted(rows, key=_response_sort_key, reverse=True):
            plat = str(r.platform or "").strip().lower()
            if plat not in _AEO_COVERAGE_PLATFORM_SET:
                continue
            if plat not in best:
                best[plat] = r
        return best

    snapshots_awaiting_extraction = False
    for key in monitored:
        rows = by_prompt.get(key, [])
        plat_latest = latest_snapshot_per_platform(rows)
        for plat in _AEO_PENDING_PLATFORM_ORDER:
            if plat not in plat_latest:
                continue
            resp = plat_latest[plat]
            if not resp.extraction_snapshots.exists():
                snapshots_awaiting_extraction = True
                break
        if snapshots_awaiting_extraction:
            break

    visibility_pending = (
        execution_inflight or latest_run_extractions_inflight or snapshots_awaiting_extraction
    )

    return {
        "execution_inflight": execution_inflight,
        "latest_run_extractions_inflight": latest_run_extractions_inflight,
        "snapshots_awaiting_extraction": snapshots_awaiting_extraction,
        "visibility_pending": visibility_pending,
    }


def aeo_visibility_pipeline_quiescent(profile: BusinessProfile) -> bool:
    """True when no visibility-related pipeline leg is actively processing."""
    b = aeo_visibility_pending_breakdown(profile)
    return not b["visibility_pending"]
