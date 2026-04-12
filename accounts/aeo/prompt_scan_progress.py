"""
Monitored prompt scan progress for AEO prompt coverage (counts + semantics).

Used by ``accounts.views._build_aeo_prompt_coverage_payload`` so tests can import
without loading the full views module (e.g. optional ``stripe`` dependency).
"""

from __future__ import annotations

from datetime import datetime, timezone as py_timezone
from typing import Any, Callable

from django.utils import timezone as django_timezone

_AEO_COVERAGE_PLATFORMS_FULL: tuple[str, ...] = ("openai", "gemini", "perplexity")
_AEO_COVERAGE_PLATFORM_SET = frozenset(_AEO_COVERAGE_PLATFORMS_FULL)


def monitored_prompt_keys_in_order(selected_aeo_prompts: list | None) -> list[str]:
    """Stable list of monitored prompt strings, deduped, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in selected_aeo_prompts or []:
        k = str(raw).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def prompt_scan_completed_count(
    monitored_keys: list[str],
    by_prompt: dict[str, list],
    latest_snapshot_per_platform: Callable[[list], dict[str, Any]],
) -> int:
    """
    Count monitored prompts that are fully scanned for progress UX.

    A prompt counts as **finished** only when the latest stored response for **each** of
    OpenAI, Gemini, and Perplexity has at least one extraction snapshot. This matches the
    stricter leg of ``_aeo_profile_visibility_pending`` (responses waiting for extraction)
    and is stricter than per-cell ``has_data`` in the coverage JSON (which can be true when
    a raw response exists). Pipeline runs still in ``PENDING``/``RUNNING`` or extraction
    stages are surfaced separately via ``visibility_pending`` on the API payload.
    """
    completed = 0
    for key in monitored_keys:
        rows = by_prompt.get(key, [])
        plat_latest = latest_snapshot_per_platform(rows)
        ok = True
        for plat in ("openai", "gemini", "perplexity"):
            if plat not in plat_latest:
                ok = False
                break
            resp = plat_latest[plat]
            if not resp.extraction_snapshots.exists():
                ok = False
                break
        if ok:
            completed += 1
    return completed


def monitored_prompt_keys_missing_full_coverage(profile: Any) -> list[str]:
    """
    Monitored prompts that lack full triple-provider coverage with extractions.

    Same completion rule as ``prompt_scan_completed_count`` / prompt-coverage UX: latest
    snapshot per OpenAI, Gemini, and Perplexity each must have ≥1 extraction. Used to
    backfill Phase 1 after plan expansion without re-running prompts that already satisfy
    coverage.
    """
    from accounts.models import AEOResponseSnapshot

    monitored = monitored_prompt_keys_in_order(profile.selected_aeo_prompts)
    if not monitored:
        return []

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
            return (datetime.min.replace(tzinfo=py_timezone.utc), x.id)
        if django_timezone.is_naive(c):
            c = django_timezone.make_aware(c, py_timezone.utc)
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

    missing: list[str] = []
    for key in monitored:
        rows = by_prompt.get(key, [])
        plat_latest = latest_snapshot_per_platform(rows)
        incomplete = False
        for plat in _AEO_COVERAGE_PLATFORMS_FULL:
            if plat not in plat_latest:
                incomplete = True
                break
            if not plat_latest[plat].extraction_snapshots.exists():
                incomplete = True
                break
        if incomplete:
            missing.append(key)
    return missing
