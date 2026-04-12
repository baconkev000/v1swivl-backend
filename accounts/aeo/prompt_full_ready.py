"""
Fully-ready rules for AI Visibility prompt rows + ETA for the single progress bar.

``fully_ready`` (per monitored prompt) requires all of:
1) **Triple extraction** — same rule as ``prompt_scan_completed_count`` / latest snapshot per
   openai, gemini, perplexity each has ≥1 extraction snapshot.
2) **Phase-2 passes complete** — mirrors ``run_aeo_phase2_confidence_task`` (OpenAI, Gemini, and
   Perplexity when ``perplexity_execution_enabled()``): no further ``_push`` for in-scope providers.
3) **Recommendations slice** — aligned with ``_aeo_recommendations_pipeline_pending`` in
   ``accounts.views``: when ``AEO_ENABLE_RECOMMENDATION_STAGE`` is off, nothing blocks; when on,
   Phase 5 must not be pending on the latest ``AEOExecutionRun`` (persisted recommendations pass
   finished). Per-prompt ``improvement_recommendations`` may still be empty once Phase 5 is done.

ETA: duration from **first** ``AEOResponseSnapshot.created_at`` for this prompt text on the
profile (earliest artifact we store) until the moment we first record ``fully_ready`` for that
prompt (deduped via ``recorded_hashes`` in profile JSON). Rolling mean of the last K samples
(``AEO_FULL_PHASE_ETA_K``).
"""

from __future__ import annotations

from datetime import datetime, timezone as py_timezone
from typing import Any, Callable

from django.conf import settings
from django.db.models import Min
from django.utils import timezone as django_timezone

from accounts.aeo.aeo_execution_utils import hash_prompt
from accounts.aeo.perplexity_execution_utils import perplexity_execution_enabled
from accounts.aeo.progressive_onboarding import PASSES_PER_PROVIDER_TARGET
from accounts.aeo.prompt_scan_progress import prompt_scan_completed_count
from accounts.models import (
    AEOPromptExecutionAggregate,
    AEOResponseSnapshot,
    AEOExecutionRun,
    BusinessProfile,
)


def phase2_passes_complete(agg: AEOPromptExecutionAggregate | None) -> bool:
    """
    True when Phase-2 scheduling in ``run_aeo_phase2_confidence_task`` would enqueue no further
    work for this aggregate (mirrors ``_push`` for OpenAI, Gemini, and Perplexity when enabled).
    """
    if agg is None:
        return False
    o_count = int(agg.openai_pass_count or 0)
    g_count = int(agg.gemini_pass_count or 0)
    p_count = int(agg.perplexity_pass_count or 0)
    gemini_only_aggregate = o_count == 0 and g_count > 0
    openai_only_aggregate = g_count == 0 and o_count > 0
    perplexity_only_aggregate = p_count > 0 and o_count == 0 and g_count == 0
    p2_on = perplexity_execution_enabled()

    openai_needed = False
    if o_count < PASSES_PER_PROVIDER_TARGET:
        if not gemini_only_aggregate and not perplexity_only_aggregate:
            openai_needed = True
    elif (
        bool(agg.openai_third_pass_required)
        and not bool(agg.openai_third_pass_ran)
        and o_count < 3
    ):
        if not gemini_only_aggregate and not perplexity_only_aggregate:
            openai_needed = True

    gemini_needed = False
    if g_count < PASSES_PER_PROVIDER_TARGET:
        if not openai_only_aggregate and not perplexity_only_aggregate:
            gemini_needed = True
    elif (
        bool(agg.gemini_third_pass_required)
        and not bool(agg.gemini_third_pass_ran)
        and g_count < 3
    ):
        if not openai_only_aggregate and not perplexity_only_aggregate:
            gemini_needed = True

    perplexity_needed = False
    if p2_on:
        if p_count < PASSES_PER_PROVIDER_TARGET:
            if not openai_only_aggregate and not gemini_only_aggregate:
                perplexity_needed = True
        elif (
            bool(agg.perplexity_third_pass_required)
            and not bool(agg.perplexity_third_pass_ran)
            and p_count < 3
        ):
            if not openai_only_aggregate and not gemini_only_aggregate:
                perplexity_needed = True

    return not openai_needed and not gemini_needed and not perplexity_needed


def get_latest_aggregate_for_prompt(profile_id: int, prompt_text: str) -> AEOPromptExecutionAggregate | None:
    h = hash_prompt(prompt_text)
    return (
        AEOPromptExecutionAggregate.objects.filter(profile_id=profile_id, prompt_hash=h)
        .order_by("-updated_at", "-id")
        .first()
    )


def triple_extraction_complete_for_key(
    key: str,
    by_prompt: dict[str, list],
    latest_snapshot_per_platform: Callable[[list], dict[str, Any]],
) -> bool:
    return (
        prompt_scan_completed_count([key], by_prompt, latest_snapshot_per_platform) == 1
    )


def recommendations_pipeline_settled_for_visibility(profile: BusinessProfile) -> bool:
    """
    When Phase 5 is disabled, recommendations never block visibility readiness.
    When enabled, align with ``_aeo_recommendations_pipeline_pending`` in views: block until the
    latest run has **finished scoring** and is **not** in the recommendation sub-stage
    (pending/running). If there is no run yet, Phase 5 is not settled.
    """
    if not bool(getattr(settings, "AEO_ENABLE_RECOMMENDATION_STAGE", False)):
        return True
    latest_ex = (
        AEOExecutionRun.objects.filter(profile=profile).order_by("-created_at", "-id").first()
    )
    if latest_ex is None:
        return True
    if latest_ex.scoring_status != AEOExecutionRun.STAGE_COMPLETED:
        return False
    return latest_ex.recommendation_status not in (
        AEOExecutionRun.STAGE_PENDING,
        AEOExecutionRun.STAGE_RUNNING,
    )


def monitored_prompt_fully_ready(
    monitored_key: str,
    profile: BusinessProfile,
    by_prompt: dict[str, list],
    latest_snapshot_per_platform: Callable[[list], dict[str, Any]],
    recs_settled: bool,
) -> bool:
    if not recs_settled:
        return False
    if not triple_extraction_complete_for_key(monitored_key, by_prompt, latest_snapshot_per_platform):
        return False
    agg = get_latest_aggregate_for_prompt(profile.id, monitored_key)
    return phase2_passes_complete(agg)


def prompt_fully_ready(
    monitored_key: str,
    profile: BusinessProfile,
    by_prompt: dict[str, list],
    latest_snapshot_per_platform: Callable[[list], dict[str, Any]],
) -> bool:
    return monitored_prompt_fully_ready(
        monitored_key,
        profile,
        by_prompt,
        latest_snapshot_per_platform,
        recommendations_pipeline_settled_for_visibility(profile),
    )


def first_snapshot_created_at(profile_id: int, prompt_text: str) -> datetime | None:
    m = (
        AEOResponseSnapshot.objects.filter(profile_id=profile_id, prompt_text=prompt_text)
        .aggregate(m=Min("created_at"))
        .get("m")
    )
    return m


def _eta_k() -> int:
    try:
        return max(1, int(getattr(settings, "AEO_FULL_PHASE_ETA_K", 5)))
    except (TypeError, ValueError):
        return 5


def _default_eta_seconds_per_prompt() -> float:
    try:
        return max(1.0, float(getattr(settings, "AEO_FULL_PHASE_ETA_DEFAULT_SEC", 120)))
    except (TypeError, ValueError):
        return 120.0


def _eta_cap_seconds() -> float:
    try:
        return max(60.0, float(getattr(settings, "AEO_FULL_PHASE_ETA_CAP_SEC", 3600)))
    except (TypeError, ValueError):
        return 3600.0


def full_phase_eta_cold_start(durations: list[float], full_phase_completed: int) -> bool:
    """No completion samples yet, or nothing finished in the current onboarding wave."""
    if full_phase_completed == 0 and len(durations) == 0:
        return True
    if len(durations) == 0:
        return True
    return False


def compute_full_phase_eta_seconds(
    durations: list[float],
    remaining: int,
    full_phase_completed: int,
) -> tuple[int | None, bool]:
    """
    Returns (eta_seconds capped or None when cold-start, cold_start flag).
    ``remaining`` should already be clamped (e.g. max(0, effective_target - completed)).
    """
    cap = _eta_cap_seconds()
    if remaining <= 0:
        return 0, False
    cold = full_phase_eta_cold_start(durations, full_phase_completed)
    if cold:
        return None, True
    avg = sum(durations) / len(durations)
    raw = float(remaining) * float(avg)
    eta = int(min(cap, max(0.0, raw)))
    return eta, False


def merge_eta_state_after_completions(
    profile: BusinessProfile,
    monitored_keys: list[str],
    per_key_fully_ready: dict[str, bool],
    now: datetime | None = None,
) -> list[float]:
    """
    Append durations for prompts that just became fully_ready (by prompt hash dedupe).
    Mutates and saves ``profile.aeo_full_phase_eta_state`` when needed.
    Returns the post-update durations list (for averaging).
    """
    if now is None:
        now = django_timezone.now()
    state = profile.aeo_full_phase_eta_state
    if not isinstance(state, dict):
        state = {}
    durations: list[float] = [float(x) for x in (state.get("durations") or []) if x is not None]
    recorded: set[str] = set(str(x) for x in (state.get("recorded_hashes") or []) if x)
    k = _eta_k()
    changed = False
    for key in monitored_keys:
        if not per_key_fully_ready.get(key):
            continue
        h = hash_prompt(key)
        if h in recorded:
            continue
        first_at = first_snapshot_created_at(profile.id, key)
        if first_at is None:
            continue
        if django_timezone.is_naive(first_at):
            first_at = django_timezone.make_aware(first_at, py_timezone.utc)
        delta = (now - first_at).total_seconds()
        durations.append(max(1.0, float(delta)))
        durations = durations[-k:]
        recorded.add(h)
        changed = True
    if changed:
        profile.aeo_full_phase_eta_state = {
            "durations": durations,
            "recorded_hashes": sorted(recorded),
        }
        profile.save(update_fields=["aeo_full_phase_eta_state", "updated_at"])
    return durations
