"""
Celery tasks for SEO enrichment. Run after initial snapshot is saved; update only enrichment
fields and never recalculate main score (seo_score, search_performance_score, search_visibility_percent,
missed_searches_monthly).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Any, Dict, List

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone as django_tz

from .aeo.aeo_plan_targets import (
    aeo_effective_monitored_target_for_profile,
    aeo_http_call_bounds_for_monitoring,
    aeo_should_run_post_payment_expansion,
)

logger = logging.getLogger(__name__)

# Duplicate prevention: skip keyword enrichment if already done within this TTL.
# Requirement: ranks + competitor enrichment should stay stable for 7 days unless
# user explicitly clicks the refresh button.
KEYWORDS_ENRICHMENT_TTL = timedelta(days=7)
# Next steps: reuse existing TTL from generate_or_get_next_steps.
NEXT_STEPS_TTL = timedelta(days=7)
# Per-keyword action suggestions TTL (for \"Do these now\" UI).
KEYWORD_ACTION_TTL = timedelta(days=7)
# Cap keyword rows passed into LLM rewrite for per-keyword suggestions (cost control).
KEYWORD_ACTION_SUGGESTIONS_MAX_ROWS = 80


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=2, retry_backoff=60, ignore_result=True)
def post_payment_seo_snapshot_task(self, profile_id: int) -> None:
    """
    Stripe webhook follow-up: run a full SEO overview refresh for the workspace.

    Invoked from ``transaction.on_commit`` in ``apply_subscription_payload_to_profile`` only (never
    inline in the webhook). Uses ``force_refresh=True`` so post-payment users get a fresh snapshot
    and the existing pipeline can enqueue enrichment / ``generate_snapshot_next_steps_task`` /
    ``generate_keyword_action_suggestions_task``.
    """
    from .business_profile_access import workspace_data_user
    from .dataforseo_utils import get_or_refresh_seo_score_for_user
    from .models import BusinessProfile

    pid = int(profile_id)
    profile = BusinessProfile.objects.filter(pk=pid).select_related("user").first()
    if profile is None:
        logger.warning("[stripe->SEO] profile_not_found profile_id=%s", pid)
        return
    site = str(getattr(profile, "website_url", "") or "").strip()
    if not site:
        logger.info("[stripe->SEO] skip_no_website profile_id=%s", pid)
        return
    data_user = workspace_data_user(profile) or profile.user
    if data_user is None:
        logger.warning("[stripe->SEO] skip_no_data_user profile_id=%s", pid)
        return
    try:
        bundle = get_or_refresh_seo_score_for_user(
            data_user,
            site_url=site,
            force_refresh=True,
            business_profile=profile,
        )
        logger.info(
            "[stripe->SEO] refresh_done profile_id=%s user_id=%s has_bundle=%s",
            pid,
            getattr(data_user, "id", None),
            bundle is not None,
        )
        from .seo_snapshot_refresh import sync_enrich_current_period_seo_snapshot_for_profile

        sync_result = sync_enrich_current_period_seo_snapshot_for_profile(
            profile,
            abort_on_low_coverage=True,
        )
        if sync_result.get("aborted_low_coverage"):
            logger.warning(
                "[stripe->SEO] sync_enrich_aborted_low_coverage profile_id=%s snapshot_id=%s detail=%s",
                pid,
                sync_result.get("snapshot_id"),
                sync_result.get("detail"),
            )
        elif sync_result.get("persisted"):
            logger.info(
                "[stripe->SEO] sync_enrich_persisted profile_id=%s snapshot_id=%s",
                pid,
                sync_result.get("snapshot_id"),
            )
        elif sync_result.get("detail"):
            logger.info(
                "[stripe->SEO] sync_enrich_skipped profile_id=%s detail=%s",
                pid,
                sync_result.get("detail"),
            )
    except Exception:
        logger.exception("[stripe->SEO] refresh_failed profile_id=%s", pid)
        raise


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=2, retry_backoff=30, ignore_result=True)
def sync_enrich_seo_snapshot_for_profile_task(self, profile_id: int) -> None:
    """
    Synchronously enrich the current-period SEO snapshot (gap + LLM + Labs ranks, metrics persist).

    Call after ``get_or_refresh_seo_score_for_user`` has created/updated the ranked-keyword row
    (e.g. new business profile POST). Mirrors the second half of ``post_payment_seo_snapshot_task``.
    """
    from .business_profile_access import workspace_data_user
    from .models import BusinessProfile
    from .seo_snapshot_refresh import sync_enrich_current_period_seo_snapshot_for_profile

    pid = int(profile_id)
    profile = BusinessProfile.objects.filter(pk=pid).select_related("user").first()
    if profile is None:
        logger.warning("[SEO sync enrich task] profile_not_found profile_id=%s", pid)
        return
    data_user = workspace_data_user(profile) or profile.user
    if data_user is None:
        logger.warning("[SEO sync enrich task] skip_no_data_user profile_id=%s", pid)
        return
    try:
        sync_result = sync_enrich_current_period_seo_snapshot_for_profile(
            profile,
            data_user_fallback=data_user,
            abort_on_low_coverage=True,
        )
        if sync_result.get("aborted_low_coverage"):
            logger.warning(
                "[SEO sync enrich task] aborted_low_coverage profile_id=%s snapshot_id=%s detail=%s",
                pid,
                sync_result.get("snapshot_id"),
                sync_result.get("detail"),
            )
        elif sync_result.get("persisted"):
            logger.info(
                "[SEO sync enrich task] persisted profile_id=%s snapshot_id=%s",
                pid,
                sync_result.get("snapshot_id"),
            )
        elif sync_result.get("detail"):
            logger.info(
                "[SEO sync enrich task] skipped profile_id=%s detail=%s",
                pid,
                sync_result.get("detail"),
            )
    except Exception:
        logger.exception("[SEO sync enrich task] failed profile_id=%s", pid)
        raise


def _seo_datetime_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def seo_data_dict_from_seo_overview_snapshot(snapshot: Any) -> Dict[str, Any]:
    """
    Build a broad ``seo_data`` dict from a persisted ``SEOOverviewSnapshot`` (snapshot JSON only;
    no new DataForSEO calls). Used by Celery next-steps tasks and API refresh paths so the issue
    engine + rewrite layer see the same surface as the stored snapshot.
    """
    issues = getattr(snapshot, "seo_structured_issues", None) or []
    if not isinstance(issues, list):
        issues = []
    kws = list(getattr(snapshot, "top_keywords", None) or [])
    snapshot_context_for_rewrite = {
        "cached_domain": str(getattr(snapshot, "cached_domain", "") or ""),
        "cached_location_mode": str(getattr(snapshot, "cached_location_mode", "") or "organic"),
        "cached_location_code": int(getattr(snapshot, "cached_location_code", 0) or 0),
        "cached_location_label": str(getattr(snapshot, "cached_location_label", "") or ""),
        "local_verification_applied": bool(getattr(snapshot, "local_verification_applied", False)),
        "local_verified_keyword_count": int(getattr(snapshot, "local_verified_keyword_count", 0) or 0),
        "refreshed_at": _seo_datetime_iso(getattr(snapshot, "refreshed_at", None)),
        "keywords_enriched_at": _seo_datetime_iso(getattr(snapshot, "keywords_enriched_at", None)),
        "seo_next_steps_refreshed_at": _seo_datetime_iso(getattr(snapshot, "seo_next_steps_refreshed_at", None)),
        "keyword_action_suggestions_refreshed_at": _seo_datetime_iso(
            getattr(snapshot, "keyword_action_suggestions_refreshed_at", None)
        ),
        "snapshot_id": int(snapshot.pk),
    }
    score = int(getattr(snapshot, "search_performance_score", 0) or 0)
    return {
        "seo_score": score,
        "search_performance_score": score,
        "organic_visitors": int(getattr(snapshot, "organic_visitors", 0) or 0),
        "total_search_volume": int(getattr(snapshot, "total_search_volume", 0) or 0),
        "estimated_search_appearances_monthly": int(
            getattr(snapshot, "estimated_search_appearances_monthly", 0) or 0
        ),
        "missed_searches_monthly": int(getattr(snapshot, "missed_searches_monthly", 0) or 0),
        "search_visibility_percent": int(getattr(snapshot, "search_visibility_percent", 0) or 0),
        "keywords_ranking": int(getattr(snapshot, "keywords_ranking", 0) or 0),
        "top3_positions": int(getattr(snapshot, "top3_positions", 0) or 0),
        "top_keywords": kws,
        "seo_structured_issues": list(issues),
        "cached_domain": str(getattr(snapshot, "cached_domain", "") or ""),
        "cached_location_mode": str(getattr(snapshot, "cached_location_mode", "") or "organic"),
        "cached_location_code": int(getattr(snapshot, "cached_location_code", 0) or 0),
        "cached_location_label": str(getattr(snapshot, "cached_location_label", "") or ""),
        "local_verification_applied": bool(getattr(snapshot, "local_verification_applied", False)),
        "local_verified_keyword_count": int(getattr(snapshot, "local_verified_keyword_count", 0) or 0),
        "refreshed_at": _seo_datetime_iso(getattr(snapshot, "refreshed_at", None)),
        "keywords_enriched_at": _seo_datetime_iso(getattr(snapshot, "keywords_enriched_at", None)),
        "snapshot_context_for_rewrite": snapshot_context_for_rewrite,
    }


def _seo_snapshot_corpus_newer_than_next_steps(snapshot: Any) -> bool:
    """
    When ranked keywords or headline metrics were refreshed after the last next-steps run,
    bypass ``NEXT_STEPS_TTL`` so a new Celery pass can consume enriched ``top_keywords``.
    """
    steps_at = getattr(snapshot, "seo_next_steps_refreshed_at", None)
    if not steps_at:
        return True
    enriched = getattr(snapshot, "keywords_enriched_at", None)
    if enriched and enriched > steps_at:
        return True
    refreshed = getattr(snapshot, "refreshed_at", None)
    if refreshed and refreshed > steps_at:
        return True
    return False


def _seo_snapshot_corpus_newer_than_keyword_suggestions(snapshot: Any) -> bool:
    steps_at = getattr(snapshot, "keyword_action_suggestions_refreshed_at", None)
    if not steps_at:
        return True
    enriched = getattr(snapshot, "keywords_enriched_at", None)
    if enriched and enriched > steps_at:
        return True
    refreshed = getattr(snapshot, "refreshed_at", None)
    if refreshed and refreshed > steps_at:
        return True
    return False


def _is_onboarding_sample_size_profile(profile) -> bool:
    """Starter / unpaid: lighter monitored set. Pro+ get full pipeline staging."""
    from .models import BusinessProfile

    slug = str(getattr(profile, "plan", "") or "").strip().lower()
    if slug in (
        BusinessProfile.PLAN_PRO,
        "professional",
        BusinessProfile.PLAN_ADVANCED,
        "enterprise",
        "scale",
    ):
        return False
    return True


def _aeo_recommendation_stage_enabled() -> bool:
    return bool(getattr(settings, "AEO_ENABLE_RECOMMENDATION_STAGE", False))


def _aeo_phase3_extract_one_snapshot(snapshot_id: int, run_started_at: Any) -> tuple[int, int]:
    """
    Run extraction for one response snapshot in a thread worker.
    Returns (created_increment, failed_increment) for aggregation.
    """
    from django.db import close_old_connections

    from .aeo.aeo_extraction_utils import run_single_extraction
    from .models import AEOResponseSnapshot

    close_old_connections()
    try:
        snap = AEOResponseSnapshot.objects.select_related("profile").filter(pk=snapshot_id).first()
        if snap is None:
            return 0, 1
        if run_started_at is not None:
            if snap.extraction_snapshots.filter(created_at__gte=run_started_at).exists():
                return 0, 0
        result = run_single_extraction(snap, save=True)
        if result.get("save_error"):
            return 0, 1
        if result.get("extraction_snapshot_id"):
            return 1, 0
        return 0, 0
    finally:
        close_old_connections()


def _enqueue_seo_after_aeo(run_id: int) -> None:
    """
    MVP sequencing: once AEO reaches terminal state, warm SEO in a separate task.
    """
    trigger_seo_warmup_after_aeo_task.delay(run_id)


def _chain_post_phase3_extraction(run) -> None:
    """
    After Phase 3 extractions for Phase 1 snapshots:

    - Starter-scale profiles: go straight to Phase 4 (Phase 2 is already queued from Phase 1).
    - Pro/Advanced: run Phase 2 multi-pass (OpenAI/Gemini confidence) first; Phase 2 ends with Phase 4.

    Expansion/backfill runs use the same path: Phase 1 → Phase 3 → Phase 2 (with ``prompt_set`` from
    this run's aggregates via ``phase2_prompt_plan_items_for_execution_run``) → Phase 4 → Phase 5.

    Ordering avoids scoring/recommendations on aggregates that still need a second provider pass.
    """
    from .aeo.aeo_utils import phase2_prompt_plan_items_for_execution_run

    refresh_competitor_snapshot_for_profile_task.delay(run.profile_id)
    refresh_aeo_dashboard_bundle_cache_task.delay(run.profile_id)
    if _is_onboarding_sample_size_profile(run.profile):
        run_aeo_phase4_scoring_task.delay(run.id)
    else:
        payload = phase2_prompt_plan_items_for_execution_run(run)
        run_aeo_phase2_confidence_task.delay(run.id, payload if payload else None)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def enrich_snapshot_keywords_task(self, snapshot_id: int) -> None:
    """
    Load snapshot by id; enrich top_keywords with gap + LLM keywords; then atomically
    recompute and persist all dependent SEO snapshot metrics.
    """
    from .models import SEOOverviewSnapshot

    try:
        snapshot = (
            SEOOverviewSnapshot.objects.filter(pk=snapshot_id)
            .select_related("business_profile", "user")
            .first()
        )
    except Exception as e:
        logger.warning("[SEO async] enrich_snapshot_keywords_task snapshot load failed snapshot_id=%s: %s", snapshot_id, e)
        return

    if not snapshot:
        logger.warning("[SEO async] enrich_snapshot_keywords_task snapshot not found snapshot_id=%s", snapshot_id)
        return

    if getattr(snapshot, "keywords_enriched_at", None):
        # Use Django timezone-aware now; snapshot timestamps are already aware.
        if django_tz.now() - snapshot.keywords_enriched_at <= KEYWORDS_ENRICHMENT_TTL:
            return

    domain = (snapshot.cached_domain or "").strip()
    if not domain:
        logger.warning("[SEO async] enrich_snapshot_keywords_task no cached_domain snapshot_id=%s", snapshot_id)
        return

    location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    language_code = getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en")
    user = snapshot.user
    # Must match the snapshot row: multi-company accounts share one user; using is_main would
    # enrich non-main snapshots with the wrong profile (competitors, usage caps, Labs context).
    profile_for_usage = getattr(snapshot, "business_profile", None)
    if profile_for_usage is None:
        profile_for_usage = (
            user.business_profiles.filter(is_main=True).first()
            or user.business_profiles.first()
        )

    # Copy so we don't mutate in-place until we're ready to save
    top_keywords: List[Dict[str, Any]] = [dict(k) for k in (getattr(snapshot, "top_keywords", None) or [])]
    for _row in top_keywords:
        if not (_row or {}).get("keyword_origin"):
            _row["keyword_origin"] = "ranked"

    # Debug evidence: do we already have rank values before gap/LLM enrichment?
    try:
        from .dataforseo_utils import _dbg_ba84ae_log  # avoid module-level import

        ranked_before_positive = sum(
            1 for k in top_keywords if isinstance(k.get("rank"), int) and (k.get("rank") or 0) > 0
        )
        ranked_before_none = sum(1 for k in top_keywords if k.get("rank") is None)
        _dbg_ba84ae_log(
            hypothesisId="H3_enrichment_before_snapshot_rank_counts",
            location="accounts/tasks.py:enrich_snapshot_keywords_task",
            message="before enrichment rank counts",
            data={
                "snapshot_id": snapshot_id,
                "top_keywords_count": len(top_keywords),
                "rank_positive_count": ranked_before_positive,
                "rank_none_count": ranked_before_none,
            },
            runId="pre-fix",
        )
    except Exception:
        pass

    try:
        from accounts.third_party_usage import usage_profile_context

        from .dataforseo_utils import (
            enrich_with_gap_keywords,
            enrich_with_llm_keywords,
            enrich_keyword_ranks_from_labs,
            recompute_snapshot_metrics_from_keywords,
            normalize_seo_snapshot_metrics,
            sort_top_keywords_for_display,
        )

        with usage_profile_context(profile_for_usage):
            enrich_with_gap_keywords(
                domain,
                location_code,
                language_code,
                user,
                top_keywords,
                business_profile=profile_for_usage,
            )
            enrich_with_llm_keywords(
                user,
                location_code,
                top_keywords,
                business_profile=profile_for_usage,
            )
            rank_stats = enrich_keyword_ranks_from_labs(
                domain=domain,
                location_code=location_code,
                language_code=language_code,
                top_keywords=top_keywords,
                user=user,
                business_profile=profile_for_usage,
            )
        total = int(rank_stats.get("total") or 0)
        ranked_after = int(rank_stats.get("non_null_after") or 0)
        coverage = (ranked_after / total) if total > 0 else 0.0
        logger.info(
            "[SEO async] rank enrichment coverage snapshot_id=%s total=%s ranked_after=%s coverage=%.2f%% filled_ranked=%s filled_gap=%s",
            snapshot_id,
            total,
            ranked_after,
            coverage * 100.0,
            int(rank_stats.get("filled_from_ranked") or 0),
            int(rank_stats.get("filled_from_gap") or 0),
        )
        min_coverage = float(getattr(settings, "SEO_RANK_ENRICHMENT_MIN_COVERAGE", 0.05))
        if total > 0 and coverage < min_coverage:
            logger.warning(
                "[SEO async] rank enrichment below threshold; skipping save snapshot_id=%s coverage=%.2f%% threshold=%.2f%%",
                snapshot_id,
                coverage * 100.0,
                min_coverage * 100.0,
            )
            return
    except Exception as e:
        logger.exception(
            "[SEO async] enrich_snapshot_keywords_task enrichment failed snapshot_id=%s: %s",
            snapshot_id,
            e,
        )
        return

    max_kw = int(getattr(settings, "SEO_TOP_KEYWORDS_MAX_PERSISTED", 200))
    top_keywords_sorted = sort_top_keywords_for_display(top_keywords, max_rows=max_kw)

    # Debug guardrails: quickly detect blank competitor/rank regressions in UI tables.
    total_keywords = len(top_keywords_sorted)
    keywords_with_rank = sum(
        1 for k in top_keywords_sorted if isinstance(k.get("rank"), int) and (k.get("rank") or 0) > 0
    )
    keywords_with_competitor = sum(
        1
        for k in top_keywords_sorted
        if (
            (k.get("top_competitor_domain") or k.get("top_competitor"))
            or (isinstance(k.get("competitors"), list) and len(k.get("competitors") or []) > 0)
        )
    )
    keywords_with_outranking_competitor = sum(
        1
        for k in top_keywords_sorted
        if (
            isinstance(k.get("rank"), int)
            and (k.get("rank") or 0) > 0
            and (
                (
                    isinstance(k.get("top_competitor_rank"), int)
                    and (k.get("top_competitor_rank") or 0) > 0
                    and int(k.get("top_competitor_rank")) < int(k.get("rank"))
                )
                or (
                    isinstance(k.get("competitors"), list)
                    and any(
                        isinstance(c.get("rank"), int)
                        and (c.get("rank") or 0) > 0
                        and int(c.get("rank")) < int(k.get("rank"))
                        for c in (k.get("competitors") or [])
                    )
                )
            )
        )
    )
    rank_pct = (keywords_with_rank / total_keywords * 100.0) if total_keywords > 0 else 0.0
    competitor_pct = (keywords_with_competitor / total_keywords * 100.0) if total_keywords > 0 else 0.0
    outranking_competitor_pct = (
        keywords_with_outranking_competitor / total_keywords * 100.0
        if total_keywords > 0
        else 0.0
    )
    logger.info(
        "[SEO async] keyword coverage snapshot_id=%s total=%s rank_non_null_pct=%.2f competitor_data_pct=%.2f outranking_competitor_pct=%.2f",
        snapshot_id,
        total_keywords,
        rank_pct,
        competitor_pct,
        outranking_competitor_pct,
    )

    profile_for_metrics = getattr(snapshot, "business_profile", None)
    if profile_for_metrics is None:
        profile_for_metrics = (
            snapshot.user.business_profiles.filter(is_main=True).first()
            or snapshot.user.business_profiles.first()
        )
    from .dataforseo_utils import resolve_snapshot_location_context as _resolve_snap_ctx

    snapshot_context = _resolve_snap_ctx(
        profile=profile_for_metrics,
        default_location_code=int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840)),
    )
    snapshot_mode = str(snapshot_context.get("mode") or "organic")

    try:
        metrics = normalize_seo_snapshot_metrics(
            recompute_snapshot_metrics_from_keywords(
                top_keywords=top_keywords_sorted,
                domain=domain,
                location_code=location_code,
                language_code=language_code,
                seo_location_mode=snapshot_mode,
                business_profile=profile_for_metrics,
            )
        )
        logger.info(
            "[SEO async] recompute snapshot_id=%s keywords_with_rank=%s estimated_traffic_before=%s estimated_traffic_after=%s appearances_before=%s appearances_after=%s total_search_volume_before=%s total_search_volume_after=%s visibility_before=%s visibility_after=%s missed_before=%s missed_after=%s",
            snapshot_id,
            keywords_with_rank,
            int(snapshot.organic_visitors or 0),
            int(metrics.get("estimated_traffic") or 0),
            int(snapshot.estimated_search_appearances_monthly or 0),
            int(metrics.get("estimated_search_appearances_monthly") or 0),
            int(snapshot.total_search_volume or 0),
            int(metrics.get("total_search_volume") or 0),
            int(snapshot.search_visibility_percent or 0),
            int(metrics.get("search_visibility_percent") or 0),
            int(snapshot.missed_searches_monthly or 0),
            int(metrics.get("missed_searches_monthly") or 0),
        )
        if (
            keywords_with_rank > 0
            and int(metrics.get("search_visibility_percent") or 0) == 0
            and int(metrics.get("total_search_volume") or 0) > 0
        ):
            logger.warning(
                "[SEO async] consistency_check snapshot_id=%s ranked_keywords=%s visibility_zero_with_volume=true",
                snapshot_id,
                keywords_with_rank,
            )
        with transaction.atomic():
            snapshot.top_keywords = top_keywords_sorted
            snapshot.keywords_enriched_at = django_tz.now()
            snapshot.organic_visitors = int(metrics["estimated_traffic"])
            snapshot.total_search_volume = int(metrics["total_search_volume"])
            snapshot.estimated_search_appearances_monthly = int(metrics["estimated_search_appearances_monthly"])
            snapshot.search_visibility_percent = int(metrics["search_visibility_percent"])
            snapshot.missed_searches_monthly = int(metrics["missed_searches_monthly"])
            snapshot.search_performance_score = int(metrics["search_performance_score"])
            snapshot.keywords_ranking = int(metrics["keywords_ranking"])
            snapshot.top3_positions = int(metrics["top3_positions"])
            snapshot.cached_location_mode = str(snapshot_context.get("mode") or "organic")
            snapshot.cached_location_code = int(snapshot_context.get("code") or 0)
            snapshot.cached_location_label = str(snapshot_context.get("label") or "")
            snapshot.local_verification_applied = any(
                str((row or {}).get("rank_source") or "baseline") == "local_verified"
                for row in top_keywords_sorted
            )
            snapshot.local_verified_keyword_count = sum(
                1 for row in top_keywords_sorted if (row or {}).get("local_verified_rank") is not None
            )
            snapshot.save(
                update_fields=[
                    "top_keywords",
                    "keywords_enriched_at",
                    "organic_visitors",
                    "total_search_volume",
                    "estimated_search_appearances_monthly",
                    "search_visibility_percent",
                    "missed_searches_monthly",
                    "search_performance_score",
                    "keywords_ranking",
                    "top3_positions",
                    "cached_location_mode",
                    "cached_location_code",
                    "cached_location_label",
                    "local_verification_applied",
                    "local_verified_keyword_count",
                ]
            )

            # Enrichment completes after the initial snapshot save; re-queue next steps / keyword
            # suggestions so they run on enriched ``top_keywords`` (TTL bypass sees newer timestamps).
            sid = int(snapshot.id)

            def _enqueue_post_enrichment_seo_tasks() -> None:
                generate_snapshot_next_steps_task.delay(sid)
                generate_keyword_action_suggestions_task.delay(sid)

            transaction.on_commit(_enqueue_post_enrichment_seo_tasks)

        # Debug evidence after enrichment: did we keep rank values or end up null?
        try:
            from .dataforseo_utils import _dbg_ba84ae_log  # avoid module-level import

            ranked_after_positive = sum(
                1 for k in top_keywords_sorted if isinstance(k.get("rank"), int) and (k.get("rank") or 0) > 0
            )
            ranked_after_none = sum(1 for k in top_keywords_sorted if k.get("rank") is None)
            _dbg_ba84ae_log(
                hypothesisId="H3_enrichment_after_snapshot_rank_counts",
                location="accounts/tasks.py:enrich_snapshot_keywords_task",
                message="after enrichment rank counts",
                data={
                    "snapshot_id": snapshot_id,
                    "top_keywords_count": len(top_keywords_sorted),
                    "rank_positive_count": ranked_after_positive,
                    "rank_none_count": ranked_after_none,
                },
                runId="pre-fix",
            )
        except Exception:
            pass
    except Exception as e:
        logger.exception(
            "[SEO async] enrich_snapshot_keywords_task save failed snapshot_id=%s: %s",
            snapshot_id,
            e,
        )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def generate_snapshot_next_steps_task(self, snapshot_id: int) -> None:
    """
    Load snapshot by id; generate seo_next_steps via LLM; update only
    seo_next_steps and seo_next_steps_refreshed_at. Does not change core score or visibility metrics.
    """
    from .models import SEOOverviewSnapshot

    try:
        snapshot = SEOOverviewSnapshot.objects.filter(pk=snapshot_id).first()
    except Exception as e:
        logger.warning("[SEO async] generate_snapshot_next_steps_task snapshot load failed snapshot_id=%s: %s", snapshot_id, e)
        return

    if not snapshot:
        logger.warning("[SEO async] generate_snapshot_next_steps_task snapshot not found snapshot_id=%s", snapshot_id)
        return

    # TTL: skip duplicate work within NEXT_STEPS_TTL unless snapshot keywords/metrics were refreshed
    # more recently than the last next-steps generation (enrichment pipeline finishes after first pass).
    if getattr(snapshot, "seo_next_steps_refreshed_at", None):
        within_ttl = django_tz.now() - snapshot.seo_next_steps_refreshed_at <= NEXT_STEPS_TTL
        if within_ttl and not _seo_snapshot_corpus_newer_than_next_steps(snapshot):
            return

    kws = list(getattr(snapshot, "top_keywords", None) or [])
    if not kws:
        logger.info(
            "[SEO async] generate_snapshot_next_steps_task skip empty top_keywords snapshot_id=%s",
            snapshot_id,
        )
        return

    try:
        from .openai_utils import generate_seo_next_steps
    except ImportError as e:
        logger.warning("[SEO async] generate_snapshot_next_steps_task openai_utils import failed: %s", e)
        return

    seo_data = seo_data_dict_from_seo_overview_snapshot(snapshot)

    try:
        steps = generate_seo_next_steps(seo_data, snapshot=snapshot)
    except Exception as e:
        logger.exception(
            "[SEO async] generate_snapshot_next_steps_task LLM failed snapshot_id=%s: %s",
            snapshot_id,
            e,
        )
        return

    steps_list = list(steps)[:6] if steps else []

    try:
        snapshot.seo_next_steps = steps_list
        snapshot.seo_next_steps_refreshed_at = django_tz.now()
        snapshot.save(
            update_fields=[
                "seo_next_steps",
                "seo_next_steps_refreshed_at",
            ]
        )
    except Exception as e:
        logger.exception(
            "[SEO async] generate_snapshot_next_steps_task save failed snapshot_id=%s: %s",
            snapshot_id,
            e,
        )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def generate_keyword_action_suggestions_task(self, snapshot_id: int) -> None:
    """
    Load snapshot by id; generate per-keyword SEO action suggestions via OpenAI.
    Updates only keyword_action_suggestions and keyword_action_suggestions_refreshed_at.
    """
    from .models import SEOOverviewSnapshot

    try:
        snapshot = SEOOverviewSnapshot.objects.filter(pk=snapshot_id).first()
    except Exception as e:
        logger.warning(
            "[SEO async] generate_keyword_action_suggestions_task snapshot load failed snapshot_id=%s: %s",
            snapshot_id,
            e,
        )
        return

    if not snapshot:
        logger.warning(
            "[SEO async] generate_keyword_action_suggestions_task snapshot not found snapshot_id=%s",
            snapshot_id,
        )
        return

    if getattr(snapshot, "keyword_action_suggestions_refreshed_at", None):
        within_ttl = django_tz.now() - snapshot.keyword_action_suggestions_refreshed_at <= KEYWORD_ACTION_TTL
        if within_ttl and not _seo_snapshot_corpus_newer_than_keyword_suggestions(snapshot):
            return

    keywords: list[Dict[str, Any]] = getattr(snapshot, "top_keywords", None) or []
    if not keywords:
        return
    keywords = keywords[:KEYWORD_ACTION_SUGGESTIONS_MAX_ROWS]

    try:
        from .openai_utils import generate_keyword_action_suggestions
    except ImportError as e:
        logger.warning(
            "[SEO async] generate_keyword_action_suggestions_task openai_utils import failed: %s",
            e,
        )
        return

    snapshot_seo_data = seo_data_dict_from_seo_overview_snapshot(snapshot)

    try:
        suggestions = generate_keyword_action_suggestions(
            keywords,
            snapshot=snapshot,
            snapshot_seo_data=snapshot_seo_data,
        )
    except Exception as e:
        logger.exception(
            "[SEO async] generate_keyword_action_suggestions_task LLM failed snapshot_id=%s: %s",
            snapshot_id,
            e,
        )
        return

    try:
        snapshot.keyword_action_suggestions = suggestions or []
        snapshot.keyword_action_suggestions_refreshed_at = django_tz.now()
        snapshot.save(
            update_fields=["keyword_action_suggestions", "keyword_action_suggestions_refreshed_at"]
        )
    except Exception as e:
        logger.exception(
            "[SEO async] generate_keyword_action_suggestions_task save failed snapshot_id=%s: %s",
            snapshot_id,
            e,
        )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def run_aeo_phase1_execution_task(
    self,
    run_id: int,
    prompt_set: list[dict] | None = None,
    providers: list[str] | None = None,
    force_refresh: bool = False,
) -> None:
    """
    Execute Phase 1 AEO prompt batch asynchronously with 30-day cache policy.

    ``providers`` — optional ``['openai']``, ``['gemini']``, ``['perplexity']``, or combinations, for dashboard refresh.
    ``force_refresh`` — when True, skip the 30-day execution cache (explicit user refresh).
    """
    from .models import AEOExecutionRun, AEOResponseSnapshot
    from .aeo.aeo_execution_utils import run_aeo_prompt_batch
    from .aeo.aeo_extraction_utils import run_single_extraction
    from .aeo.progressive_onboarding import (
        PASSES_PER_PROVIDER_TARGET,
        build_phase1_provider_batches,
        classify_prompt_category,
        update_prompt_aggregate_from_extraction,
    )

    run = (
        AEOExecutionRun.objects.select_related("profile")
        .filter(pk=run_id)
        .first()
    )
    if not run:
        logger.warning("[AEO phase1] run not found run_id=%s", run_id)
        return

    if run.status in {AEOExecutionRun.STATUS_COMPLETED, AEOExecutionRun.STATUS_FAILED, AEOExecutionRun.STATUS_SKIPPED_CACHED}:
        return

    profile = run.profile
    provider_set = {str(p).strip().lower() for p in (providers or []) if str(p).strip()}
    extraction_platform: str | None = None
    if len(provider_set) == 1:
        extraction_platform = next(iter(provider_set))

    duplicate_inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exclude(pk=run.pk).exists()
    if duplicate_inflight:
        run.status = AEOExecutionRun.STATUS_SKIPPED_CACHED
        run.cache_hit = True
        run.fetch_mode = AEOExecutionRun.FETCH_MODE_CACHE_HIT
        run.extraction_status = AEOExecutionRun.STAGE_SKIPPED
        run.error_message = "skipped_duplicate_inflight"
        run.finished_at = django_tz.now()
        run.save(
            update_fields=[
                "status",
                "cache_hit",
                "fetch_mode",
                "extraction_status",
                "error_message",
                "finished_at",
                "updated_at",
            ]
        )
        logger.info(
            "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=duplicate_inflight",
            run.id,
            profile.id,
            run.status,
        )
        _enqueue_seo_after_aeo(run.id)
        run_aeo_phase4_scoring_task.delay(run.id)
        return

    run.status = AEOExecutionRun.STATUS_RUNNING
    run.started_at = django_tz.now()
    run.save(update_fields=["status", "started_at", "updated_at"])

    latest_success = None
    if not force_refresh:
        latest_success = (
            AEOExecutionRun.objects.filter(
                profile=profile,
                status=AEOExecutionRun.STATUS_COMPLETED,
                fetch_mode=AEOExecutionRun.FETCH_MODE_FRESH_FETCH,
                finished_at__isnull=False,
            )
            .exclude(pk=run.pk)
            .order_by("-finished_at")
            .first()
        )
    if (
        not force_refresh
        and latest_success
        and latest_success.finished_at
        and (django_tz.now() - latest_success.finished_at) <= timedelta(days=30)
    ):
        run.status = AEOExecutionRun.STATUS_SKIPPED_CACHED
        run.cache_hit = True
        run.fetch_mode = AEOExecutionRun.FETCH_MODE_CACHE_HIT
        run.extraction_status = AEOExecutionRun.STAGE_SKIPPED
        run.prompt_count_executed = 0
        run.prompt_count_failed = 0
        run.finished_at = django_tz.now()
        run.save(
            update_fields=[
                "status",
                "cache_hit",
                "fetch_mode",
                "extraction_status",
                "prompt_count_executed",
                "prompt_count_failed",
                "finished_at",
                "updated_at",
            ]
        )
        logger.info("[AEO phase1] cache hit, skipping execution run_id=%s profile_id=%s", run.id, profile.id)
        logger.info(
            "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=skipped_cached",
            run.id,
            profile.id,
            run.status,
        )
        _enqueue_seo_after_aeo(run.id)
        run_aeo_phase4_scoring_task.delay(run.id)
        return

    selected = list(prompt_set or [])
    target = aeo_effective_monitored_target_for_profile(profile)
    selected = selected[:target]
    run.prompt_count_requested = len(selected)
    run.cache_hit = False
    run.fetch_mode = AEOExecutionRun.FETCH_MODE_FRESH_FETCH
    run.save(update_fields=["prompt_count_requested", "cache_hit", "fetch_mode", "updated_at"])

    from accounts.third_party_usage import usage_profile_context

    extraction_created = 0
    extraction_failed = 0

    def _process_result(one: dict[str, Any], meta: dict[str, Any]) -> None:
        nonlocal extraction_created, extraction_failed
        if not one.get("success"):
            return
        snapshot_id = one.get("snapshot_id")
        if snapshot_id is None:
            extraction_failed += 1
            logger.error(
                "[AEO phase1] success result missing snapshot_id run_id=%s profile_id=%s prompt_hash=%s",
                run.id,
                profile.id,
                one.get("prompt_hash"),
            )
            return
        snap = AEOResponseSnapshot.objects.select_related("profile").filter(id=snapshot_id).first()
        if snap is None:
            extraction_failed += 1
            logger.error(
                "[AEO phase1] snapshot missing for callback run_id=%s profile_id=%s snapshot_id=%s",
                run.id,
                profile.id,
                snapshot_id,
            )
            return
        if snap.profile_id != profile.id:
            extraction_failed += 1
            logger.error(
                "[AEO phase1] snapshot profile mismatch run_id=%s profile_id=%s snapshot_profile_id=%s snapshot_id=%s",
                run.id,
                profile.id,
                snap.profile_id,
                snapshot_id,
            )
            return
        ex = run_single_extraction(snap, save=True)
        ex_id = ex.get("extraction_snapshot_id")
        if not ex_id:
            extraction_failed += 1
            logger.error(
                "[AEO phase1] extraction missing id run_id=%s profile_id=%s snapshot_id=%s",
                run.id,
                profile.id,
                snapshot_id,
            )
            return
        extraction_created += 1
        extraction_row = snap.extraction_snapshots.filter(id=ex_id).first()
        if extraction_row is None:
            extraction_failed += 1
            logger.error(
                "[AEO phase1] extraction row missing run_id=%s profile_id=%s snapshot_id=%s extraction_id=%s",
                run.id,
                profile.id,
                snapshot_id,
                ex_id,
            )
            return
        prompt_obj = meta.get("prompt_obj") if isinstance(meta, dict) else {}
        prompt_category = str(
            (prompt_obj or {}).get("_aeo_category") or classify_prompt_category(prompt_obj or {})
        )
        update_prompt_aggregate_from_extraction(
            profile=profile,
            execution_run_id=run.id,
            response_snapshot=snap,
            extraction_snapshot=extraction_row,
            prompt_category=prompt_category,
        )

    try:
        with usage_profile_context(profile):
            if _is_onboarding_sample_size_profile(profile) and providers is None:
                p1_batches = build_phase1_provider_batches(selected)
                openai_selected = p1_batches.get("openai", [])
                gemini_selected = p1_batches.get("gemini", [])
                batch_openai = run_aeo_prompt_batch(
                    openai_selected,
                    profile,
                    save=True,
                    execution_run=run,
                    providers=["openai"],
                    on_result=_process_result,
                )
                batch_gemini = run_aeo_prompt_batch(
                    gemini_selected,
                    profile,
                    save=True,
                    execution_run=run,
                    providers=["gemini"],
                    on_result=_process_result,
                )
                batch = {
                    "executed": int(batch_openai.get("executed") or 0) + int(batch_gemini.get("executed") or 0),
                    "failed": int(batch_openai.get("failed") or 0) + int(batch_gemini.get("failed") or 0),
                    "results": list(batch_openai.get("results") or []) + list(batch_gemini.get("results") or []),
                }
            else:
                batch = run_aeo_prompt_batch(
                    selected,
                    profile,
                    save=True,
                    execution_run=run,
                    providers=providers,
                    on_result=_process_result,
                )
        response_snapshot_ids = [
            int(one.get("snapshot_id"))
            for one in (batch.get("results") or [])
            if one.get("success") and one.get("snapshot_id") is not None
        ]
        run.prompt_count_executed = int(batch.get("executed") or 0)
        run.prompt_count_failed = int(batch.get("failed") or 0)
        successful_results = sum(1 for one in (batch.get("results") or []) if one.get("success"))
        response_snapshot_count = len(response_snapshot_ids)
        run.extraction_count = extraction_created
        artifacts_ok = (
            successful_results > 0
            and successful_results == response_snapshot_count
            and extraction_created == response_snapshot_count
        )
        run.extraction_status = (
            AEOExecutionRun.STAGE_COMPLETED if artifacts_ok else AEOExecutionRun.STAGE_FAILED
        )
        run.phase1_completed_at = django_tz.now()
        run.phase1_provider_calls = int(len(batch.get("results") or []))
        run.background_status = (
            AEOExecutionRun.STAGE_RUNNING if _is_onboarding_sample_size_profile(profile) else run.background_status
        )
        run.status = AEOExecutionRun.STATUS_COMPLETED if artifacts_ok else AEOExecutionRun.STATUS_FAILED
        run.finished_at = django_tz.now()
        run.error_message = "" if artifacts_ok else "phase1_artifact_mismatch_missing_response_or_extraction"
        run.save(
            update_fields=[
                "prompt_count_executed",
                "prompt_count_failed",
                "status",
                "finished_at",
                "error_message",
                "extraction_count",
                "extraction_status",
                "phase1_completed_at",
                "phase1_provider_calls",
                "background_status",
                "updated_at",
            ]
        )
        logger.info(
            "[AEO phase1] execution complete run_id=%s profile_id=%s executed=%s failed=%s",
            run.id,
            profile.id,
            run.prompt_count_executed,
            run.prompt_count_failed,
        )
        logger.info(
            "[AEO phase1] provider_scope run_id=%s profile_id=%s provider=%s created_ids=%s count=%s",
            run.id,
            profile.id,
            extraction_platform or "any",
            response_snapshot_ids,
            len(response_snapshot_ids),
        )
        if run.status != AEOExecutionRun.STATUS_COMPLETED:
            logger.error(
                "[AEO phase1] artifact mismatch run_id=%s profile_id=%s success=%s snapshots=%s extractions=%s",
                run.id,
                profile.id,
                successful_results,
                response_snapshot_count,
                extraction_created,
            )
            _enqueue_seo_after_aeo(run.id)
            return
        if _is_onboarding_sample_size_profile(profile) and providers is None:
            # Phase 2 enqueues Phase 4 once when multi-pass work finishes; do not schedule Phase 4 here
            # (duplicate Phase 4 races on score_snapshot_id / scoring_status and can double-fire Phase 5).
            run_aeo_phase2_confidence_task.delay(run.id, selected)
        else:
            run_aeo_phase3_extraction_task.delay(run.id, response_snapshot_ids, extraction_platform)
    except Exception as e:
        run.status = AEOExecutionRun.STATUS_FAILED
        run.error_message = f"{type(e).__name__}: {e}"
        run.finished_at = django_tz.now()
        run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        logger.exception("[AEO phase1] run failed run_id=%s", run_id)
        logger.info(
            "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=phase1_failed",
            run.id,
            profile.id,
            run.status,
        )
        _enqueue_seo_after_aeo(run.id)


def _aggregate_priority_bucket(agg: Any) -> int:
    # legacy helper retained for compatibility
    reasons = set(agg.stability_reasons or [])
    if agg.stability_status == "unstable":
        return 0
    if "brand_mention_changed_across_provider" in reasons:
        return 1
    if "wrong_url_attribution_present" in reasons:
        return 2
    if "missing_second_provider_or_pass" in reasons:
        return 3
    return 4


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def run_aeo_phase2_confidence_task(self, run_id: int, prompt_set: list[dict] | None = None) -> None:
    from .aeo.aeo_execution_utils import run_aeo_prompt_batch
    from .aeo.aeo_extraction_utils import run_single_extraction
    from .aeo.progressive_onboarding import (
        PASSES_PER_PROVIDER_TARGET,
        classify_prompt_category,
        update_prompt_aggregate_from_extraction,
    )
    from .models import AEOExecutionRun, AEOPromptExecutionAggregate, AEOResponseSnapshot

    run = AEOExecutionRun.objects.select_related("profile").filter(pk=run_id).first()
    if not run:
        return
    profile = run.profile
    saved_prompts = list(prompt_set or [])
    if not saved_prompts:
        from .aeo.prompt_storage import plan_items_dicts_fallback_from_profile

        saved_prompts = plan_items_dicts_fallback_from_profile(profile)

    saved_by_hash = {}
    for p in saved_prompts:
        cat = classify_prompt_category(p)
        spec = dict(p)
        spec["_aeo_category"] = cat
        h = __import__("accounts.aeo.aeo_execution_utils", fromlist=["hash_prompt"]).hash_prompt(
            str(spec.get("prompt") or "")
        )
        saved_by_hash[h] = spec

    extraction_created = 0
    extraction_failed = 0

    def _process_result(one: dict[str, Any], meta: dict[str, Any]) -> None:
        nonlocal extraction_created, extraction_failed
        if not one.get("success"):
            return
        snapshot_id = one.get("snapshot_id")
        if snapshot_id is None:
            extraction_failed += 1
            logger.error(
                "[AEO phase2] success result missing snapshot_id run_id=%s profile_id=%s prompt_hash=%s",
                run.id,
                profile.id,
                one.get("prompt_hash"),
            )
            return
        snap = AEOResponseSnapshot.objects.select_related("profile").filter(id=snapshot_id).first()
        if snap is None:
            extraction_failed += 1
            logger.error(
                "[AEO phase2] snapshot missing for callback run_id=%s profile_id=%s snapshot_id=%s",
                run.id,
                profile.id,
                snapshot_id,
            )
            return
        if snap.profile_id != profile.id:
            extraction_failed += 1
            logger.error(
                "[AEO phase2] snapshot profile mismatch run_id=%s profile_id=%s snapshot_profile_id=%s snapshot_id=%s",
                run.id,
                profile.id,
                snap.profile_id,
                snapshot_id,
            )
            return
        ex = run_single_extraction(snap, save=True)
        ex_id = ex.get("extraction_snapshot_id")
        if not ex_id:
            extraction_failed += 1
            logger.error(
                "[AEO phase2] extraction missing id run_id=%s profile_id=%s snapshot_id=%s",
                run.id,
                profile.id,
                snapshot_id,
            )
            return
        extraction_created += 1
        extraction_row = snap.extraction_snapshots.filter(id=ex_id).first()
        if extraction_row is None:
            extraction_failed += 1
            logger.error(
                "[AEO phase2] extraction row missing run_id=%s profile_id=%s snapshot_id=%s extraction_id=%s",
                run.id,
                profile.id,
                snapshot_id,
                ex_id,
            )
            return
        prompt_obj = meta.get("prompt_obj") if isinstance(meta, dict) else {}
        update_prompt_aggregate_from_extraction(
            profile=profile,
            execution_run_id=run.id,
            response_snapshot=snap,
            extraction_snapshot=extraction_row,
            prompt_category=str((prompt_obj or {}).get("_aeo_category") or classify_prompt_category(prompt_obj or {})),
        )

    from accounts.aeo.perplexity_execution_utils import perplexity_execution_enabled
    from accounts.third_party_usage import usage_profile_context

    p2_enabled = perplexity_execution_enabled()
    # Iterate until each provider/prompt pair reaches done criteria:
    # - pass_count >=2 and stable, or
    # - pass_count ==3 after unstable-at-2 third pass.
    for _round in range(4):
        aggs = list(
            AEOPromptExecutionAggregate.objects.filter(profile=profile, execution_run=run).order_by("id")
        )
        if not aggs:
            if _round == 0:
                run.background_status = AEOExecutionRun.STAGE_SKIPPED
                run.save(update_fields=["background_status", "updated_at"])
                run_aeo_phase4_scoring_task.delay(run.id)
                return
            break
        aggs.sort(key=lambda a: (_aggregate_priority_bucket(a), str(a.prompt_hash)))

        provider_batches: dict[str, list[dict[str, Any]]] = {"openai": [], "gemini": [], "perplexity": []}
        provider_priority: dict[str, dict[str, int]] = {"openai": {}, "gemini": {}, "perplexity": {}}

        def _push(provider: str, spec: dict[str, Any], prio: int) -> None:
            h = __import__("accounts.aeo.aeo_execution_utils", fromlist=["hash_prompt"]).hash_prompt(
                str(spec.get("prompt") or "")
            )
            cur = provider_priority[provider].get(h)
            if cur is None or prio < cur:
                provider_priority[provider][h] = prio
                if all(
                    __import__("accounts.aeo.aeo_execution_utils", fromlist=["hash_prompt"]).hash_prompt(
                        str(s.get("prompt") or "")
                    )
                    != h
                    for s in provider_batches[provider]
                ):
                    provider_batches[provider].append(spec)

        for agg in aggs:
            spec = saved_by_hash.get(agg.prompt_hash)
            if not spec:
                continue
            o_count = int(agg.openai_pass_count or 0)
            g_count = int(agg.gemini_pass_count or 0)
            p_count = int(agg.perplexity_pass_count or 0)
            # Single-provider refresh runs should not spawn other providers here.
            gemini_only_aggregate = o_count == 0 and g_count > 0
            openai_only_aggregate = g_count == 0 and o_count > 0
            perplexity_only_aggregate = p_count > 0 and o_count == 0 and g_count == 0
            if o_count < PASSES_PER_PROVIDER_TARGET:
                if not gemini_only_aggregate and not perplexity_only_aggregate:
                    _push("openai", spec, 0)
            elif (
                bool(agg.openai_third_pass_required)
                and not bool(agg.openai_third_pass_ran)
                and o_count < 3
            ):
                if not gemini_only_aggregate and not perplexity_only_aggregate:
                    _push("openai", spec, 1)
            if g_count < PASSES_PER_PROVIDER_TARGET:
                if not openai_only_aggregate and not perplexity_only_aggregate:
                    _push("gemini", spec, 0)
            elif (
                bool(agg.gemini_third_pass_required)
                and not bool(agg.gemini_third_pass_ran)
                and g_count < 3
            ):
                if not openai_only_aggregate and not perplexity_only_aggregate:
                    _push("gemini", spec, 1)
            if p2_enabled:
                if p_count < PASSES_PER_PROVIDER_TARGET:
                    if not openai_only_aggregate and not gemini_only_aggregate:
                        _push("perplexity", spec, 0)
                elif (
                    bool(agg.perplexity_third_pass_required)
                    and not bool(agg.perplexity_third_pass_ran)
                    and p_count < 3
                ):
                    if not openai_only_aggregate and not gemini_only_aggregate:
                        _push("perplexity", spec, 1)

        for provider in ("openai", "gemini", "perplexity"):
            provider_batches[provider].sort(
                key=lambda s: (
                    provider_priority[provider].get(
                        __import__("accounts.aeo.aeo_execution_utils", fromlist=["hash_prompt"]).hash_prompt(
                            str(s.get("prompt") or "")
                        ),
                        9,
                    ),
                    __import__("accounts.aeo.aeo_execution_utils", fromlist=["hash_prompt"]).hash_prompt(
                        str(s.get("prompt") or "")
                    ),
                )
            )
        if (
            not provider_batches["openai"]
            and not provider_batches["gemini"]
            and not provider_batches["perplexity"]
        ):
            break

        with usage_profile_context(profile):
            if provider_batches["openai"]:
                run_aeo_prompt_batch(
                    provider_batches["openai"],
                    profile,
                    save=True,
                    execution_run=run,
                    providers=["openai"],
                    on_result=_process_result,
                )
            if provider_batches["gemini"]:
                run_aeo_prompt_batch(
                    provider_batches["gemini"],
                    profile,
                    save=True,
                    execution_run=run,
                    providers=["gemini"],
                    on_result=_process_result,
                )
            if p2_enabled and provider_batches["perplexity"]:
                run_aeo_prompt_batch(
                    provider_batches["perplexity"],
                    profile,
                    save=True,
                    execution_run=run,
                    providers=["perplexity"],
                    on_result=_process_result,
                )

    n_mon = len(saved_prompts)
    bounds = aeo_http_call_bounds_for_monitoring(n_mon)
    third_o = AEOPromptExecutionAggregate.objects.filter(
        profile=profile, execution_run=run, openai_third_pass_ran=True
    ).count()
    third_g = AEOPromptExecutionAggregate.objects.filter(
        profile=profile, execution_run=run, gemini_third_pass_ran=True
    ).count()
    third_p = AEOPromptExecutionAggregate.objects.filter(
        profile=profile, execution_run=run, perplexity_third_pass_ran=True
    ).count()
    logger.info(
        "[AEO phase2] finished run_id=%s profile_id=%s monitored_prompts=%s http_call_bounds=%s "
        "third_pass_openai=%s third_pass_gemini=%s third_pass_perplexity=%s extractions_created=%s",
        run.id,
        profile.id,
        n_mon,
        bounds,
        third_o,
        third_g,
        third_p,
        extraction_created,
    )

    run.extraction_count = int(run.extraction_count or 0) + extraction_created
    run.background_status = (
        AEOExecutionRun.STAGE_COMPLETED if extraction_failed == 0 else AEOExecutionRun.STAGE_FAILED
    )
    if extraction_failed:
        run.error_message = "phase2_artifact_mismatch_missing_response_or_extraction"
    update_fields = ["extraction_count", "background_status", "updated_at"]
    if extraction_failed:
        update_fields.append("error_message")
    run.save(update_fields=update_fields)
    run_aeo_phase4_scoring_task.delay(run.id)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def run_aeo_gemini_refresh_task(
    self,
    run_id: int,
    prompt_set: list[dict] | None = None,
) -> None:
    """
    Explicit Gemini-only refresh path (dashboard button).

    Prompt source defaults to BusinessProfile.selected_aeo_prompts on the run profile.
    This task always forces refresh and never executes OpenAI provider calls.
    """
    from .aeo.aeo_utils import plan_items_from_saved_prompt_strings
    from .models import AEOExecutionRun

    run = AEOExecutionRun.objects.select_related("profile").filter(pk=run_id).first()
    if not run:
        logger.warning("[AEO gemini refresh] run not found run_id=%s", run_id)
        return

    profile = run.profile
    selected = list(prompt_set or [])
    if not selected:
        saved = profile.selected_aeo_prompts or []
        selected = plan_items_from_saved_prompt_strings(saved)

    if not selected:
        run.status = AEOExecutionRun.STATUS_FAILED
        run.error_message = "gemini_refresh_no_selected_prompts"
        run.finished_at = django_tz.now()
        run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        logger.warning(
            "[AEO gemini refresh] provider=gemini run_id=%s profile_id=%s status=failed reason=no_prompts",
            run.id,
            profile.id,
        )
        return

    logger.info(
        "[AEO gemini refresh] provider=gemini run_id=%s profile_id=%s prompts=%s status=queued_phase1",
        run.id,
        profile.id,
        len(selected),
    )
    run_aeo_phase1_execution_task(
        run.id,
        prompt_set=selected,
        providers=["gemini"],
        force_refresh=True,
    )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def run_aeo_perplexity_refresh_task(
    self,
    run_id: int,
    prompt_set: list[dict] | None = None,
) -> None:
    """
    Explicit Perplexity-only refresh path (dashboard button / API).
    """
    from .aeo.aeo_utils import plan_items_from_saved_prompt_strings
    from .models import AEOExecutionRun

    run = AEOExecutionRun.objects.select_related("profile").filter(pk=run_id).first()
    if not run:
        logger.warning("[AEO perplexity refresh] run not found run_id=%s", run_id)
        return

    profile = run.profile
    selected = list(prompt_set or [])
    if not selected:
        saved = profile.selected_aeo_prompts or []
        selected = plan_items_from_saved_prompt_strings(saved)

    if not selected:
        run.status = AEOExecutionRun.STATUS_FAILED
        run.error_message = "perplexity_refresh_no_selected_prompts"
        run.finished_at = django_tz.now()
        run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        logger.warning(
            "[AEO perplexity refresh] provider=perplexity run_id=%s profile_id=%s status=failed reason=no_prompts",
            run.id,
            profile.id,
        )
        return

    logger.info(
        "[AEO perplexity refresh] provider=perplexity run_id=%s profile_id=%s prompts=%s status=queued_phase1",
        run.id,
        profile.id,
        len(selected),
    )
    run_aeo_phase1_execution_task(
        run.id,
        prompt_set=selected,
        providers=["perplexity"],
        force_refresh=True,
    )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def run_aeo_phase3_extraction_task(
    self,
    run_id: int,
    response_snapshot_ids: list[int] | None = None,
    response_platform: str | None = None,
) -> None:
    from .aeo.worker_limits import aeo_execution_max_workers
    from .models import AEOExecutionRun, AEOResponseSnapshot

    run = AEOExecutionRun.objects.select_related("profile").filter(pk=run_id).first()
    if not run:
        logger.warning("[AEO phase3] run not found run_id=%s", run_id)
        return
    if run.extraction_status == AEOExecutionRun.STAGE_COMPLETED:
        _chain_post_phase3_extraction(run)
        return

    run.extraction_status = AEOExecutionRun.STAGE_RUNNING
    run.save(update_fields=["extraction_status", "updated_at"])
    logger.info("[AEO phase3] extraction start run_id=%s profile_id=%s", run.id, run.profile_id)

    qs = AEOResponseSnapshot.objects.filter(profile=run.profile)
    # Critical scoping: if caller provided IDs (even empty), never broaden to profile-wide query.
    if response_snapshot_ids is not None:
        qs = qs.filter(id__in=response_snapshot_ids)
    if response_platform:
        qs = qs.filter(platform=response_platform)
    responses = list(qs.order_by("id"))
    logger.info(
        "[AEO phase3] scoped extraction run_id=%s profile_id=%s provider=%s response_ids=%s selected=%s parser_provider=openai",
        run.id,
        run.profile_id,
        response_platform or "any",
        response_snapshot_ids if response_snapshot_ids is not None else "all_profile",
        len(responses),
    )
    created_count = 0
    failed_count = 0
    try:
        to_process: list[int] = []
        for response in responses:
            if run.started_at and response.extraction_snapshots.filter(
                created_at__gte=run.started_at
            ).exists():
                continue
            to_process.append(response.id)

        max_workers = aeo_execution_max_workers()
        started_at = run.started_at
        if to_process:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(_aeo_phase3_extract_one_snapshot, rid, started_at)
                    for rid in to_process
                ]
                for fut in as_completed(futures):
                    c_inc, f_inc = fut.result()
                    created_count += c_inc
                    failed_count += f_inc

        run.extraction_count = created_count
        run.extraction_status = AEOExecutionRun.STAGE_COMPLETED
        run.save(update_fields=["extraction_count", "extraction_status", "updated_at"])
        logger.info(
            "[AEO phase3] extraction complete run_id=%s profile_id=%s created=%s failed=%s",
            run.id,
            run.profile_id,
            created_count,
            failed_count,
        )
        _chain_post_phase3_extraction(run)
    except Exception as exc:
        run.extraction_status = AEOExecutionRun.STAGE_FAILED
        run.error_message = f"{run.error_message}\nphase3:{type(exc).__name__}: {exc}".strip()
        run.save(update_fields=["extraction_status", "error_message", "updated_at"])
        logger.exception("[AEO phase3] extraction failed run_id=%s", run.id)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def refresh_competitor_snapshot_for_profile_task(self, profile_id: int) -> None:
    from .aeo.competitor_snapshots import compute_and_save_competitor_snapshot
    from .models import BusinessProfile

    profile = BusinessProfile.objects.filter(id=profile_id).first()
    if not profile:
        return
    try:
        compute_and_save_competitor_snapshot(profile, platform_scope="all")
    except Exception:
        logger.exception(
            "[AEO competitors] snapshot refresh failed profile_id=%s",
            profile_id,
        )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def refresh_aeo_dashboard_bundle_cache_task(self, profile_id: int) -> None:
    from .models import AEODashboardBundleCache, BusinessProfile
    from .views import _build_aeo_prompt_coverage_payload

    profile = BusinessProfile.objects.filter(id=profile_id).first()
    if not profile:
        return
    try:
        payload = _build_aeo_prompt_coverage_payload(profile, ready_only=False)
        AEODashboardBundleCache.objects.update_or_create(
            profile=profile,
            defaults={"payload_json": payload},
        )
    except Exception:
        logger.exception(
            "[AEO dashboard cache] refresh failed profile_id=%s",
            profile_id,
        )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def run_aeo_phase4_scoring_task(self, run_id: int) -> None:
    from .models import AEOExecutionRun
    from .aeo.aeo_scoring_utils import calculate_aeo_scores_for_business, calculate_layered_scores_from_aggregates
    from .models import AEOScoreSnapshot

    run: AEOExecutionRun | None = None
    profile = None
    idempotent_phase5_rid: int | None = None
    idempotent_seo_rid: int | None = None

    with transaction.atomic():
        locked = (
            AEOExecutionRun.objects.select_for_update()
            .select_related("profile")
            .filter(pk=run_id)
            .first()
        )
        if not locked:
            logger.warning("[AEO phase4] run not found run_id=%s", run_id)
            return
        if locked.scoring_status == AEOExecutionRun.STAGE_COMPLETED and locked.score_snapshot_id:
            if _aeo_recommendation_stage_enabled():
                idempotent_phase5_rid = locked.id
            else:
                locked.recommendation_status = AEOExecutionRun.STAGE_SKIPPED
                locked.save(update_fields=["recommendation_status", "updated_at"])
                logger.info(
                    "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=phase4_completed_no_phase5",
                    locked.id,
                    locked.profile_id,
                    locked.status,
                )
                idempotent_seo_rid = locked.id
        elif locked.scoring_status == AEOExecutionRun.STAGE_RUNNING:
            logger.info("[AEO phase4] skip concurrent duplicate run_id=%s profile_id=%s", run_id, locked.profile_id)
            return
        else:
            locked.scoring_status = AEOExecutionRun.STAGE_RUNNING
            locked.save(update_fields=["scoring_status", "updated_at"])
            profile = locked.profile
            run = locked

    if idempotent_phase5_rid is not None:
        run_aeo_phase5_recommendation_task.delay(idempotent_phase5_rid)
        return
    if idempotent_seo_rid is not None:
        _enqueue_seo_after_aeo(idempotent_seo_rid)
        return

    assert run is not None and profile is not None

    logger.info("[AEO phase4] scoring start run_id=%s profile_id=%s", run.id, run.profile_id)
    try:
        # Starter profiles: layered scoring (sample → confidence). CONFIDENCE uses strict aggregate
        # filters and may match zero rows while SAMPLE still produced a row — we must not drop the
        # sample snapshot_id on the execution run (downstream Phase 5 / staff enqueue require it).
        if _is_onboarding_sample_size_profile(profile):
            sample_score_data = calculate_layered_scores_from_aggregates(
                profile,
                execution_run_id=run.id,
                score_layer=AEOScoreSnapshot.LAYER_SAMPLE,
                save=True,
            )
            score_data = calculate_layered_scores_from_aggregates(
                profile,
                execution_run_id=run.id,
                score_layer=AEOScoreSnapshot.LAYER_CONFIDENCE,
                save=True,
            )
            snapshot_id = score_data.get("snapshot_id") or sample_score_data.get("snapshot_id")
            total_prompts = int(score_data.get("total_prompts") or sample_score_data.get("total_prompts") or 0)
            if snapshot_id is None:
                score_data = calculate_aeo_scores_for_business(
                    profile, save=True, execution_run_id=run.id
                )
                snapshot_id = score_data.get("snapshot_id")
                total_prompts = int(score_data.get("total_prompts") or 0)
        else:
            score_data = calculate_aeo_scores_for_business(
                profile, save=True, execution_run_id=run.id
            )
            snapshot_id = score_data.get("snapshot_id")
            total_prompts = int(score_data.get("total_prompts") or 0)

        run.score_snapshot_id = snapshot_id
        run.scoring_status = AEOExecutionRun.STAGE_COMPLETED
        run.save(update_fields=["score_snapshot_id", "scoring_status", "updated_at"])
        logger.info(
            "[AEO phase4] scoring complete run_id=%s profile_id=%s score_snapshot_id=%s total_prompts=%s",
            run.id,
            run.profile_id,
            run.score_snapshot_id,
            total_prompts,
        )
        if _aeo_recommendation_stage_enabled():
            run_aeo_phase5_recommendation_task.delay(run.id)
        else:
            run.recommendation_status = AEOExecutionRun.STAGE_SKIPPED
            run.save(update_fields=["recommendation_status", "updated_at"])
            logger.info(
                "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=phase4_completed_no_phase5",
                run.id,
                run.profile_id,
                run.status,
            )
            _enqueue_seo_after_aeo(run.id)
    except Exception as exc:
        run.scoring_status = AEOExecutionRun.STAGE_FAILED
        run.error_message = f"{run.error_message}\nphase4:{type(exc).__name__}: {exc}".strip()
        run.save(update_fields=["scoring_status", "error_message", "updated_at"])
        logger.exception("[AEO phase4] scoring failed run_id=%s", run.id)
        logger.info(
            "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=phase4_failed",
            run.id,
            run.profile_id,
            run.status,
        )
        _enqueue_seo_after_aeo(run.id)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def run_aeo_phase5_recommendation_task(self, run_id: int) -> None:
    """
    Each invocation creates a new ``AEORecommendationRun`` when it completes.

    Follow-up runs after expansion/backfill must not short-circuit on a prior
    ``recommendation_status=completed``; ``GET /api/aeo/prompt-coverage/`` uses the latest run by
    ``created_at``/``id``.
    """
    from .models import AEOExecutionRun
    from .aeo.aeo_recommendation_utils import generate_aeo_recommendations

    run = AEOExecutionRun.objects.select_related("profile").filter(pk=run_id).first()
    if not run:
        logger.warning("[AEO phase5] run not found run_id=%s", run_id)
        return
    if not _aeo_recommendation_stage_enabled():
        run.recommendation_status = AEOExecutionRun.STAGE_SKIPPED
        run.save(update_fields=["recommendation_status", "updated_at"])
        logger.info(
            "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=phase5_disabled",
            run.id,
            run.profile_id,
            run.status,
        )
        _enqueue_seo_after_aeo(run.id)
        return

    run.recommendation_status = AEOExecutionRun.STAGE_RUNNING
    run.save(update_fields=["recommendation_status", "updated_at"])
    logger.info("[AEO phase5] recommendation start run_id=%s profile_id=%s", run.id, run.profile_id)
    try:
        data = generate_aeo_recommendations(
            run.profile,
            save=True,
            execution_run_id=run.id,
        )
        run.recommendation_run_id = data.get("recommendation_run_id")
        run.recommendation_status = AEOExecutionRun.STAGE_COMPLETED
        run.save(update_fields=["recommendation_run_id", "recommendation_status", "updated_at"])
        logger.info(
            "[AEO phase5] recommendation complete run_id=%s profile_id=%s recommendation_run_id=%s count=%s",
            run.id,
            run.profile_id,
            run.recommendation_run_id,
            len(data.get("recommendations") or []),
        )
        logger.info(
            "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=phase5_completed",
            run.id,
            run.profile_id,
            run.status,
        )
        refresh_aeo_dashboard_bundle_cache_task.delay(run.profile_id)
        _enqueue_seo_after_aeo(run.id)
    except Exception as exc:
        run.recommendation_status = AEOExecutionRun.STAGE_FAILED
        run.error_message = f"{run.error_message}\nphase5:{type(exc).__name__}: {exc}".strip()
        run.save(update_fields=["recommendation_status", "error_message", "updated_at"])
        logger.exception("[AEO phase5] recommendation failed run_id=%s", run.id)
        logger.info(
            "[AEO orchestrator] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true reason=phase5_failed",
            run.id,
            run.profile_id,
            run.status,
        )
        _enqueue_seo_after_aeo(run.id)


def try_enqueue_aeo_full_monitored_pipeline(profile_id: int, *, source: str = "") -> dict[str, Any]:
    """
    Enqueue Phase 1 for **all** prompts on ``BusinessProfile.selected_aeo_prompts`` (via
    ``plan_items_from_saved_prompt_strings``), with ``force_refresh=True`` — same path as production
    refresh (Phase 1 → Phase 3 / Phase 2 chain → Phase 4 → Phase 5).

    Used by staff UI and optional Celery Beat (``aeo_scheduled_full_monitoring_tick_task``).

    Returns keys: ``ok``, ``queued``, ``run_id``, ``reason``, ``message``, ``prompt_count``.
    Does not start a new run while another is ``pending`` or ``running`` for the profile.
    """
    from django.db import transaction

    from .aeo.aeo_utils import plan_items_from_saved_prompt_strings
    from .aeo.prompt_storage import monitored_prompt_keys_in_order
    from .models import AEOExecutionRun, BusinessProfile

    pid = int(profile_id)
    profile = BusinessProfile.objects.filter(pk=pid).first()
    if profile is None:
        return {
            "ok": False,
            "queued": False,
            "run_id": None,
            "reason": "profile_not_found",
            "message": "Business profile not found.",
            "prompt_count": 0,
        }

    saved = list(profile.selected_aeo_prompts or [])
    monitored_n = len(monitored_prompt_keys_in_order(saved))
    payload = plan_items_from_saved_prompt_strings(saved)
    if not payload:
        return {
            "ok": False,
            "queued": False,
            "run_id": None,
            "reason": "no_prompts",
            "message": "No monitored prompts on this profile.",
            "prompt_count": monitored_n,
        }

    inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exists()
    if inflight:
        return {
            "ok": True,
            "queued": False,
            "run_id": None,
            "reason": "duplicate_inflight",
            "message": "A pipeline run is already in progress for this profile.",
            "prompt_count": monitored_n,
        }

    tag = (source or "").strip()[:400]
    em = "full_rerun"
    if tag:
        em = f"{em} source={tag}"[:2000]

    with transaction.atomic():
        run = AEOExecutionRun.objects.create(
            profile_id=pid,
            prompt_count_requested=len(payload),
            status=AEOExecutionRun.STATUS_PENDING,
            error_message=em,
        )
        rid = int(run.id)
        ps = list(payload)
        transaction.on_commit(
            lambda: run_aeo_phase1_execution_task.delay(rid, prompt_set=ps, force_refresh=True)
        )

    return {
        "ok": True,
        "queued": True,
        "run_id": rid,
        "reason": "enqueued",
        "message": f"Enqueued full monitoring pipeline for {len(payload)} prompt(s).",
        "prompt_count": monitored_n,
    }


def try_enqueue_aeo_phase5_recommendations_only(
    profile_id: int, *, source: str = ""
) -> dict[str, Any]:
    """
    Enqueue ``run_aeo_phase5_recommendation_task`` for the latest **completed** execution run on
    the profile that already has Phase 4 scoring (``scoring_status=completed`` + ``score_snapshot_id``).

    Does not start Phase 1 / extraction / scoring. Refused while another execution run is
    ``pending`` or ``running`` for the profile (same guard as full monitored pipeline).

    Return keys align with ``try_enqueue_aeo_full_monitored_pipeline``: ``ok``, ``queued``,
    ``run_id`` (target execution run id when applicable), ``reason``, ``message``, ``prompt_count``.
    """
    from django.db import transaction

    from .aeo.prompt_storage import monitored_prompt_keys_in_order
    from .models import AEOExecutionRun, BusinessProfile

    pid = int(profile_id)
    profile = BusinessProfile.objects.filter(pk=pid).first()
    if profile is None:
        return {
            "ok": False,
            "queued": False,
            "run_id": None,
            "reason": "profile_not_found",
            "message": "Business profile not found.",
            "prompt_count": 0,
        }

    saved = list(profile.selected_aeo_prompts or [])
    monitored_n = len(monitored_prompt_keys_in_order(saved))
    if monitored_n == 0:
        return {
            "ok": False,
            "queued": False,
            "run_id": None,
            "reason": "no_prompts",
            "message": "No monitored prompts on this profile.",
            "prompt_count": 0,
        }

    if not _aeo_recommendation_stage_enabled():
        return {
            "ok": False,
            "queued": False,
            "run_id": None,
            "reason": "recommendations_disabled",
            "message": "AEO recommendation stage is disabled in settings (AEO_ENABLE_RECOMMENDATION_STAGE).",
            "prompt_count": monitored_n,
        }

    inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exists()
    if inflight:
        return {
            "ok": True,
            "queued": False,
            "run_id": None,
            "reason": "duplicate_inflight",
            "message": (
                "A pipeline run is already in progress for this profile; wait for it to finish "
                "before re-running recommendations."
            ),
            "prompt_count": monitored_n,
        }

    eligible = (
        AEOExecutionRun.objects.filter(
            profile_id=pid,
            scoring_status=AEOExecutionRun.STAGE_COMPLETED,
            score_snapshot_id__isnull=False,
            status__in=[
                AEOExecutionRun.STATUS_COMPLETED,
                AEOExecutionRun.STATUS_SKIPPED_CACHED,
            ],
        )
        .order_by("-id")
        .first()
    )
    if eligible is None:
        return {
            "ok": False,
            "queued": False,
            "run_id": None,
            "reason": "no_eligible_run",
            "message": (
                "No completed execution run with finished scoring found for this profile. "
                "Run the full pipeline first."
            ),
            "prompt_count": monitored_n,
        }

    if eligible.recommendation_status == AEOExecutionRun.STAGE_RUNNING:
        return {
            "ok": True,
            "queued": False,
            "run_id": int(eligible.id),
            "reason": "phase5_inflight",
            "message": "Phase 5 recommendations are already running for the latest scored execution run.",
            "prompt_count": monitored_n,
        }

    tag = (source or "").strip()[:400]
    rid = int(eligible.id)
    with transaction.atomic():
        locked = (
            AEOExecutionRun.objects.select_for_update()
            .filter(pk=rid, profile_id=pid)
            .first()
        )
        if locked is None:
            return {
                "ok": False,
                "queued": False,
                "run_id": None,
                "reason": "no_eligible_run",
                "message": "Execution run no longer available.",
                "prompt_count": monitored_n,
            }
        if locked.recommendation_status == AEOExecutionRun.STAGE_RUNNING:
            return {
                "ok": True,
                "queued": False,
                "run_id": int(locked.id),
                "reason": "phase5_inflight",
                "message": "Phase 5 recommendations are already running for this execution run.",
                "prompt_count": monitored_n,
            }

        transaction.on_commit(lambda r=rid: run_aeo_phase5_recommendation_task.delay(r))

    msg = f"Enqueued Phase 5 recommendations for execution run {rid}."
    if tag:
        msg = f"{msg} (source={tag})"
    return {
        "ok": True,
        "queued": True,
        "run_id": rid,
        "reason": "enqueued",
        "message": msg[:2000],
        "prompt_count": monitored_n,
    }


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def aeo_scheduled_full_monitoring_tick_task(self) -> None:
    """
    Celery Beat hook for scheduled full AEO re-runs.

    Controlled by Django settings (env-backed):
    - ``AEO_SCHEDULED_FULL_MONITORING_ENABLED`` — master switch (default False).
    - ``AEO_SCHEDULED_FULL_MONITORING_PROFILE_IDS`` — list of BusinessProfile pk to enqueue each tick;
      when empty, this task no-ops even if enabled (safe default: staff manual only until IDs are set).
    - ``AEO_SCHEDULED_FULL_MONITORING_CRON_HOUR`` / ``_MINUTE`` — schedule registered in ``CELERY_BEAT_SCHEDULE``.
    """
    from django.conf import settings

    if not bool(getattr(settings, "AEO_SCHEDULED_FULL_MONITORING_ENABLED", False)):
        return
    raw_ids = getattr(settings, "AEO_SCHEDULED_FULL_MONITORING_PROFILE_IDS", None) or []
    if not raw_ids:
        logger.info("[AEO cron] tick skipped (AEO_SCHEDULED_FULL_MONITORING_PROFILE_IDS empty)")
        return
    for raw in raw_ids:
        try:
            pid = int(raw)
        except (TypeError, ValueError):
            logger.warning("[AEO cron] skip non-int profile id %r", raw)
            continue
        out = try_enqueue_aeo_full_monitored_pipeline(pid, source="scheduled_cron")
        logger.info(
            "[AEO cron] profile_id=%s reason=%s queued=%s run_id=%s",
            pid,
            out.get("reason"),
            out.get("queued"),
            out.get("run_id"),
        )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def trigger_seo_warmup_after_aeo_task(self, run_id: int) -> None:
    """
    Sequential MVP handoff: AEO terminal -> SEO warmup.
    Uses existing SEO helper for TTL/freshness behavior and avoids duplicate trigger replays.
    """
    from .models import AEOExecutionRun
    from .dataforseo_utils import get_or_refresh_seo_score_for_user

    run = AEOExecutionRun.objects.select_related("profile", "profile__user").filter(pk=run_id).first()
    if not run:
        logger.warning("[AEO->SEO] run not found aeo_run_id=%s", run_id)
        return

    terminal = run.status in {
        AEOExecutionRun.STATUS_COMPLETED,
        AEOExecutionRun.STATUS_FAILED,
        AEOExecutionRun.STATUS_SKIPPED_CACHED,
    }
    if not terminal:
        logger.info(
            "[AEO->SEO] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=false seo_trigger_outcome=not_terminal",
            run.id,
            run.profile_id,
            run.status,
        )
        return

    if run.seo_triggered_at and (django_tz.now() - run.seo_triggered_at) <= timedelta(minutes=10):
        logger.info(
            "[AEO->SEO] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=false seo_trigger_outcome=duplicate_recent",
            run.id,
            run.profile_id,
            run.status,
        )
        return

    website = str(getattr(run.profile, "website_url", "") or "").strip()
    if not website:
        run.seo_triggered_at = django_tz.now()
        run.seo_trigger_status = "skipped_no_website"
        run.save(update_fields=["seo_triggered_at", "seo_trigger_status", "updated_at"])
        logger.info(
            "[AEO->SEO] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true seo_trigger_outcome=skipped_no_website",
            run.id,
            run.profile_id,
            run.status,
        )
        return

    try:
        logger.info(
            "[AEO->SEO] AEO terminal -> SEO start aeo_run_id=%s profile_id=%s aeo_status=%s website=%s",
            run.id,
            run.profile_id,
            run.status,
            website,
        )
        get_or_refresh_seo_score_for_user(
            run.profile.user,
            site_url=website,
            force_refresh=False,
            business_profile=run.profile,
        )
        run.seo_triggered_at = django_tz.now()
        run.seo_trigger_status = "success"
        run.save(update_fields=["seo_triggered_at", "seo_trigger_status", "updated_at"])
        logger.info(
            "[AEO->SEO] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true seo_trigger_outcome=success",
            run.id,
            run.profile_id,
            run.status,
        )
    except Exception as exc:
        run.seo_triggered_at = django_tz.now()
        run.seo_trigger_status = "failed"
        run.error_message = f"{run.error_message}\nseo_trigger:{type(exc).__name__}: {exc}".strip()
        run.save(update_fields=["seo_triggered_at", "seo_trigger_status", "error_message", "updated_at"])
        logger.exception(
            "[AEO->SEO] aeo_run_id=%s profile_id=%s aeo_status=%s seo_trigger_attempted=true seo_trigger_outcome=failed",
            run.id,
            run.profile_id,
            run.status,
        )


def _clean_profile_prompts_for_expansion(profile) -> list[str]:
    from .aeo.prompt_storage import monitored_prompt_keys_in_order

    return monitored_prompt_keys_in_order(profile.selected_aeo_prompts)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def aeo_backfill_monitored_prompt_execution_task(
    self,
    profile_id: int,
    before_prompt_count: int | None = None,
    after_prompt_count: int | None = None,
    *,
    repair: bool = False,
) -> None:
    """
    Run Phase 1 for monitored prompts missing full OpenAI + Gemini + Perplexity extractions.

    Enqueued after post-payment expansion when new prompt strings were merged. Uses
    ``force_refresh=True`` on Phase 1 so the 30-day "skip cached" path does not block
    execution for prompts that were not in the original onboarding run.

    Pro/Advanced: Phase 1 → Phase 3 extraction → Phase 4 scoring → Phase 5 recommendations
    (when ``AEO_ENABLE_RECOMMENDATION_STAGE``), same as the full pipeline. Starter-scale
    profiles are skipped (expansion does not run for them).
    """
    from .aeo.aeo_utils import plan_items_from_saved_prompt_strings
    from .aeo.prompt_scan_progress import monitored_prompt_keys_missing_full_coverage
    from .models import AEOExecutionRun, BusinessProfile

    pid = int(profile_id)
    profile = BusinessProfile.objects.filter(pk=pid).select_related("user").first()
    if profile is None:
        logger.warning("[AEO backfill] profile not found profile_id=%s", pid)
        return
    if _is_onboarding_sample_size_profile(profile):
        logger.info("[AEO backfill] skip starter-scale profile profile_id=%s", pid)
        return
    if not repair and not aeo_should_run_post_payment_expansion(profile):
        logger.info("[AEO backfill] skip plan not expansion-eligible profile_id=%s", pid)
        return

    missing = monitored_prompt_keys_missing_full_coverage(profile)
    if not missing:
        logger.info(
            "[AEO backfill] nothing_missing profile_id=%s before_count=%s after_count=%s",
            pid,
            before_prompt_count,
            after_prompt_count,
        )
        return

    inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exists()
    if inflight:
        logger.info(
            "[AEO backfill] skip_duplicate_inflight profile_id=%s missing_count=%s before_count=%s after_count=%s",
            pid,
            len(missing),
            before_prompt_count,
            after_prompt_count,
        )
        return

    target_cap = aeo_effective_monitored_target_for_profile(profile)
    payload = plan_items_from_saved_prompt_strings(missing, max_items=max(len(missing), target_cap))
    if not payload:
        return

    run = AEOExecutionRun.objects.create(
        profile=profile,
        prompt_count_requested=len(payload),
        status=AEOExecutionRun.STATUS_PENDING,
        error_message=f"expansion_backfill_profile_id={pid}",
    )
    try:
        run_aeo_phase1_execution_task.delay(run.id, prompt_set=payload, force_refresh=True)
        logger.info(
            "[AEO backfill] phase1_enqueued profile_id=%s run_id=%s before_count=%s after_count=%s "
            "backfill_prompts=%s missing_total=%s",
            pid,
            run.id,
            before_prompt_count,
            after_prompt_count,
            len(payload),
            len(missing),
        )
    except Exception as exc:
        run.status = AEOExecutionRun.STATUS_FAILED
        run.error_message = f"backfill_enqueue_failed:{type(exc).__name__}: {exc}"[:2000]
        run.finished_at = django_tz.now()
        run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        logger.exception("[AEO backfill] phase1 enqueue failed profile_id=%s run_id=%s", pid, run.id)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def aeo_repair_stalled_visibility_pipeline_task(self, profile_id: int) -> None:
    """
    When the visibility queue is idle but monitored prompts lack triple coverage or Phase-2
    completion, enqueue the same work as expansion backfill / Phase 2 (no duplicate Phase 1
    while a run is already active).
    """
    from datetime import datetime, timezone as py_tz

    from .aeo.aeo_utils import phase2_prompt_plan_items_for_execution_run
    from .aeo.prompt_full_ready import (
        get_latest_aggregate_for_prompt,
        phase2_passes_complete,
        recommendations_pipeline_settled_for_visibility,
        triple_extraction_complete_for_key,
    )
    from .aeo.prompt_scan_progress import monitored_prompt_keys_missing_full_coverage
    from .aeo.visibility_pending import aeo_visibility_pending_breakdown
    from .models import AEOExecutionRun, AEOResponseSnapshot, BusinessProfile

    pid = int(profile_id)
    profile = BusinessProfile.objects.filter(pk=pid).first()
    if profile is None:
        return
    if _is_onboarding_sample_size_profile(profile):
        return
    if not aeo_should_run_post_payment_expansion(profile):
        return

    bd = aeo_visibility_pending_breakdown(profile)
    if bd["visibility_pending"]:
        logger.info(
            "[AEO repair] skip work_in_progress profile_id=%s breakdown=%s",
            pid,
            bd,
        )
        return

    inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exists()
    if inflight:
        logger.info("[AEO repair] skip execution_inflight profile_id=%s", pid)
        return

    missing = monitored_prompt_keys_missing_full_coverage(profile)
    if missing:
        aeo_backfill_monitored_prompt_execution_task.delay(pid, repair=True)
        logger.info(
            "[AEO repair] enqueued phase1_backfill profile_id=%s missing=%s",
            pid,
            len(missing),
        )
        return

    if not recommendations_pipeline_settled_for_visibility(profile):
        logger.info("[AEO repair] skip recommendations_not_settled profile_id=%s", pid)
        return

    from .aeo.prompt_storage import monitored_prompt_keys_in_order

    monitored = monitored_prompt_keys_in_order(profile.selected_aeo_prompts)
    if not monitored:
        return

    responses = list(
        AEOResponseSnapshot.objects.filter(profile=profile).order_by("-created_at", "-id")
    )
    by_prompt: dict[str, list] = {}
    for resp in responses:
        key = (resp.prompt_text or "").strip()
        if not key:
            continue
        by_prompt.setdefault(key, []).append(resp)

    def _response_sort_key(x) -> tuple:
        c = x.created_at
        if c is None:
            return (datetime.min.replace(tzinfo=py_tz.utc), x.id)
        if django_tz.is_naive(c):
            c = django_tz.make_aware(c, py_tz.utc)
        return (c, x.id)

    def latest_snapshot_per_platform(rows: list) -> dict:
        best: dict = {}
        for r in sorted(rows, key=_response_sort_key, reverse=True):
            plat = str(r.platform or "").strip().lower()
            if plat not in {"openai", "gemini", "perplexity"}:
                continue
            if plat not in best:
                best[plat] = r
        return best

    run_ids: set[int] = set()
    for key in monitored:
        if not triple_extraction_complete_for_key(key, by_prompt, latest_snapshot_per_platform):
            continue
        agg = get_latest_aggregate_for_prompt(profile.id, key)
        if agg is None or phase2_passes_complete(agg):
            continue
        rid = getattr(agg, "execution_run_id", None)
        if rid:
            run_ids.add(int(rid))

    if not run_ids:
        return

    run_id = max(run_ids)
    run = AEOExecutionRun.objects.filter(pk=run_id, profile=profile).first()
    if run is None:
        return
    if run.status != AEOExecutionRun.STATUS_COMPLETED:
        return
    if run.extraction_status != AEOExecutionRun.STAGE_COMPLETED:
        return
    if run.scoring_status == AEOExecutionRun.STAGE_COMPLETED:
        logger.info(
            "[AEO repair] skip phase2 scoring_already_completed run_id=%s profile_id=%s",
            run.id,
            pid,
        )
        return

    payload = phase2_prompt_plan_items_for_execution_run(run)
    if not payload:
        logger.info("[AEO repair] skip phase2 empty_payload run_id=%s profile_id=%s", run.id, pid)
        return

    run_aeo_phase2_confidence_task.delay(run.id, payload)
    logger.info(
        "[AEO repair] enqueued phase2 run_id=%s profile_id=%s prompts=%s",
        run.id,
        pid,
        len(payload),
    )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=2, retry_backoff=30, ignore_result=True)
def schedule_aeo_prompt_plan_expansion(
    self,
    profile_id: int,
    expected_plan_slug: str | None = None,
    expansion_cap: int | None = None,
) -> None:
    """
    After Pro/Advanced billing, grow selected_aeo_prompts toward the plan cap (idempotent).

    **Enqueue only from** ``apply_subscription_payload_to_profile`` (Stripe webhooks), on commit, with
    ``expected_plan_slug`` and ``expansion_cap`` from the same price/link resolution used to set
    ``profile.plan`` for that event. At task start, if ``expected_plan_slug`` is set and the DB
    profile's plan differs (race with another update), the task logs and exits without mutating
    prompts. When ``expansion_cap`` is set, it is used as the target count (production caps from
    the webhook); otherwise the cap is derived from the current profile plan. Starter / testing
    mode still short-circuits via ``aeo_should_run_post_payment_expansion``.
    """
    from .aeo.aeo_utils import aeo_business_input_from_onboarding_payload, build_full_aeo_prompt_plan
    from .dataforseo_utils import normalize_domain
    from .models import BusinessProfile, OnboardingOnPageCrawl

    pid = int(profile_id)
    profile = BusinessProfile.objects.filter(pk=pid).select_related("user").first()
    if profile is None:
        return

    db_plan = str(getattr(profile, "plan", "") or "")
    logger.info(
        "[AEO expansion] start profile_id=%s expected_plan_slug=%s expansion_cap=%s profile.plan=%s",
        pid,
        expected_plan_slug,
        expansion_cap,
        db_plan,
    )
    if expected_plan_slug is not None:
        exp_norm = str(expected_plan_slug).strip().lower()
        if db_plan.strip().lower() != exp_norm:
            logger.warning(
                "[AEO expansion] plan_mismatch skip profile_id=%s expected_plan_slug=%s db_plan=%s",
                pid,
                expected_plan_slug,
                db_plan,
            )
            return

    now = django_tz.now()

    def touch(**kwargs: Any) -> None:
        kwargs.setdefault("updated_at", django_tz.now())
        BusinessProfile.objects.filter(pk=pid).update(**kwargs)

    if not aeo_should_run_post_payment_expansion(profile):
        cap_meta = aeo_effective_monitored_target_for_profile(profile)
        touch(
            aeo_prompt_expansion_status=BusinessProfile.AEO_PROMPT_EXPANSION_COMPLETE,
            aeo_prompt_expansion_target=cap_meta,
            aeo_prompt_expansion_progress=len(_clean_profile_prompts_for_expansion(profile)),
            aeo_prompt_expansion_last_error="",
            aeo_prompt_expansion_updated_at=now,
        )
        return

    cap = (
        int(expansion_cap)
        if expansion_cap is not None
        else aeo_effective_monitored_target_for_profile(profile)
    )
    if cap < 1:
        cap = aeo_effective_monitored_target_for_profile(profile)

    existing = _clean_profile_prompts_for_expansion(profile)
    if len(existing) >= cap:
        touch(
            aeo_prompt_expansion_status=BusinessProfile.AEO_PROMPT_EXPANSION_COMPLETE,
            aeo_prompt_expansion_target=cap,
            aeo_prompt_expansion_progress=len(existing),
            aeo_prompt_expansion_last_error="",
            aeo_prompt_expansion_updated_at=now,
        )
        return

    profile.refresh_from_db()
    if (
        profile.aeo_prompt_expansion_status == BusinessProfile.AEO_PROMPT_EXPANSION_RUNNING
        and profile.aeo_prompt_expansion_updated_at
        and (now - profile.aeo_prompt_expansion_updated_at).total_seconds() < 120
    ):
        logger.info("[AEO expansion] skip concurrent profile_id=%s", pid)
        return

    touch(
        aeo_prompt_expansion_status=BusinessProfile.AEO_PROMPT_EXPANSION_RUNNING,
        aeo_prompt_expansion_target=cap,
        aeo_prompt_expansion_progress=len(existing),
        aeo_prompt_expansion_last_error="",
        aeo_prompt_expansion_updated_at=now,
    )

    try:
        domain = normalize_domain(profile.website_url or "")
        crawl = None
        if domain:
            crawl = (
                OnboardingOnPageCrawl.objects.filter(
                    user=profile.user,
                    domain=domain,
                    status=OnboardingOnPageCrawl.STATUS_COMPLETED,
                )
                .order_by("-created_at")
                .first()
            )

        ctx: dict[str, Any] = {}
        if crawl and isinstance(crawl.context, dict):
            ctx = dict(crawl.context)

        review_rows: list[Any] = []
        if crawl and isinstance(crawl.review_topics, list):
            review_rows = list(crawl.review_topics)

        selected_topics = [
            str((row or {}).get("topic") or "").strip()
            for row in review_rows
            if str((row or {}).get("topic") or "").strip()
        ]
        details: list[dict[str, Any]] = []
        for row in review_rows:
            topic = str((row or {}).get("topic") or "").strip()
            if not topic:
                continue
            item: dict[str, Any] = {"keyword": topic}
            cat = (row or {}).get("category")
            if isinstance(cat, str) and cat.strip():
                item["aeo_category"] = cat.strip()
            rat = (row or {}).get("rationale")
            if isinstance(rat, str) and rat.strip():
                item["aeo_reason"] = rat.strip()
            details.append(item)

        website_for_input = str(profile.website_url or "").strip()
        if not website_for_input and crawl and getattr(crawl, "domain", None):
            d = str(crawl.domain or "").strip()
            if d:
                website_for_input = f"https://{d}" if "://" not in d else d

        business_input = aeo_business_input_from_onboarding_payload(
            business_name=str(ctx.get("business_name") or profile.business_name or ""),
            website_url=website_for_input,
            location=str(ctx.get("location") or profile.business_address or ""),
            language=str(ctx.get("language") or ""),
            selected_topics=selected_topics,
            customer_reach=str(
                ctx.get("customer_reach")
                or getattr(profile, "customer_reach", "")
                or "online"
            ),
            customer_reach_state=str(
                ctx.get("customer_reach_state")
                or getattr(profile, "customer_reach_state", "")
                or ""
            ),
            customer_reach_city=str(
                ctx.get("customer_reach_city")
                or getattr(profile, "customer_reach_city", "")
                or ""
            ),
        )

        logger.info(
            "[AEO expansion] openai_generate profile_id=%s cap=%s have=%s http_bounds=%s",
            pid,
            cap,
            len(existing),
            aeo_http_call_bounds_for_monitoring(cap),
        )

        plan = build_full_aeo_prompt_plan(
            profile,
            business_input=business_input,
            onboarding_topic_details=details if details else None,
            include_openai=True,
            target_combined_count=cap,
        )
        combined = list(plan.get("combined") or [])
        seen = {x.casefold() for x in existing}
        merged = list(existing)
        for item in combined:
            if len(merged) >= cap:
                break
            t = str(item.get("prompt") or "").strip()
            if not t:
                continue
            k = t.casefold()
            if k in seen:
                continue
            merged.append(t)
            seen.add(k)

        profile.refresh_from_db()
        profile.selected_aeo_prompts = merged[:cap]
        profile.save(update_fields=["selected_aeo_prompts", "updated_at"])
        final_n = len(_clean_profile_prompts_for_expansion(profile))
        meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
        err_tail = ""
        if final_n < cap:
            err_tail = (
                f"partial_expansion target={cap} got={final_n} openai_status={meta.get('openai_status')}"
            )[:2000]
        touch(
            aeo_prompt_expansion_status=BusinessProfile.AEO_PROMPT_EXPANSION_COMPLETE,
            aeo_prompt_expansion_target=cap,
            aeo_prompt_expansion_progress=final_n,
            aeo_prompt_expansion_last_error=err_tail,
            aeo_prompt_expansion_updated_at=django_tz.now(),
        )
        logger.info("[AEO expansion] complete profile_id=%s final=%s target=%s", pid, final_n, cap)

        before_key_set = {x.casefold() for x in existing}
        after_key_set = {x.casefold() for x in _clean_profile_prompts_for_expansion(profile)}
        added_keys = after_key_set - before_key_set
        if added_keys:
            before_n = len(existing)
            ac_after = final_n

            def _enqueue_backfill() -> None:
                try:
                    aeo_backfill_monitored_prompt_execution_task.delay(
                        pid,
                        before_prompt_count=before_n,
                        after_prompt_count=ac_after,
                    )
                except Exception:
                    logger.exception("[AEO expansion] backfill enqueue failed profile_id=%s", pid)

            transaction.on_commit(_enqueue_backfill)
            logger.info(
                "[AEO expansion] backfill_on_commit profile_id=%s new_prompt_keys=%s before_count=%s after_count=%s",
                pid,
                len(added_keys),
                before_n,
                ac_after,
            )
    except Exception as exc:
        logger.exception("[AEO expansion] failed profile_id=%s", pid)
        profile.refresh_from_db()
        touch(
            aeo_prompt_expansion_status=BusinessProfile.AEO_PROMPT_EXPANSION_ERROR,
            aeo_prompt_expansion_target=cap,
            aeo_prompt_expansion_progress=len(_clean_profile_prompts_for_expansion(profile)),
            aeo_prompt_expansion_last_error=f"{type(exc).__name__}: {exc}"[:2000],
            aeo_prompt_expansion_updated_at=django_tz.now(),
        )


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def onboarding_onpage_crawl_task(self, crawl_id: int) -> None:
    """DataForSEO On-Page crawl for onboarding (10 pages, Celery)."""
    from .onboarding_onpage import execute_onboarding_onpage_crawl

    execute_onboarding_onpage_crawl(int(crawl_id))


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def onboarding_prompt_generation_task(self, crawl_id: int) -> None:
    """
    Build onboarding prompt plan from stored crawl ``review_topics`` (all rows).

    Not enqueued automatically after on-page crawl; onboarding and add-company flows call
    ``/api/aeo/onboarding-prompt-plan/`` with user-selected topics instead. Kept for tests
    and any explicit invocation. Idempotent at crawl level (queued/running/completed states).
    """
    from .aeo.aeo_utils import (
        aeo_business_input_from_onboarding_payload,
        build_full_aeo_prompt_plan,
    )
    from datetime import timedelta

    from .models import AEOExecutionRun, OnboardingOnPageCrawl
    from .views import _serialize_aeo_prompt_items

    crawl = (
        OnboardingOnPageCrawl.objects.select_related("business_profile")
        .filter(pk=crawl_id)
        .first()
    )
    if not crawl:
        logger.warning("[onboarding prompt-plan] crawl id=%s not found", crawl_id)
        return
    if crawl.prompt_plan_status == OnboardingOnPageCrawl.PROMPT_PLAN_COMPLETED:
        return
    if crawl.status != OnboardingOnPageCrawl.STATUS_COMPLETED:
        return

    profile = crawl.business_profile
    ctx = crawl.context if isinstance(crawl.context, dict) else {}
    review_rows = crawl.review_topics if isinstance(crawl.review_topics, list) else []
    selected_topics = [
        str((row or {}).get("topic") or "").strip()
        for row in review_rows
        if str((row or {}).get("topic") or "").strip()
    ]
    if not selected_topics:
        crawl.prompt_plan_status = OnboardingOnPageCrawl.PROMPT_PLAN_FAILED
        crawl.prompt_plan_error = "no_review_topics"
        crawl.prompt_plan_finished_at = django_tz.now()
        crawl.save(
            update_fields=[
                "prompt_plan_status",
                "prompt_plan_error",
                "prompt_plan_finished_at",
                "updated_at",
            ]
        )
        return

    # Map LLM review rows onto legacy ``keyword`` + aeo_* keys for prompt-plan compatibility.
    details = []
    for row in review_rows:
        topic = str((row or {}).get("topic") or "").strip()
        if not topic:
            continue
        d: dict[str, object] = {"keyword": topic}
        cat = (row or {}).get("category")
        if isinstance(cat, str) and cat.strip():
            d["aeo_category"] = cat.strip()
        rat = (row or {}).get("rationale")
        if isinstance(rat, str) and rat.strip():
            d["aeo_reason"] = rat.strip()
        details.append(d)

    crawl.prompt_plan_status = OnboardingOnPageCrawl.PROMPT_PLAN_RUNNING
    crawl.prompt_plan_started_at = django_tz.now()
    crawl.prompt_plan_error = ""
    crawl.save(
        update_fields=[
            "prompt_plan_status",
            "prompt_plan_started_at",
            "prompt_plan_error",
            "updated_at",
        ]
    )
    try:
        business_input = aeo_business_input_from_onboarding_payload(
            business_name=str(ctx.get("business_name") or profile.business_name or ""),
            website_url=str(profile.website_url or crawl.domain or ""),
            location=str(ctx.get("location") or profile.business_address or ""),
            language=str(ctx.get("language") or ""),
            selected_topics=selected_topics,
            customer_reach=str(
                ctx.get("customer_reach")
                or getattr(profile, "customer_reach", "")
                or "online"
            ),
            customer_reach_state=str(
                ctx.get("customer_reach_state")
                or getattr(profile, "customer_reach_state", "")
                or ""
            ),
            customer_reach_city=str(
                ctx.get("customer_reach_city")
                or getattr(profile, "customer_reach_city", "")
                or ""
            ),
        )
        target_prompt_count = aeo_effective_monitored_target_for_profile(profile)
        plan = build_full_aeo_prompt_plan(
            profile,
            business_input=business_input,
            onboarding_topic_details=details,
            include_openai=True,
            target_combined_count=target_prompt_count,
        )
        combined = list(plan.get("combined") or [])
        meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
        openai_status = str(meta.get("openai_status") or "").strip().lower()
        if not combined or openai_status == "failed_empty":
            crawl.prompt_plan_status = OnboardingOnPageCrawl.PROMPT_PLAN_FAILED
            crawl.prompt_plan_prompt_count = 0
            crawl.prompt_plan_finished_at = django_tz.now()
            crawl.prompt_plan_error = (
                f"empty_prompt_plan_combined_count={len(combined)} openai_status={openai_status or 'unknown'}"
            )[:2000]
            crawl.save(
                update_fields=[
                    "prompt_plan_status",
                    "prompt_plan_prompt_count",
                    "prompt_plan_finished_at",
                    "prompt_plan_error",
                    "updated_at",
                ]
            )
            logger.error(
                "[onboarding prompt-plan] failed_empty crawl_id=%s profile_id=%s combined_count=%s openai_status=%s",
                crawl.id,
                profile.id,
                len(combined),
                openai_status or "unknown",
            )
            return

        prompt_texts = [str(x.get("prompt") or "").strip() for x in combined if str(x.get("prompt") or "").strip()]
        profile.selected_aeo_prompts = prompt_texts
        profile.save(update_fields=["selected_aeo_prompts", "updated_at"])
        from .aeo.prompt_storage import monitored_prompt_keys_in_order

        verify_saved = monitored_prompt_keys_in_order(profile.selected_aeo_prompts)
        if len(verify_saved) != len(prompt_texts):
            logger.error(
                "[onboarding prompt-plan] selected_aeo_prompts_save_verify_failed crawl_id=%s profile_id=%s expected=%s got=%s",
                crawl.id,
                profile.id,
                len(prompt_texts),
                len(verify_saved),
            )
            crawl.prompt_plan_status = OnboardingOnPageCrawl.PROMPT_PLAN_FAILED
            crawl.prompt_plan_finished_at = django_tz.now()
            crawl.prompt_plan_error = "selected_aeo_prompts_save_verify_failed"
            crawl.save(
                update_fields=[
                    "prompt_plan_status",
                    "prompt_plan_finished_at",
                    "prompt_plan_error",
                    "updated_at",
                ]
            )
            return

        crawl.prompt_plan_status = OnboardingOnPageCrawl.PROMPT_PLAN_COMPLETED
        crawl.prompt_plan_prompt_count = len(prompt_texts)
        crawl.prompt_plan_finished_at = django_tz.now()
        crawl.prompt_plan_error = ""
        crawl.save(
            update_fields=[
                "prompt_plan_status",
                "prompt_plan_prompt_count",
                "prompt_plan_finished_at",
                "prompt_plan_error",
                "updated_at",
            ]
        )
        logger.info(
            "[onboarding prompt-plan] generated crawl_id=%s profile_id=%s combined_count=%s openai_status=%s",
            crawl.id,
            profile.id,
            len(prompt_texts),
            openai_status or "unknown",
        )

        # Guarantee execution enqueue for onboarding-generated prompts.
        now = django_tz.now()
        stale_cutoff = now - timedelta(minutes=20)
        inflight = list(
            AEOExecutionRun.objects.filter(
                profile=profile,
                status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
            ).order_by("created_at")
        )
        for r in inflight:
            has_snapshots = r.response_snapshots.exists()
            if not has_snapshots and (r.created_at and r.created_at < stale_cutoff):
                r.status = AEOExecutionRun.STATUS_FAILED
                r.error_message = "stale_inflight_no_snapshots_timeout"
                r.finished_at = now
                r.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
                logger.error(
                    "[onboarding execute] stale_inflight_marked_failed run_id=%s profile_id=%s",
                    r.id,
                    profile.id,
                )

        inflight_after_recovery = AEOExecutionRun.objects.filter(
            profile=profile,
            status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
        ).exists()
        if inflight_after_recovery:
            logger.info(
                "[onboarding execute] skipped_enqueue_inflight_exists crawl_id=%s profile_id=%s",
                crawl.id,
                profile.id,
            )
            return

        run_payload = _serialize_aeo_prompt_items(combined)
        run = AEOExecutionRun.objects.create(
            profile=profile,
            prompt_count_requested=len(run_payload),
            status=AEOExecutionRun.STATUS_PENDING,
            error_message=f"onboarding_crawl_id={crawl.id}",
        )
        try:
            run_aeo_phase1_execution_task.delay(run.id, run_payload)
            logger.info(
                "[onboarding execute] phase1_enqueued crawl_id=%s profile_id=%s run_id=%s prompts=%s",
                crawl.id,
                profile.id,
                run.id,
                len(run_payload),
            )
        except Exception as exc:
            run.status = AEOExecutionRun.STATUS_FAILED
            run.error_message = f"phase1_enqueue_failed:{type(exc).__name__}: {exc}"[:2000]
            run.finished_at = django_tz.now()
            run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
            crawl.prompt_plan_status = OnboardingOnPageCrawl.PROMPT_PLAN_FAILED
            crawl.prompt_plan_finished_at = django_tz.now()
            crawl.prompt_plan_error = f"phase1_enqueue_failed:{type(exc).__name__}: {exc}"[:2000]
            crawl.save(
                update_fields=[
                    "prompt_plan_status",
                    "prompt_plan_finished_at",
                    "prompt_plan_error",
                    "updated_at",
                ]
            )
            logger.exception(
                "[onboarding execute] phase1 enqueue failed crawl_id=%s profile_id=%s run_id=%s",
                crawl.id,
                profile.id,
                run.id,
            )
            return
    except Exception as exc:
        crawl.prompt_plan_status = OnboardingOnPageCrawl.PROMPT_PLAN_FAILED
        crawl.prompt_plan_finished_at = django_tz.now()
        crawl.prompt_plan_error = f"{type(exc).__name__}: {exc}"[:2000]
        crawl.save(
            update_fields=[
                "prompt_plan_status",
                "prompt_plan_finished_at",
                "prompt_plan_error",
                "updated_at",
            ]
        )
        logger.exception("[onboarding prompt-plan] failed crawl_id=%s", crawl_id)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def onboarding_review_topics_backfill_task(self, crawl_id: int) -> None:
    """
    Fill ``review_topics`` for legacy crawls that have ranked_keywords but no LLM topics yet.
    Does not enqueue prompt generation; the client requests prompts after topic selection.
    """
    from .models import OnboardingOnPageCrawl
    from .onboarding_review_topics import generate_review_topics_for_domain

    crawl = (
        OnboardingOnPageCrawl.objects.select_related("business_profile")
        .filter(pk=int(crawl_id))
        .first()
    )
    if not crawl:
        logger.warning("[onboarding review_topics backfill] crawl id=%s not found", crawl_id)
        return
    if crawl.status != OnboardingOnPageCrawl.STATUS_COMPLETED:
        return
    if crawl.review_topics:
        return

    rt_list, rt_err = generate_review_topics_for_domain(
        domain=crawl.domain,
        business_profile=crawl.business_profile,
    )
    crawl.review_topics = rt_list
    crawl.review_topics_error = (rt_err or "")[:2000]
    crawl.save(update_fields=["review_topics", "review_topics_error", "updated_at"])
