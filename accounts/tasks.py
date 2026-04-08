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

logger = logging.getLogger(__name__)

# Duplicate prevention: skip keyword enrichment if already done within this TTL.
# Requirement: ranks + competitor enrichment should stay stable for 7 days unless
# user explicitly clicks the refresh button.
KEYWORDS_ENRICHMENT_TTL = timedelta(days=7)
# Next steps: reuse existing TTL from generate_or_get_next_steps.
NEXT_STEPS_TTL = timedelta(days=7)
# Per-keyword action suggestions TTL (for \"Do these now\" UI).
KEYWORD_ACTION_TTL = timedelta(days=7)


def _aeo_target_prompt_count() -> int:
    testing_mode = bool(getattr(settings, "AEO_TESTING_MODE", False))
    if testing_mode:
        try:
            return max(1, int(getattr(settings, "AEO_TEST_PROMPT_COUNT", 10)))
        except (TypeError, ValueError):
            return 10
    try:
        return max(1, int(getattr(settings, "AEO_PROD_PROMPT_COUNT", 50)))
    except (TypeError, ValueError):
        return 50


def _is_onboarding_sample_size() -> bool:
    return _aeo_target_prompt_count() == 10


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


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def enrich_snapshot_keywords_task(self, snapshot_id: int) -> None:
    """
    Load snapshot by id; enrich top_keywords with gap + LLM keywords; then atomically
    recompute and persist all dependent SEO snapshot metrics.
    """
    from .models import SEOOverviewSnapshot

    try:
        snapshot = SEOOverviewSnapshot.objects.filter(pk=snapshot_id).first()
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
    profile_for_usage = (
        user.business_profiles.filter(is_main=True).first()
        or user.business_profiles.first()
    )

    # Copy so we don't mutate in-place until we're ready to save
    top_keywords: List[Dict[str, Any]] = [dict(k) for k in (getattr(snapshot, "top_keywords", None) or [])]

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
        )

        with usage_profile_context(profile_for_usage):
            enrich_with_gap_keywords(domain, location_code, language_code, user, top_keywords)
            enrich_with_llm_keywords(user, location_code, top_keywords)
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

    top_keywords_sorted = sorted(
        top_keywords,
        key=lambda x: x.get("search_volume", 0),
        reverse=True,
    )[:20]

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

    if getattr(snapshot, "seo_next_steps_refreshed_at", None):
        if django_tz.now() - snapshot.seo_next_steps_refreshed_at <= NEXT_STEPS_TTL:
            return

    try:
        from .openai_utils import generate_seo_next_steps
    except ImportError as e:
        logger.warning("[SEO async] generate_snapshot_next_steps_task openai_utils import failed: %s", e)
        return

    # Build minimal seo_data for generate_seo_next_steps (no onpage in task context)
    seo_data = {
        "seo_score": int(snapshot.search_performance_score or 0),
        "missed_searches_monthly": int(getattr(snapshot, "missed_searches_monthly", 0) or 0),
        "organic_visitors": int(snapshot.organic_visitors or 0),
        "total_search_volume": int(getattr(snapshot, "total_search_volume", 0) or 0),
        "search_visibility_percent": int(getattr(snapshot, "search_visibility_percent", 0) or 0),
        "top_keywords": getattr(snapshot, "top_keywords", None) or [],
        "onpage_issue_summaries": {},
    }

    try:
        steps = generate_seo_next_steps(seo_data)
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
        snapshot.save(update_fields=["seo_next_steps", "seo_next_steps_refreshed_at"])
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
        if django_tz.now() - snapshot.keyword_action_suggestions_refreshed_at <= KEYWORD_ACTION_TTL:
            return

    keywords: list[Dict[str, Any]] = getattr(snapshot, "top_keywords", None) or []
    if not keywords:
        return

    try:
        from .openai_utils import generate_keyword_action_suggestions
    except ImportError as e:
        logger.warning(
            "[SEO async] generate_keyword_action_suggestions_task openai_utils import failed: %s",
            e,
        )
        return

    try:
        suggestions = generate_keyword_action_suggestions(keywords)
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
    target = _aeo_target_prompt_count()
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
            if _is_onboarding_sample_size() and providers is None:
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
        run.background_status = AEOExecutionRun.STAGE_RUNNING if _is_onboarding_sample_size() else run.background_status
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
        if _is_onboarding_sample_size() and providers is None:
            run_aeo_phase2_confidence_task.delay(run.id, selected)
            run_aeo_phase4_scoring_task.delay(run.id)
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
        saved_prompts = [{"prompt": p} for p in (profile.selected_aeo_prompts or []) if str(p).strip()]

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

    from accounts.third_party_usage import usage_profile_context
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
                return
            break
        aggs.sort(key=lambda a: (_aggregate_priority_bucket(a), str(a.prompt_hash)))

        provider_batches: dict[str, list[dict[str, Any]]] = {"openai": [], "gemini": []}
        provider_priority: dict[str, dict[str, int]] = {"openai": {}, "gemini": {}}

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
            if int(agg.openai_pass_count or 0) < PASSES_PER_PROVIDER_TARGET:
                _push("openai", spec, 0)
            elif (
                bool(agg.openai_third_pass_required)
                and not bool(agg.openai_third_pass_ran)
                and int(agg.openai_pass_count or 0) < 3
            ):
                _push("openai", spec, 1)
            if int(agg.gemini_pass_count or 0) < PASSES_PER_PROVIDER_TARGET:
                _push("gemini", spec, 0)
            elif (
                bool(agg.gemini_third_pass_required)
                and not bool(agg.gemini_third_pass_ran)
                and int(agg.gemini_pass_count or 0) < 3
            ):
                _push("gemini", spec, 1)

        for provider in ("openai", "gemini"):
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
        if not provider_batches["openai"] and not provider_batches["gemini"]:
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
        selected = plan_items_from_saved_prompt_strings([str(x) for x in saved if str(x).strip()])

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
        selected = plan_items_from_saved_prompt_strings([str(x) for x in saved if str(x).strip()])

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
        run_aeo_phase4_scoring_task.delay(run.id)
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
        run_aeo_phase4_scoring_task.delay(run.id)
    except Exception as exc:
        run.extraction_status = AEOExecutionRun.STAGE_FAILED
        run.error_message = f"{run.error_message}\nphase3:{type(exc).__name__}: {exc}".strip()
        run.save(update_fields=["extraction_status", "error_message", "updated_at"])
        logger.exception("[AEO phase3] extraction failed run_id=%s", run.id)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def run_aeo_phase4_scoring_task(self, run_id: int) -> None:
    from .models import AEOExecutionRun
    from .aeo.aeo_scoring_utils import calculate_aeo_scores_for_business, calculate_layered_scores_from_aggregates
    from .models import AEOScoreSnapshot

    run = AEOExecutionRun.objects.select_related("profile").filter(pk=run_id).first()
    if not run:
        logger.warning("[AEO phase4] run not found run_id=%s", run_id)
        return
    if run.scoring_status == AEOExecutionRun.STAGE_COMPLETED and run.score_snapshot_id:
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
        return

    run.scoring_status = AEOExecutionRun.STAGE_RUNNING
    run.save(update_fields=["scoring_status", "updated_at"])
    logger.info("[AEO phase4] scoring start run_id=%s profile_id=%s", run.id, run.profile_id)
    try:
        if _is_onboarding_sample_size():
            calculate_layered_scores_from_aggregates(
                run.profile,
                execution_run_id=run.id,
                score_layer=AEOScoreSnapshot.LAYER_SAMPLE,
                save=True,
            )
            score_data = calculate_layered_scores_from_aggregates(
                run.profile,
                execution_run_id=run.id,
                score_layer=AEOScoreSnapshot.LAYER_CONFIDENCE,
                save=True,
            )
        else:
            score_data = calculate_aeo_scores_for_business(run.profile, save=True)
        run.score_snapshot_id = score_data.get("snapshot_id")
        run.scoring_status = AEOExecutionRun.STAGE_COMPLETED
        run.save(update_fields=["score_snapshot_id", "scoring_status", "updated_at"])
        logger.info(
            "[AEO phase4] scoring complete run_id=%s profile_id=%s score_snapshot_id=%s total_prompts=%s",
            run.id,
            run.profile_id,
            run.score_snapshot_id,
            int(score_data.get("total_prompts") or 0),
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
    if run.recommendation_status == AEOExecutionRun.STAGE_COMPLETED and run.recommendation_run_id:
        return

    run.recommendation_status = AEOExecutionRun.STAGE_RUNNING
    run.save(update_fields=["recommendation_status", "updated_at"])
    logger.info("[AEO phase5] recommendation start run_id=%s profile_id=%s", run.id, run.profile_id)
    try:
        data = generate_aeo_recommendations(run.profile, save=True)
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


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def onboarding_onpage_crawl_task(self, crawl_id: int) -> None:
    """DataForSEO On-Page crawl for onboarding (10 pages, Celery)."""
    from .onboarding_onpage import execute_onboarding_onpage_crawl

    execute_onboarding_onpage_crawl(int(crawl_id))


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def onboarding_prompt_generation_task(self, crawl_id: int) -> None:
    """
    Build onboarding prompt plan in background immediately after keyword pipeline.
    Idempotent at crawl level (queued/running/completed states).
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
    selected_topics = [
        str(r.get("keyword") or "").strip()
        for r in (crawl.ranked_keywords or [])
        if str(r.get("keyword") or "").strip()
    ]
    if not selected_topics:
        crawl.prompt_plan_status = OnboardingOnPageCrawl.PROMPT_PLAN_FAILED
        crawl.prompt_plan_error = "no_ranked_keywords"
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

    details = []
    for row in crawl.ranked_keywords or []:
        kw = str((row or {}).get("keyword") or "").strip()
        if not kw:
            continue
        d = {"keyword": kw}
        for key in ("search_volume", "rank", "rank_group", "aeo_score", "aeo_category", "aeo_reason"):
            if key in row and row.get(key) not in (None, ""):
                d[key] = row.get(key)
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
        )
        target_prompt_count = _aeo_target_prompt_count()
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
        verify_saved = [
            str(x).strip() for x in (profile.selected_aeo_prompts or []) if str(x).strip()
        ]
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
