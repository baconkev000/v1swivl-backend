"""
Celery tasks for SEO enrichment. Run after initial snapshot is saved; update only enrichment
fields and never recalculate main score (seo_score, search_performance_score, search_visibility_percent,
missed_searches_monthly).
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List

from celery import shared_task
from django.conf import settings
from django.utils import timezone as django_tz

logger = logging.getLogger(__name__)

# Duplicate prevention: skip keyword enrichment if already done within this TTL.
KEYWORDS_ENRICHMENT_TTL = timedelta(hours=1)
# Next steps: reuse existing TTL from generate_or_get_next_steps.
NEXT_STEPS_TTL = timedelta(days=7)
# Per-keyword action suggestions TTL (for \"Do these now\" UI).
KEYWORD_ACTION_TTL = timedelta(days=7)


@shared_task(bind=True, autoretry_for=(Exception,), max_retries=0, ignore_result=True)
def enrich_snapshot_keywords_task(self, snapshot_id: int) -> None:
    """
    Load snapshot by id; enrich top_keywords with gap + LLM keywords; update only
    top_keywords and keywords_enriched_at. Does not change core score or visibility metrics.
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
        from .dataforseo_utils import (
            enrich_with_gap_keywords,
            enrich_with_llm_keywords,
            enrich_keyword_ranks_from_labs,
        )

        enrich_with_gap_keywords(domain, location_code, language_code, user, top_keywords)
        enrich_with_llm_keywords(user, location_code, top_keywords)
        rank_stats = enrich_keyword_ranks_from_labs(
            domain=domain,
            location_code=location_code,
            language_code=language_code,
            top_keywords=top_keywords,
            user=user,
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
    rank_pct = (keywords_with_rank / total_keywords * 100.0) if total_keywords > 0 else 0.0
    competitor_pct = (keywords_with_competitor / total_keywords * 100.0) if total_keywords > 0 else 0.0
    logger.info(
        "[SEO async] keyword coverage snapshot_id=%s total=%s rank_non_null_pct=%.2f competitor_data_pct=%.2f",
        snapshot_id,
        total_keywords,
        rank_pct,
        competitor_pct,
    )

    try:
        snapshot.top_keywords = top_keywords_sorted
        snapshot.keywords_enriched_at = django_tz.now()
        snapshot.save(update_fields=["top_keywords", "keywords_enriched_at"])

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
