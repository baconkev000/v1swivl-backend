"""
Synchronous SEO overview snapshot enrichment (gap keywords, LLM keywords, Labs ranks,
metric recompute + persist). Shared by the staff refresh endpoint and post-payment Celery work.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from django.conf import settings
from django.db import transaction

from .business_profile_access import workspace_data_user
from .models import BusinessProfile, SEOOverviewSnapshot
from .third_party_usage import usage_profile_context

logger = logging.getLogger(__name__)


def sync_enrich_current_period_seo_snapshot_for_profile(
    profile: BusinessProfile,
    *,
    data_user_fallback: Optional[Any] = None,
    force_domain_intersection: bool = False,
    abort_on_low_coverage: bool = True,
) -> dict[str, Any]:
    """
    Enrich the current calendar month's SEOOverviewSnapshot for ``profile``'s workspace user,
    matching ``POST /api/seo/refresh-snapshot/`` (minus HTTP).

    If no snapshot row exists yet, calls ``get_or_refresh_seo_score_for_user`` once (no force)
    then retries the lookup.

    Returns a dict with at least:
      - ok: bool — completed without unexpected failure
      - persisted: bool — snapshot row was updated
      - external_api_called: bool
      - aborted_low_coverage: bool — rank coverage below threshold and abort_on_low_coverage
      - snapshot_id: int | None
      - detail: str | None — human-readable when aborted_low_coverage
      - rank_coverage_percent: float | None
    """
    from .dataforseo_utils import (
        enrich_keyword_ranks_from_labs,
        enrich_with_gap_keywords,
        enrich_with_llm_keywords,
        get_or_refresh_seo_score_for_user,
        get_profile_location_code,
        normalize_domain,
        normalize_seo_snapshot_metrics,
        recompute_snapshot_metrics_from_keywords,
        seo_snapshot_context_for_profile,
        sort_top_keywords_for_display,
    )
    from .tasks import (
        generate_keyword_action_suggestions_task,
        generate_snapshot_next_steps_task,
    )

    site_url = str(getattr(profile, "website_url", "") or "").strip()
    if not site_url:
        return {
            "ok": True,
            "persisted": False,
            "external_api_called": False,
            "aborted_low_coverage": False,
            "snapshot_id": None,
            "detail": "no_website_url",
            "rank_coverage_percent": None,
        }

    data_user = workspace_data_user(profile) or data_user_fallback or profile.user
    if data_user is None:
        return {
            "ok": False,
            "persisted": False,
            "external_api_called": False,
            "aborted_low_coverage": False,
            "snapshot_id": None,
            "detail": "no_data_user",
            "rank_coverage_percent": None,
        }

    domain = normalize_domain(site_url) or ""
    if not domain:
        return {
            "ok": False,
            "persisted": False,
            "external_api_called": False,
            "aborted_low_coverage": False,
            "snapshot_id": None,
            "detail": "domain_normalize_failed",
            "rank_coverage_percent": None,
        }

    today = datetime.now(timezone.utc).date()
    start_current = today.replace(day=1)
    snapshot_mode, snapshot_location_code = seo_snapshot_context_for_profile(profile)

    snapshot = (
        SEOOverviewSnapshot.objects.filter(
            business_profile=profile,
            period_start=start_current,
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
        )
        .order_by("-last_fetched_at")
        .first()
    )

    if not snapshot:
        get_or_refresh_seo_score_for_user(
            data_user,
            site_url=site_url,
            business_profile=profile,
            skip_keyword_enrichment_enqueue=True,
        )
        snapshot = (
            SEOOverviewSnapshot.objects.filter(
                business_profile=profile,
                period_start=start_current,
                cached_location_mode=snapshot_mode,
                cached_location_code=snapshot_location_code,
            )
            .order_by("-last_fetched_at")
            .first()
        )

    if not snapshot:
        return {
            "ok": True,
            "persisted": False,
            "external_api_called": False,
            "aborted_low_coverage": False,
            "snapshot_id": None,
            "detail": "no_snapshot_after_bootstrap",
            "rank_coverage_percent": None,
        }

    external_api_called = False
    location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    language_code = getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en")
    user = snapshot.user

    top_keywords: list[dict] = [dict(k) for k in (getattr(snapshot, "top_keywords", None) or [])]
    with usage_profile_context(profile):
        enrich_with_gap_keywords(
            domain=domain,
            location_code=location_code,
            language_code=language_code,
            user=user,
            top_keywords=top_keywords,
            force_refresh=force_domain_intersection,
            business_profile=profile,
        )
        external_api_called = True
        enrich_with_llm_keywords(
            user=user,
            location_code=location_code,
            top_keywords=top_keywords,
            business_profile=profile,
        )
        rank_stats = enrich_keyword_ranks_from_labs(
            domain=domain,
            location_code=location_code,
            language_code=language_code,
            top_keywords=top_keywords,
            user=user,
            business_profile=profile,
            force_refresh_domain_intersection=force_domain_intersection,
        )
        external_api_called = True

    total = int(rank_stats.get("total") or 0)
    ranked_after = int(rank_stats.get("non_null_after") or 0)
    coverage = (ranked_after / total) if total > 0 else 0.0
    min_coverage = float(getattr(settings, "SEO_RANK_ENRICHMENT_MIN_COVERAGE", 0.05))
    logger.info(
        "[SEO sync enrich] rank enrichment coverage user_id=%s total=%s ranked_after=%s coverage=%.2f%% filled_ranked=%s filled_gap=%s",
        getattr(user, "id", None),
        total,
        ranked_after,
        coverage * 100.0,
        int(rank_stats.get("filled_from_ranked") or 0),
        int(rank_stats.get("filled_from_gap") or 0),
    )
    if total > 0 and coverage < min_coverage and abort_on_low_coverage:
        detail = (
            "Rank enrichment coverage is too low; refresh aborted to avoid returning all-none ranks."
        )
        logger.warning(
            "[SEO sync enrich] %s user_id=%s coverage=%.2f%% threshold=%.2f%%",
            detail,
            getattr(user, "id", None),
            coverage * 100.0,
            min_coverage * 100.0,
        )
        return {
            "ok": True,
            "persisted": False,
            "external_api_called": external_api_called,
            "aborted_low_coverage": True,
            "snapshot_id": int(snapshot.id),
            "detail": detail,
            "rank_coverage_percent": round(coverage * 100.0, 2),
        }

    max_kw = int(getattr(settings, "SEO_TOP_KEYWORDS_MAX_PERSISTED", 200))
    for _row in top_keywords:
        if not (_row or {}).get("keyword_origin"):
            _row["keyword_origin"] = "ranked"
    top_keywords_sorted = sort_top_keywords_for_display(top_keywords, max_rows=max_kw)
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
        keywords_with_outranking_competitor / total_keywords * 100.0 if total_keywords > 0 else 0.0
    )
    logger.info(
        "[SEO sync enrich] keyword coverage user_id=%s total=%s rank_non_null_pct=%.2f competitor_data_pct=%.2f outranking_competitor_pct=%.2f",
        getattr(user, "id", None),
        total_keywords,
        rank_pct,
        competitor_pct,
        outranking_competitor_pct,
    )
    metrics = normalize_seo_snapshot_metrics(
        recompute_snapshot_metrics_from_keywords(
            top_keywords=top_keywords_sorted,
            domain=domain,
            location_code=location_code,
            language_code=language_code,
            seo_location_mode=str(snapshot_mode or "organic"),
            business_profile=profile,
        )
    )
    logger.info(
        "[SEO sync enrich] recompute user_id=%s keywords_with_rank=%s estimated_traffic_before=%s estimated_traffic_after=%s appearances_before=%s appearances_after=%s total_search_volume_before=%s total_search_volume_after=%s visibility_before=%s visibility_after=%s missed_before=%s missed_after=%s",
        getattr(user, "id", None),
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
            "[SEO sync enrich] consistency_check user_id=%s ranked_keywords=%s visibility_zero_with_volume=true",
            getattr(user, "id", None),
            keywords_with_rank,
        )

    with transaction.atomic():
        snapshot_context: dict[str, Any] = {
            "mode": snapshot_mode,
            "code": snapshot_location_code,
            "label": "",
        }
        if snapshot_mode == "local":
            default_location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
            resolved_code, _used_fallback, resolved_label = get_profile_location_code(
                profile, default_location_code
            )
            snapshot_context["code"] = int(resolved_code or 0)
            snapshot_context["label"] = str(resolved_label or "")
        snapshot.top_keywords = top_keywords_sorted
        snapshot.keywords_enriched_at = datetime.now(timezone.utc)
        snapshot.refreshed_at = datetime.now(timezone.utc)
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
                "refreshed_at",
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

    generate_snapshot_next_steps_task.delay(snapshot.id)
    generate_keyword_action_suggestions_task.delay(snapshot.id)

    return {
        "ok": True,
        "persisted": True,
        "external_api_called": external_api_called,
        "aborted_low_coverage": False,
        "snapshot_id": int(snapshot.id),
        "detail": None,
        "rank_coverage_percent": round(coverage * 100.0, 2) if total > 0 else None,
    }


def run_full_seo_snapshot_for_profile(
    profile: BusinessProfile,
    *,
    data_user_fallback: Optional[Any] = None,
    force_refresh: bool = True,
    abort_on_low_coverage: bool = True,
) -> dict[str, Any]:
    """
    One-shot SEO pipeline for a profile: ranked Labs snapshot (``get_or_refresh_seo_score_for_user``)
    then ``sync_enrich_current_period_seo_snapshot_for_profile`` (gap + LLM seeds + Labs rank fill,
    metrics persist, next-steps + keyword-action tasks enqueued).

    Used after onboarding billing bypass, new company profile creation, staff snapshot refresh, and
    post-payment Celery follow-up so ``top_keywords`` includes ranked and enrichment rows without
    relying on a separate async ``enrich_snapshot_keywords_task`` from the ranked-only path.
    """
    from .business_profile_access import workspace_data_user
    from .dataforseo_utils import get_or_refresh_seo_score_for_user, normalize_domain

    site_url = str(getattr(profile, "website_url", "") or "").strip()
    if not site_url:
        return {
            "ok": True,
            "persisted": False,
            "external_api_called": False,
            "aborted_low_coverage": False,
            "snapshot_id": None,
            "detail": "no_website_url",
            "rank_coverage_percent": None,
        }
    if not (normalize_domain(site_url) or "").strip():
        return {
            "ok": False,
            "persisted": False,
            "external_api_called": False,
            "aborted_low_coverage": False,
            "snapshot_id": None,
            "detail": "domain_normalize_failed",
            "rank_coverage_percent": None,
        }
    data_user = data_user_fallback or workspace_data_user(profile) or getattr(profile, "user", None)
    if data_user is None:
        return {
            "ok": False,
            "persisted": False,
            "external_api_called": False,
            "aborted_low_coverage": False,
            "snapshot_id": None,
            "detail": "no_data_user",
            "rank_coverage_percent": None,
        }
    bundle = get_or_refresh_seo_score_for_user(
        data_user,
        site_url=site_url,
        force_refresh=force_refresh,
        business_profile=profile,
        skip_keyword_enrichment_enqueue=True,
    )
    if bundle is None:
        return {
            "ok": False,
            "persisted": False,
            "external_api_called": False,
            "aborted_low_coverage": False,
            "snapshot_id": None,
            "detail": "ranked_refresh_unavailable",
            "rank_coverage_percent": None,
        }
    return sync_enrich_current_period_seo_snapshot_for_profile(
        profile,
        data_user_fallback=data_user,
        abort_on_low_coverage=abort_on_low_coverage,
    )
