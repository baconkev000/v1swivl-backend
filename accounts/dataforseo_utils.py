from __future__ import annotations

"""
Helpers for calling DataForSEO Labs APIs for the SEO Agent.

Primary endpoints used:
- POST /v3/dataforseo_labs/google/ranked_keywords/live      → current visibility
- POST /v3/dataforseo_labs/google/domain_intersection/live  → keyword gap vs competitors
"""

from typing import Any, Dict, List, Optional

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import json
import math
import requests
from django.conf import settings

from .models import SEOOverviewSnapshot, OnPageAuditSnapshot
from .onpage_audit import run_onpage_audit_for_user

logger = logging.getLogger(__name__)

BASE_URL = "https://api.dataforseo.com"
DEBUG_LOG_PATH = "debug-098bfd.log"


def _get_auth() -> Optional[tuple[str, str]]:
    """
    Return (login, password) tuple for DataForSEO HTTP Basic auth.

    Expected settings or env:
    - DATAFORSEO_LOGIN
    - DATAFORSEO_PASSWORD
    """
    login = getattr(settings, "DATAFORSEO_LOGIN", None)
    password = getattr(settings, "DATAFORSEO_PASSWORD", None)

    if not login or not password:
        logger.warning(
            "[DataForSEO] Missing DATAFORSEO_LOGIN/DATAFORSEO_PASSWORD; skipping API call.",
        )
        return None
    return str(login), str(password)


def _post(path: str, payload: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Low-level POST helper with basic error handling."""
    auth = _get_auth()
    if auth is None:
        return None

    url = f"{BASE_URL}{path}"
    try:
        resp = requests.post(
            url,
            json=payload,
            auth=auth,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - network failure
        logger.exception("[DataForSEO] POST %s failed: %s", path, exc)
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "accounts/dataforseo_utils.py:_post",
                    "message": "HTTP error when calling DataForSEO",
                    "data": {"path": path, "error": str(exc)},
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return None

    if resp.status_code != 200:
        # DataForSEO returns 200 with status_code inside body for logical errors;
        # non-200 here is a transport-level issue.
        logger.warning(
            "[DataForSEO] POST %s HTTP %s: %s",
            path,
            resp.status_code,
            resp.text[:500],
        )
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "accounts/dataforseo_utils.py:_post",
                    "message": "Non-200 from DataForSEO",
                    "data": {
                        "path": path,
                        "status_code": resp.status_code,
                        "body_preview": resp.text[:200],
                    },
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return None

    try:
        data = resp.json()
    except ValueError:  # pragma: no cover - unexpected non-JSON
        logger.warning("[DataForSEO] POST %s returned non-JSON body.", path)
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "accounts/dataforseo_utils.py:_post",
                    "message": "Non-JSON response from DataForSEO",
                    "data": {"path": path, "body_preview": resp.text[:200]},
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return None

    # DataForSEO wraps results in tasks / result; callers will unpack further.
    # #region agent log
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "sessionId": "098bfd",
                "runId": "pre-fix",
                "hypothesisId": "H1",
                "location": "accounts/dataforseo_utils.py:_post",
                "message": "DataForSEO response summary",
                "data": {
                    "path": path,
                    "has_tasks": bool((data or {}).get("tasks")),
                },
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion
    return data


def _ctr_for_position(position: int) -> float:
    """
    Simple CTR curve approximating typical SERP click-through rates.
    Position 1 ≈ 0.28, 2 ≈ 0.15, 3 ≈ 0.10, 4 ≈ 0.07, 5 ≈ 0.05, 6–10 decreasing.
    """
    if position <= 0:
        return 0.0
    if position == 1:
        return 0.28
    if position == 2:
        return 0.15
    if position == 3:
        return 0.10
    if position == 4:
        return 0.07
    if position == 5:
        return 0.05
    if 6 <= position <= 10:
        # Linearly decay from 0.04 at 6 → 0.01 at 10
        return max(0.01, 0.04 - (position - 6) * 0.0075)
    if 11 <= position <= 20:
        return 0.005
    return 0.002


def _extract_keyword_difficulty(keyword_info: Dict[str, Any]) -> Optional[float]:
    """
    Extract a difficulty / competition score from DataForSEO keyword_info.
    Handles either 0–1 or 0–100 scales and normalises to 0–100.
    """
    if not keyword_info:
        return None

    for key in ("competition", "competition_level", "difficulty", "keyword_difficulty"):
        if key in keyword_info and keyword_info[key] is not None:
            try:
                val = float(keyword_info[key])
            except (TypeError, ValueError):
                continue
            # If looks like 0–1, scale to 0–100
            if 0.0 <= val <= 1.0:
                return val * 100.0
            return max(0.0, min(100.0, val))

    return None


def _get_competitor_average_traffic(
    target_domain: str,
    *,
    location_code: int,
    language_code: str,
) -> float:
    """
    Call competitors_domain/live to estimate the average organic traffic/visibility
    of the top competitors for the given domain.
    """
    payload = [
        {
            "target": target_domain,
            "location_code": int(location_code),
            "language_code": language_code,
        },
    ]

    data = _post("/v3/dataforseo_labs/google/competitors_domain/live", payload)
    if not data:
        return 0.0

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            return 0.0
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            return 0.0
        result = results[0]
        items = result.get("items") or []
        if not items:
            return 0.0

        competitor_scores: List[float] = []
        for item in items:
            metrics = item.get("organic_metrics") or item
            raw_vis = (
                metrics.get("estimated_traffic")
                or metrics.get("visibility")
                or metrics.get("sum_search_volume")
                or 0
            )
            try:
                vis = float(raw_vis)
            except (TypeError, ValueError):
                vis = 0.0
            if vis > 0:
                competitor_scores.append(vis)

        if not competitor_scores:
            return 0.0

        return sum(competitor_scores) / len(competitor_scores)
    except Exception:
        logger.exception(
            "[DataForSEO] competitors_domain parsing failed for target_domain=%s",
            target_domain,
        )
        return 0.0


def get_ranked_keywords_visibility(
    target_domain: str,
    *,
    location_code: int,
    language_code: str = "en",
    limit: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Call ranked_keywords/live to get visibility & ranking metrics for a domain.

    Returns a dict with at least:
    - visibility: float
    - keywords_count: int
    - top3_positions: int
    """
    # #region agent log
    from . import debug_log as _debug
    _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:entry", "DataForSEO call", {"target_domain": target_domain, "location_code": location_code}, "H3")
    # #endregion
    payload = [
        {
            "target": target_domain,
            "location_code": int(location_code),
            "language_code": language_code,
            "limit": int(limit),
        },
    ]

    data = _post("/v3/dataforseo_labs/google/ranked_keywords/live", payload)
    # #region agent log
    _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:after_post", "API response", {"has_data": data is not None}, "H4")
    # #endregion
    if not data:
        return None

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            # #region agent log
            _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:no_tasks", "no tasks in response", {"target_domain": target_domain}, "H5")
            # #endregion
            logger.warning("[DataForSEO] ranked_keywords: no tasks in response for %s", target_domain)
            return None
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            # #region agent log
            _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:no_result", "no result for domain", {"target_domain": target_domain}, "H5")
            # #endregion
            logger.warning("[DataForSEO] ranked_keywords: no result for %s", target_domain)
            return None
        result = results[0]
        items = result.get("items") or []

        keywords_count = len(items)

        top3 = sum(
            1 for item in items
            if item.get("rank_absolute") and item["rank_absolute"] <= 3
        )

        # crude visibility proxy
        visibility = sum(
            (item.get("search_volume") or 0)
            for item in items
        )

        logger.info(
            "[DataForSEO] ranked_keywords target=%s visibility=%.2f keywords_count=%s top3=%s",
            target_domain,
            visibility,
            keywords_count,
            top3,
        )

        # #region agent log
        _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:success", "returning visibility dict", {"keywords_count": keywords_count, "top3_positions": top3}, "H5")
        # #endregion
        return {
            "visibility": visibility,
            "keywords_count": keywords_count,
            "top3_positions": top3,
        }
    except Exception as exc:  # pragma: no cover - defensive
        # #region agent log
        _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:exception", "parsing exception", {"exc_type": type(exc).__name__, "exc_msg": str(exc)[:200]}, "H4")
        # #endregion
        logger.exception(
            "[DataForSEO] ranked_keywords parsing failed for %s: %s",
            target_domain,
            exc,
        )
        return None


def get_keyword_gap_keywords(
    target_domain: str,
    competitor_domains: List[str],
    *,
    location_code: int,
    language_code: str = "en",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Call domain_intersection/live to compute keyword gaps vs competitors.

    For now we return a simplified list of items suitable for the frontend SEO keywords table:
    - keyword
    - search_volume
    """
    cleaned_competitors = [c.strip().lower() for c in competitor_domains if c.strip()]
    if not cleaned_competitors:
        logger.info(
            "[DataForSEO] domain_intersection skipped for %s: no competitors configured",
            target_domain,
        )
        return []

    # The domain_intersection endpoint expects target1/target2 rather than a
    # "targets" array. Call it once per competitor and merge the results.
    # We also collect a short list of competitor URLs that are currently
    # ranking for each keyword so the frontend can show where competitors
    # dominate the SERP.
    aggregated: Dict[str, Dict[str, Any]] = {}

    for competitor in cleaned_competitors:
        payload = [
            {
                "target1": target_domain,
                "target2": competitor,
                "location_code": int(location_code),
                "language_code": language_code,
                # intersections=true ensures we only get queries where both
                # domains have ranking URLs in the SERP.
                "intersections": True,
                "limit": int(limit),
            },
        ]

        data = _post("/v3/dataforseo_labs/google/domain_intersection/live", payload)
        if not data:
            continue
        try:
            tasks = data.get("tasks") or []
            if not tasks:
                logger.warning(
                    "[DataForSEO] domain_intersection: no tasks in response for %s vs %s",
                    target_domain,
                    competitor,
                )
                continue

            for task in tasks:
                results = task.get("result") or []
                if not results:
                    logger.warning(
                        "[DataForSEO] domain_intersection: no result for %s vs %s",
                        target_domain,
                        competitor,
                    )
                    continue

                for result in results:
                    items = result.get("items") or []
                    for item in items:
                        keyword_data = item.get("keyword_data") or {}
                        kw = keyword_data.get("keyword") or item.get("keyword")
                        if not kw:
                            continue

                        keyword_info = keyword_data.get("keyword_info") or {}
                        search_volume = (
                            keyword_info.get("search_volume")
                            or keyword_info.get("search_volume_global")
                            or keyword_info.get("sum_search_volume")
                            or item.get("search_volume")
                            or item.get("sum_search_volume")
                        )

                        # For each keyword, keep track of competitor URLs and
                        # ranks for the competitor side of the intersection.
                        serp_el = item.get("second_domain_serp_element") or item.get("first_domain_serp_element") or {}
                        comp_domain = (
                            serp_el.get("main_domain")
                            or serp_el.get("domain")
                            or competitor
                        )
                        comp_url = serp_el.get("url") or ""
                        comp_rank = serp_el.get("rank_absolute")

                        existing = aggregated.get(kw)
                        if existing is None:
                            competitors_list: List[Dict[str, Any]] = []
                            if comp_url:
                                competitors_list.append(
                                    {
                                        "domain": comp_domain,
                                        "url": comp_url,
                                        "rank": comp_rank,
                                    },
                                )
                            aggregated[kw] = {
                                "keyword": kw,
                                "search_volume": search_volume,
                                "competitors": competitors_list,
                            }
                        else:
                            # Keep the highest search volume we have seen.
                            prev_sv = existing.get("search_volume") or 0
                            curr_sv = search_volume or 0
                            if curr_sv > prev_sv:
                                existing["search_volume"] = search_volume

                            # Merge competitor URLs, keeping at most 3 per keyword,
                            # preferring better (lower) ranks.
                            if comp_url:
                                comp_entry = {
                                    "domain": comp_domain,
                                    "url": comp_url,
                                    "rank": comp_rank,
                                }
                                existing_list: List[Dict[str, Any]] = existing.setdefault("competitors", [])

                                # Avoid exact duplicates (same domain + URL).
                                if not any(
                                    c.get("domain") == comp_entry["domain"] and c.get("url") == comp_entry["url"]
                                    for c in existing_list
                                ):
                                    existing_list.append(comp_entry)

                                # Keep only top 3 by rank (ascending), with None ranks sorted last.
                                existing_list.sort(
                                    key=lambda c: (c.get("rank") or 10_000_000),
                                )
                                if len(existing_list) > 3:
                                    del existing_list[3:]

        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "[DataForSEO] domain_intersection parsing failed for %s vs %s: %s",
                target_domain,
                competitor,
                exc,
            )
            continue

    gap_keywords = list(aggregated.values())

    logger.info(
        "[DataForSEO] domain_intersection target=%s competitors=%s keyword_count=%s",
        target_domain,
        ",".join(cleaned_competitors),
        len(gap_keywords),
    )
    return gap_keywords


def compute_professional_seo_score(
    *,
    estimated_traffic: float,
    keywords_count: int,
    top3_positions: int,
    top10_positions: int,
    avg_keyword_difficulty: Optional[float],
    competitor_avg_traffic: float,
) -> int:
    """
    Collapse core SEO metrics into a single 0–100 "professional grade" score using:
    - estimated organic traffic (via CTR curve)
    - keyword breadth
    - ranking quality (share in top 3 / top 10)
    - keyword difficulty strength
    - competitive market share vs. top competitors

    All components are soft-scaled so smaller sites can still make meaningful progress.
    """
    try:
        traffic = max(float(estimated_traffic), 0.0)
        k = max(int(keywords_count), 0)
        t = max(int(top3_positions), 0)
        t10 = max(int(top10_positions), 0)
    except (TypeError, ValueError):
        traffic, k, t, t10 = 0.0, 0, 0, 0

    import math

    # 1) Traffic visibility: log-scaled estimated organic traffic.
    # 0 → 0, 100 → ~30, 1k → ~55, 10k → ~75, 100k → ~90, 1M+ → ~100
    if traffic <= 0:
        visibility_score = 0.0
    else:
        visibility_score = min(100.0, (math.log10(traffic + 10.0) / 6.0) * 100.0)

    # 2) Keyword breadth: log-scaled number of ranking keywords.
    # 1 → ~15, 10 → ~35, 100 → ~60, 1k → ~85, 10k → ~100
    if k <= 0:
        breadth_score = 0.0
    else:
        breadth_score = min(100.0, (math.log10(k + 1.0) / 4.0) * 100.0)

    # 3) Ranking quality: combination of share in top 3 and share in top 10.
    if k <= 0:
        ranking_score = 0.0
    else:
        top3_share = min(t / k, 1.0)
        top10_share = min(t10 / k, 1.0)
        ranking_score = (top3_share * 0.7 + top10_share * 0.3) * 100.0

    # 4) Keyword difficulty strength: reward sites ranking on harder keywords.
    # We treat difficulty as 0–100 where higher means more competitive queries.
    if avg_keyword_difficulty is None:
        difficulty_score = 0.0
    else:
        d = max(0.0, min(100.0, float(avg_keyword_difficulty)))
        # Slightly compress extremes so very hard portfolios don't instantly max out.
        difficulty_score = 10.0 + (d * 0.8)
        difficulty_score = max(0.0, min(100.0, difficulty_score))

    # 5) Competitive market share: share of traffic vs. average competitor.
    if traffic <= 0 or competitor_avg_traffic <= 0:
        market_share_score = 0.0
    else:
        # ratio > 1 means above-average vs. top competitors.
        ratio = traffic / max(competitor_avg_traffic, 1e-6)
        # Soft saturation: 0.5 → ~33, 1 → ~66, 2 → ~85, 3+ → ~95
        market_share_score = min(
            95.0,
            (math.log10(ratio + 1.0) / math.log10(4.0)) * 100.0,
        )

    # Weighted blend:
    # - visibility: 40%
    # - breadth: 20%
    # - ranking quality: 15%
    # - keyword difficulty: 15%
    # - competitive strength: 10%
    final = (
        visibility_score * 0.40
        + breadth_score * 0.20
        + ranking_score * 0.15
        + difficulty_score * 0.15
        + market_share_score * 0.10
    )

    return max(0, min(100, int(round(final))))


def get_or_refresh_seo_score_for_user(
    user,
    *,
    site_url: str | None,
) -> Dict[str, Any] | None:
    """
    Fetch a cached, professional-grade SEO score + core metrics for the given user,
    refreshing from DataForSEO at most once per hour (same cadence as the dashboard)
    and combining:
    - Search Performance (ranked_keywords + competitors)
    - On-Page SEO (metadata, headings, alt text)
    - Technical SEO (links, canonical, robots, sitemap)

    Returns a dict with at least:
    - seo_score (Overall SEO Score 0–100)
    - search_performance_score
    - onpage_seo_score
    - technical_seo_score
    - organic_visitors (estimated traffic)
    - keywords_ranking
    - top3_positions
    or None if no website is configured / domain cannot be derived.
    """
    today = datetime.now(timezone.utc).date()
    start_current = today.replace(day=1)
    if not site_url:
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H2",
                    "location": "accounts/dataforseo_utils.py:get_or_refresh_seo_score_for_user",
                    "message": "No site_url; skipping SEO score calculation",
                    "data": {"user_id": getattr(user, "id", None)},
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return None

    parsed = urlparse(site_url)
    domain = (parsed.netloc or parsed.path or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if not domain:
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H2",
                    "location": "accounts/dataforseo_utils.py:get_or_refresh_seo_score_for_user",
                    "message": "Could not normalise domain from site_url",
                    "data": {"user_id": getattr(user, "id", None), "site_url": site_url},
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return None

    location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    language_code = getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en")

    # Call ranked_keywords/live with richer parsing for the scoring algorithm.
    payload = [
        {
            "target": domain,
            "location_code": int(location_code),
            "language_code": language_code,
            "limit": 100,
        },
    ]

    logger.info(
        "[SEO score] ranked_keywords request user_id=%s domain=%s payload=%s",
        getattr(user, "id", None),
        domain,
        payload,
    )
    ranked_data = _post("/v3/dataforseo_labs/google/ranked_keywords/live", payload)
    if not ranked_data:
        logger.warning(
            "[SEO score] ranked_keywords returned no data for user_id=%s domain=%s",
            getattr(user, "id", None),
            domain,
        )
        seo_score = compute_professional_seo_score(
            estimated_traffic=0.0,
            keywords_count=0,
            top3_positions=0,
            top10_positions=0,
            avg_keyword_difficulty=None,
            competitor_avg_traffic=0.0,
        )
        # No search performance data; still try to attach on-page snapshot if available.
        search_performance_score = seo_score
        onpage_snapshot = run_onpage_audit_for_user(user, site_url)
        onpage_score = onpage_snapshot.onpage_seo_score if onpage_snapshot else 0
        technical_score = onpage_snapshot.technical_seo_score if onpage_snapshot else 0
        overall = int(round(
            search_performance_score * 0.5
            + onpage_score * 0.3
            + technical_score * 0.2
        ))
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H3",
                    "location": "accounts/dataforseo_utils.py:get_or_refresh_seo_score_for_user",
                    "message": "No ranked_data; using fallback SEO score",
                    "data": {"user_id": getattr(user, "id", None), "domain": domain, "seo_score": seo_score},
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return {
            "seo_score": overall,
            "search_performance_score": search_performance_score,
            "onpage_seo_score": onpage_score,
            "technical_seo_score": technical_score,
            "organic_visitors": 0,
            "keywords_ranking": 0,
            "top3_positions": 0,
            "search_visibility_percent": 0,
            "missed_searches_monthly": 0,
        }

    try:
        tasks = ranked_data.get("tasks") or []
        if not tasks:
            raise ValueError("no tasks")
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            raise ValueError("no results")
        result = results[0]
        items = result.get("items") or []
        logger.info(
            "[SEO score] ranked_keywords parsed user_id=%s domain=%s task_status=%s items=%s",
            getattr(user, "id", None),
            domain,
            task.get("status_code"),
            len(items),
        )
        # Log a small preview of the first item for debugging (no PII or secrets).
        if items:
            preview = {
                "rank_absolute": items[0].get("rank_absolute"),
                "search_volume": (
                    ((items[0].get("keyword_data") or {}).get("keyword_info") or {}).get("search_volume")
                    or items[0].get("search_volume")
                ),
                "keyword": (items[0].get("keyword_data") or {}).get("keyword") or items[0].get("keyword"),
            }
            logger.info(
                "[SEO score] ranked_keywords first_item_preview user_id=%s domain=%s preview=%s",
                getattr(user, "id", None),
                domain,
                preview,
            )
    except Exception as exc:
        items = []
        logger.exception(
            "[SEO score] Error parsing ranked_keywords for user_id=%s domain=%s raw_preview=%s",
            getattr(user, "id", None),
            domain,
            str(ranked_data)[:200],
        )

    if not items:
        seo_score = compute_professional_seo_score(
            estimated_traffic=0.0,
            keywords_count=0,
            top3_positions=0,
            top10_positions=0,
            avg_keyword_difficulty=None,
            competitor_avg_traffic=0.0,
        )
        search_performance_score = seo_score
        onpage_snapshot = run_onpage_audit_for_user(user, site_url)
        onpage_score = onpage_snapshot.onpage_seo_score if onpage_snapshot else 0
        technical_score = onpage_snapshot.technical_seo_score if onpage_snapshot else 0
        overall = int(round(
            search_performance_score * 0.5
            + onpage_score * 0.3
            + technical_score * 0.2
        ))
        return {
            "seo_score": overall,
            "search_performance_score": search_performance_score,
            "onpage_seo_score": onpage_score,
            "technical_seo_score": technical_score,
            "organic_visitors": 0,
            "keywords_ranking": 0,
            "top3_positions": 0,
            "search_visibility_percent": 0,
            "missed_searches_monthly": 0,
        }

    # Derive metrics from ranked keyword items.
    estimated_traffic = 0.0
    keywords_ranking = len(items)
    top3_positions = 0
    top10_positions = 0
    difficulties: List[float] = []
    total_search_volume = 0.0
    total_traffic_share = 0.0
    missed_searches_volume = 0.0
    MIN_SEARCH_VOLUME = 10

    for item in items:
        rank = item.get("rank_absolute") or item.get("rank_group")
        try:
            rank_int = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank_int = None

        kw_info = (item.get("keyword_data") or {}).get("keyword_info") or {}
        search_volume = (
            kw_info.get("search_volume")
            or kw_info.get("search_volume_global")
            or kw_info.get("sum_search_volume")
            or item.get("search_volume")
            or item.get("sum_search_volume")
            or 0
        )
        try:
            sv = float(search_volume)
        except (TypeError, ValueError):
            sv = 0.0

        if sv > 0:
            total_search_volume += sv

        traffic_share_raw = item.get("traffic_share")
        try:
            traffic_share = float(traffic_share_raw) if traffic_share_raw is not None else 0.0
        except (TypeError, ValueError):
            traffic_share = 0.0
        if traffic_share > 0:
            total_traffic_share += traffic_share

        if rank_int is not None and rank_int > 0 and sv > 0:
            ctr = _ctr_for_position(rank_int)
            estimated_traffic += sv * ctr

            if rank_int <= 3:
                top3_positions += 1
            if rank_int <= 10:
                top10_positions += 1

        # Missed searches: keywords where we're effectively not capturing traffic yet:
        # - position is > 10, or no traffic_share, with sufficient volume.
        if (
            sv >= MIN_SEARCH_VOLUME
            and (rank_int is None or rank_int > 10 or traffic_share <= 0.0)
        ):
            missed_searches_volume += sv

        diff = _extract_keyword_difficulty(kw_info)
        if diff is not None:
            difficulties.append(diff)

    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else None

    # Search visibility and missed searches metrics
    if total_search_volume > 0 and total_traffic_share > 0:
        search_visibility_percent = int(round(
            max(0.0, min(100.0, (total_traffic_share / total_search_volume) * 100.0))
        ))
    else:
        search_visibility_percent = 0

    missed_searches_monthly = int(round(missed_searches_volume)) if missed_searches_volume > 0 else 0

    # Competitor baseline from competitors_domain/live
    competitor_avg_traffic = _get_competitor_average_traffic(
        domain,
        location_code=location_code,
        language_code=language_code,
    )

    try:
        snapshot, _ = SEOOverviewSnapshot.objects.get_or_create(
            user=user,
            period_start=start_current,
        )
        snapshot.organic_visitors = int(round(estimated_traffic))
        snapshot.keywords_ranking = keywords_ranking
        snapshot.top3_positions = top3_positions
        snapshot.save()
    except Exception:
        # If snapshot persistence fails, still return the live metrics.
        pass

    search_performance_score = compute_professional_seo_score(
        estimated_traffic=estimated_traffic,
        keywords_count=keywords_ranking,
        top3_positions=top3_positions,
        top10_positions=top10_positions,
        avg_keyword_difficulty=avg_difficulty,
        competitor_avg_traffic=competitor_avg_traffic,
    )

    # Attach On-Page / Technical SEO scores via OnPageAuditSnapshot
    onpage_snapshot = run_onpage_audit_for_user(user, site_url)
    onpage_score = onpage_snapshot.onpage_seo_score if onpage_snapshot else 0
    technical_score = onpage_snapshot.technical_seo_score if onpage_snapshot else 0
    pages_audited = onpage_snapshot.pages_audited if onpage_snapshot else 0
    onpage_issue_summaries = onpage_snapshot.issue_summaries if onpage_snapshot else {}

    overall = int(round(
        search_performance_score * 0.5
        + onpage_score * 0.3
        + technical_score * 0.2
    ))

    return {
        "seo_score": overall,
        "search_performance_score": search_performance_score,
        "onpage_seo_score": onpage_score,
        "technical_seo_score": technical_score,
        "pages_audited": pages_audited,
        "onpage_issue_summaries": onpage_issue_summaries,
        "organic_visitors": int(round(estimated_traffic)),
        "keywords_ranking": keywords_ranking,
        "top3_positions": top3_positions,
        "search_visibility_percent": search_visibility_percent,
        "missed_searches_monthly": missed_searches_monthly,
    }

