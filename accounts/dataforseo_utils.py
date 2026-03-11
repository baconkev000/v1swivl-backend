from __future__ import annotations

"""
Helpers for calling DataForSEO Labs APIs for the SEO Agent.

Primary endpoints used:
- POST /v3/dataforseo_labs/google/ranked_keywords/live      → current visibility
- POST /v3/dataforseo_labs/google/domain_intersection/live  → keyword gap vs competitors
"""

from typing import Any, Dict, List, Optional

import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.dataforseo.com"


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
        return None

    try:
        data = resp.json()
    except ValueError:  # pragma: no cover - unexpected non-JSON
        logger.warning("[DataForSEO] POST %s returned non-JSON body.", path)
        return None

    # DataForSEO wraps results in tasks / result; callers will unpack further.
    return data


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

