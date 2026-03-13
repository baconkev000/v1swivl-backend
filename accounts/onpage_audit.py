from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import requests
from django.conf import settings

from .models import OnPageAuditSnapshot

logger = logging.getLogger(__name__)

BASE_URL = "https://api.dataforseo.com"


def _get_auth() -> Optional[tuple[str, str]]:
    login = getattr(settings, "DATAFORSEO_LOGIN", None)
    password = getattr(settings, "DATAFORSEO_PASSWORD", None)
    if not login or not password:
        logger.warning(
            "[OnPageAudit] Missing DATAFORSEO_LOGIN/DATAFORSEO_PASSWORD; skipping API call.",
        )
        return None
    return str(login), str(password)


def _post(path: str, payload: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    auth = _get_auth()
    if auth is None:
        return None

    url = f"{BASE_URL}{path}"
    try:
        resp = requests.post(
            url,
            json=payload,
            auth=auth,
            timeout=45,
        )
    except Exception as exc:  # pragma: no cover - network failure
        logger.exception("[OnPageAudit] POST %s failed: %s", path, exc)
        return None

    if resp.status_code != 200:
        logger.warning(
            "[OnPageAudit] POST %s HTTP %s: %s",
            path,
            resp.status_code,
            resp.text[:500],
        )
        return None

    try:
        data = resp.json()
    except ValueError:
        logger.warning("[OnPageAudit] POST %s returned non-JSON body.", path)
        return None

    return data


def _normalize_domain(site_url: str) -> Optional[str]:
    if not site_url:
        return None
    parsed = urlparse(site_url)
    domain = (parsed.netloc or parsed.path or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or None


def _score_from_ratio(ok: int, total: int) -> int:
    if total <= 0:
        return 0
    ratio = max(0.0, min(1.0, ok / total))
    return int(round(ratio * 100))


def run_onpage_audit_for_user(user, site_url: str | None) -> Optional[OnPageAuditSnapshot]:
    """
    Run (or reuse) a full On-Page / Technical SEO audit for the user + domain.

    - Uses DataForSEO On-Page API:
      - /v3/on_page/task_post to enqueue a crawl
      - /v3/on_page/pages to fetch crawled pages
    - Aggregates page-level findings into site-wide metrics & scores.
    - Caches results in OnPageAuditSnapshot, updated at most once every 24 hours.
    """
    from django.utils import timezone

    if not site_url:
        logger.info("[OnPageAudit] No site_url configured for user_id=%s", getattr(user, "id", None))
        return None

    domain = _normalize_domain(site_url)
    if not domain:
        logger.info("[OnPageAudit] Could not normalise domain from site_url=%s user_id=%s", site_url, getattr(user, "id", None))
        return None

    now = timezone.now()
    cutoff = now - timedelta(days=1)

    snapshot, created = OnPageAuditSnapshot.objects.get_or_create(
        user=user,
        domain=domain,
        defaults={},
    )

    if not created and snapshot.last_fetched_at and snapshot.last_fetched_at >= cutoff:
        logger.info(
            "[OnPageAudit] Using cached snapshot for user_id=%s domain=%s last_fetched_at=%s",
            getattr(user, "id", None),
            domain,
            snapshot.last_fetched_at,
        )
        return snapshot

    logger.info("[OnPageAudit] Starting new crawl for user_id=%s domain=%s", getattr(user, "id", None), domain)

    # 1) Create crawl task
    task_payload = [
        {
            "target": domain,
            "max_crawl_pages": 500,
            "load_resources": True,
            "enable_javascript": True,
        },
    ]
    task_data = _post("/v3/on_page/task_post", task_payload)
    if not task_data:
        logger.warning("[OnPageAudit] task_post returned no data for domain=%s", domain)
        return snapshot

    try:
        task_id = task_data["tasks"][0]["id"]
    except Exception:
        logger.warning("[OnPageAudit] Could not extract task_id from task_post response for domain=%s", domain)
        return snapshot

    # 2) Fetch important pages only (cost-effective audit):
    #    Use DataForSEO's ordering to fetch top pages by page_rank and limit to 20.
    IMPORTANT_PAGE_LIMIT = 20
    pages_payload = [
        {
            "id": task_id,
            "limit": IMPORTANT_PAGE_LIMIT,
            "offset": 0,
            "order_by": ["page_rank,desc"],
        },
    ]
    pages_data = _post("/v3/on_page/pages", pages_payload)
    if not pages_data:
        logger.warning("[OnPageAudit] pages returned no data for task_id=%s domain=%s", task_id, domain)
        return snapshot

    try:
        tasks = pages_data.get("tasks") or []
        if not tasks:
            logger.warning(
                "[OnPageAudit] pages: no tasks in response for task_id=%s domain=%s raw=%s",
                task_id,
                domain,
                str(pages_data)[:300],
            )
            return snapshot

        results = tasks[0].get("result") or []
        if not results:
            logger.warning(
                "[OnPageAudit] pages: no result items for task_id=%s domain=%s raw_task=%s",
                task_id,
                domain,
                str(tasks[0])[:300],
            )
            return snapshot

        pages = results[0].get("items") or []
    except Exception as exc:
        logger.exception(
            "[OnPageAudit] Failed to parse pages for task_id=%s domain=%s: %s (raw=%s)",
            task_id,
            domain,
            exc,
            str(pages_data)[:300],
        )
        return snapshot

    # Only audit the limited set of "important" pages.
    total_pages = len(pages)
    pages_missing_titles = 0
    pages_missing_descriptions = 0
    pages_bad_h1 = 0
    images_missing_alt = 0
    broken_internal_links = 0
    error_pages_4xx_5xx = 0
    pages_missing_canonical = 0
    canonical_targets: Dict[str, int] = {}
    has_robots_txt = False
    has_sitemap_xml = False

    for page in pages:
        status_code = page.get("status_code") or page.get("http_status_code")
        try:
            sc = int(status_code) if status_code is not None else 200
        except (TypeError, ValueError):
            sc = 200

        if 400 <= sc < 600:
            error_pages_4xx_5xx += 1

        meta = page.get("meta", {}) or {}
        title = meta.get("title")
        description = meta.get("description")
        if not title:
            pages_missing_titles += 1
        if not description:
            pages_missing_descriptions += 1

        # Headings
        headings = page.get("page_timing", {}).get("headings") or page.get("content", {}).get("headings") or []
        h1_count = 0
        for h in headings:
            tag = (h.get("tag") or "").lower()
            if tag == "h1":
                h1_count += 1
        if h1_count != 1:
            pages_bad_h1 += 1

        # Images & alt
        images = page.get("images") or page.get("content", {}).get("images") or []
        for img in images:
            alt = img.get("alt") or ""
            if not alt.strip():
                images_missing_alt += 1

        # Internal links
        links = page.get("page_summary", {}).get("links") or page.get("links") or []
        for link in links:
            if not link.get("is_internal"):
                continue
            l_status = link.get("status_code")
            try:
                ls = int(l_status) if l_status is not None else 200
            except (TypeError, ValueError):
                ls = 200
            if 400 <= ls < 600:
                broken_internal_links += 1

        # Canonical + indexing
        canonical = page.get("canonical") or meta.get("canonical_url") or meta.get("canonical")
        if not canonical:
            pages_missing_canonical += 1
        else:
            canonical_targets[canonical] = canonical_targets.get(canonical, 0) + 1

        # Robots / indexing checks
        robots = meta.get("robots") or page.get("robots") or ""
        robots_str = ",".join(robots) if isinstance(robots, list) else str(robots)
        # We don't penalise here; indexability score will account for global state.

    # Duplicate canonical targets
    duplicate_canonical_targets = sum(1 for count in canonical_targets.values() if count > 1)

    # robots.txt and sitemap.xml presence — simple HTTP checks
    scheme = "https"
    robots_url = f"{scheme}://{domain}/robots.txt"
    sitemap_url = f"{scheme}://{domain}/sitemap.xml"
    try:
        r = requests.get(robots_url, timeout=5)
        has_robots_txt = r.status_code == 200
    except Exception:
        has_robots_txt = False
    try:
        s = requests.get(sitemap_url, timeout=5)
        has_sitemap_xml = s.status_code == 200
    except Exception:
        has_sitemap_xml = False

    # Scores
    metadata_ok = total_pages - max(pages_missing_titles, pages_missing_descriptions)
    metadata_score = _score_from_ratio(metadata_ok, max(total_pages, 1))

    content_structure_ok = total_pages - pages_bad_h1
    content_structure_score = _score_from_ratio(content_structure_ok, max(total_pages, 1))

    # Accessibility: treat images_missing_alt proportionally
    # If no images, score 100; otherwise penalise missing alts.
    total_images = images_missing_alt  # we don't know total precisely, approximate
    if total_images <= 0:
        accessibility_score = 100
    else:
        # Assume missing alt ratio approximates total alt health
        missing_ratio = 1.0  # worst-case; logging-only metric
        accessibility_score = max(0, int(round(100 * (1.0 - missing_ratio))))

    # Internal link score: penalise broken links & error pages
    internal_issue_count = broken_internal_links + error_pages_4xx_5xx
    if internal_issue_count == 0:
        internal_link_score = 100
    else:
        internal_link_score = max(0, 100 - min(60, internal_issue_count * 5))

    # Indexability score: canonical completeness + robots/sitemap presence
    indexable_ok = total_pages - pages_missing_canonical
    indexability_score = _score_from_ratio(indexable_ok, max(total_pages, 1))
    if not has_robots_txt:
        indexability_score = max(0, indexability_score - 10)
    if not has_sitemap_xml:
        indexability_score = max(0, indexability_score - 10)
    if duplicate_canonical_targets > 0:
        indexability_score = max(0, indexability_score - min(20, duplicate_canonical_targets * 2))

    # Composite scores
    onpage_seo_score = int(
        round(
            metadata_score * 0.4
            + content_structure_score * 0.3
            + accessibility_score * 0.3
        )
    )
    technical_seo_score = int(
        round(
            internal_link_score * 0.5
            + indexability_score * 0.5
        )
    )

    # Issue summaries for frontend
    issue_summaries: Dict[str, str] = {}
    issue_summaries["metadata"] = f"{pages_missing_descriptions} pages missing meta descriptions; {pages_missing_titles} pages missing titles."
    issue_summaries["headings"] = f"{pages_bad_h1} pages do not have exactly one H1 heading."
    issue_summaries["accessibility"] = f"{images_missing_alt} images are missing alt text."
    issue_summaries["links"] = f"{broken_internal_links} broken internal links and {error_pages_4xx_5xx} error pages detected."
    issue_summaries["indexability"] = (
        f"{pages_missing_canonical} pages missing canonical tags; "
        f"{duplicate_canonical_targets} duplicate canonical targets; "
        f"robots.txt present: {has_robots_txt}; sitemap.xml present: {has_sitemap_xml}."
    )

    snapshot.pages_missing_titles = pages_missing_titles
    snapshot.pages_missing_descriptions = pages_missing_descriptions
    snapshot.pages_bad_h1 = pages_bad_h1
    snapshot.images_missing_alt = images_missing_alt
    snapshot.broken_internal_links = broken_internal_links
    snapshot.error_pages_4xx_5xx = error_pages_4xx_5xx
    snapshot.pages_missing_canonical = pages_missing_canonical
    snapshot.duplicate_canonical_targets = duplicate_canonical_targets
    snapshot.has_robots_txt = has_robots_txt
    snapshot.has_sitemap_xml = has_sitemap_xml
    snapshot.metadata_score = metadata_score
    snapshot.content_structure_score = content_structure_score
    snapshot.accessibility_score = accessibility_score
    snapshot.internal_link_score = internal_link_score
    snapshot.indexability_score = indexability_score
    snapshot.onpage_seo_score = onpage_seo_score
    snapshot.technical_seo_score = technical_seo_score
    snapshot.issue_summaries = issue_summaries
    snapshot.pages_audited = total_pages
    snapshot.save()

    logger.info(
        "[OnPageAudit] Completed audit for user_id=%s domain=%s onpage_score=%s technical_score=%s",
        getattr(user, "id", None),
        domain,
        onpage_seo_score,
        technical_seo_score,
    )

    return snapshot

