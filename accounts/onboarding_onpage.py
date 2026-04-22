"""
Extract structured fields from DataForSEO On-Page API page items for onboarding.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from django.conf import settings

from .constants import ONBOARDING_RANKED_KEYWORDS_LIMIT
from accounts.third_party_usage import usage_profile_context

from .dataforseo_utils import crawl_pages_for_onboarding, fetch_ranked_keyword_items
from .models import OnboardingOnPageCrawl
from .onboarding_keyword_filter import apply_aeo_keyword_pipeline, ranked_rows_as_labs_api_shape
from .onboarding_topic_clusters import build_topic_clusters, compact_ranked_for_storage

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "your", "from", "this", "that", "home", "page",
    "our", "are", "was", "has", "have", "will", "can", "you", "not", "all", "any",
})


def _heading_rows(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = page.get("content") if isinstance(page.get("content"), dict) else {}
    timing = page.get("page_timing") if isinstance(page.get("page_timing"), dict) else {}
    raw = (
        page.get("headings")
        or content.get("headings")
        or timing.get("headings")
        or []
    )
    return [h for h in raw if isinstance(h, dict)]


def _split_h1_h2h3(rows: List[Dict[str, Any]]) -> tuple[str, List[str]]:
    h1_text = ""
    h2h3: List[str] = []
    for h in rows:
        tag = (h.get("tag") or "").lower()
        text = (h.get("text") or h.get("value") or "").strip()
        if not text:
            continue
        if tag == "h1" and not h1_text:
            h1_text = text
        elif tag in ("h2", "h3"):
            h2h3.append(f"{tag.upper()}: {text}")
    return h1_text, h2h3[:20]


def _collect_schema_types(value: Any, out: Optional[set[str]] = None) -> List[str]:
    if out is None:
        out = set()
    if value is None:
        return sorted(out)
    if isinstance(value, dict):
        t = value.get("@type") or value.get("type")
        if isinstance(t, str) and t.strip():
            out.add(t.strip())
        elif isinstance(t, list):
            for x in t:
                if isinstance(x, str) and x.strip():
                    out.add(x.strip())
        for v in value.values():
            _collect_schema_types(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_schema_types(v, out)
    elif isinstance(value, str):
        s = value.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                _collect_schema_types(json.loads(s), out)
            except (json.JSONDecodeError, TypeError):
                pass
    return sorted(out)


def _structured_data_roots(page: Dict[str, Any]) -> List[Any]:
    meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
    roots: List[Any] = []
    for key in (
        "structured_data",
        "json_ld",
        "microdata",
        "schema",
        "rich_snippet",
    ):
        v = page.get(key) or meta.get(key)
        if v is not None:
            roots.append(v)
    # Some payloads nest under checks
    checks = page.get("checks")
    if isinstance(checks, dict):
        for v in checks.values():
            if isinstance(v, (dict, list, str)):
                roots.append(v)
    return roots


def _faq_questions_from_schema(value: Any, out: Optional[List[str]] = None) -> List[str]:
    if out is None:
        out = []
    if value is None:
        return out
    if isinstance(value, dict):
        types = value.get("@type") or value.get("type")
        type_str = ""
        if isinstance(types, str):
            type_str = types.lower()
        elif isinstance(types, list):
            type_str = " ".join(str(x).lower() for x in types)
        if "question" in type_str:
            name = value.get("name")
            qtext = ""
            if isinstance(name, str):
                qtext = name.strip()
            elif isinstance(name, dict):
                qtext = str(name.get("text") or name.get("@value") or "").strip()
            if qtext and qtext not in out:
                out.append(qtext)
        for v in value.values():
            _faq_questions_from_schema(v, out)
    elif isinstance(value, list):
        for v in value:
            _faq_questions_from_schema(v, out)
    elif isinstance(value, str):
        s = value.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                _faq_questions_from_schema(json.loads(s), out)
            except (json.JSONDecodeError, TypeError):
                pass
    return out[:30]


def _page_plain_text(page: Dict[str, Any]) -> str:
    from .dataforseo_utils import _extract_text_fragments_for_question_coverage

    parts = _extract_text_fragments_for_question_coverage(page)
    return " ".join(parts)[:80000]


def _first_meaningful_paragraph(page: Dict[str, Any]) -> str:
    from .dataforseo_utils import _extract_candidate_answer_paragraphs

    paras = _extract_candidate_answer_paragraphs(page)
    if paras:
        return paras[0][:4000]
    text = _page_plain_text(page)
    chunks = re.split(r"\n\s*\n+", text)
    for ch in chunks:
        ch = ch.strip()
        words = re.findall(r"\b[\w'-]+\b", ch)
        if len(words) >= 25:
            return ch[:4000]
    return (text[:4000] if text else "").strip()


def _primary_keyword_candidates(page: Dict[str, Any], title: str, h1: str) -> List[str]:
    out: List[str] = []
    meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
    raw_kw = meta.get("keywords") or meta.get("keyword")
    if isinstance(raw_kw, str):
        out.extend(x.strip() for x in raw_kw.split(",") if x.strip())
    for key in ("focus_keyword", "primary_keyword", "news_keywords"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    for source in (title, h1):
        if not source:
            continue
        for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", source):
            lw = w.lower()
            if lw not in _STOPWORDS and len(lw) > 2:
                out.append(w)
    seen: set[str] = set()
    deduped: List[str] = []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(x)
    return deduped[:15]


def _service_location_mentions(
    page_text: str,
    business_name: str,
    location: str,
) -> Dict[str, Any]:
    text_l = page_text.lower()
    biz = (business_name or "").strip()
    loc_raw = (location or "").strip()
    result: Dict[str, Any] = {
        "business_name_found": False,
        "location_mentions": [],
    }
    if biz and len(biz) > 2 and biz.lower() in text_l:
        result["business_name_found"] = True
    if loc_raw:
        for token in re.split(r"[,;]|/", loc_raw):
            t = token.strip()
            if len(t) > 2 and t.lower() in text_l:
                result["location_mentions"].append(t)
    return result


def extract_onboarding_page_record(
    page: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Map one DataForSEO page item to the fields we persist for onboarding.
    """
    ctx = context or {}
    business_name = str(ctx.get("business_name") or "")
    location = str(ctx.get("location") or "")

    url = str(page.get("url") or page.get("full_url") or "").strip()
    parsed = urlparse(url)
    url_path = parsed.path or "/"

    meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
    title = str(meta.get("title") or page.get("title") or "").strip()
    meta_description = str(meta.get("description") or meta.get("meta_description") or "").strip()

    rows = _heading_rows(page)
    h1, h2h3 = _split_h1_h2h3(rows)

    schema_types: List[str] = []
    for root in _structured_data_roots(page):
        schema_types.extend(_collect_schema_types(root))
    schema_types = sorted(set(schema_types))

    faq_questions = []
    for root in _structured_data_roots(page):
        faq_questions.extend(_faq_questions_from_schema(root))
    faq_questions = list(dict.fromkeys(faq_questions))[:25]

    body_sample = _first_meaningful_paragraph(page)
    page_text = _page_plain_text(page)
    mentions = _service_location_mentions(page_text, business_name, location)
    keywords = _primary_keyword_candidates(page, title, h1)

    return {
        "url": url,
        "url_path": url_path,
        "page_title": title,
        "meta_description": meta_description,
        "h1": h1,
        "h2_h3_headings": h2h3,
        "primary_keyword_candidates": keywords,
        "schema_types": schema_types,
        "first_meaningful_body_paragraph": body_sample,
        "faq_questions": faq_questions,
        "service_location_mentions": mentions,
    }


def _persist_crawl_ranked_and_clusters(
    crawl: OnboardingOnPageCrawl,
    extracted: List[Dict[str, Any]],
    ranked_items: List[Dict[str, Any]],
) -> None:
    """Build clusters from extracted pages + ranked items; write ranked + cluster fields on crawl."""
    context = crawl.context if isinstance(crawl.context, dict) else {}
    bundle = build_topic_clusters(extracted, ranked_items)
    filtered_ranked = apply_aeo_keyword_pipeline(
        bundle["ranked_keywords_normalized"],
        context=context,
        seeds=bundle["crawl_topic_seeds"],
    )
    logger.info(
        "[onboarding onpage] aeo_keyword_filter crawl_id=%s raw=%s filtered=%s",
        crawl.id,
        len(bundle["ranked_keywords_normalized"]),
        len(filtered_ranked),
    )
    rebuild_items = ranked_rows_as_labs_api_shape(filtered_ranked)
    bundle_filtered = build_topic_clusters(extracted, rebuild_items)
    crawl.crawl_topic_seeds = bundle_filtered["crawl_topic_seeds"]
    crawl.topic_clusters = bundle_filtered["topic_clusters"]
    crawl.ranked_keywords = compact_ranked_for_storage(
        filtered_ranked,
        cap=ONBOARDING_RANKED_KEYWORDS_LIMIT,
    )


def execute_onboarding_ranked_keywords_fetch(crawl_id: int) -> None:
    """
    DataForSEO Labs ranked_keywords + clustering (Celery). Runs after review_topics in the main crawl.
    """
    crawl = (
        OnboardingOnPageCrawl.objects.select_related("business_profile", "user")
        .filter(pk=crawl_id)
        .first()
    )
    if not crawl:
        logger.warning("[onboarding ranked fetch] crawl id=%s not found", crawl_id)
        return
    if crawl.ranked_keywords_fetch_status != OnboardingOnPageCrawl.RANKED_FETCH_PENDING:
        return
    if crawl.status != OnboardingOnPageCrawl.STATUS_COMPLETED:
        return
    extracted = crawl.pages if isinstance(crawl.pages, list) else []
    if not extracted:
        crawl.ranked_keywords_fetch_status = OnboardingOnPageCrawl.RANKED_FETCH_COMPLETE
        crawl.save(update_fields=["ranked_keywords_fetch_status", "updated_at"])
        return

    context = crawl.context if isinstance(crawl.context, dict) else {}
    location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    language_code = str(getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en") or "en")
    ranked_items: List[Dict[str, Any]] = []
    crawl.ranked_keywords_error = ""
    try:
        with usage_profile_context(crawl.business_profile):
            try:
                ranked_items = fetch_ranked_keyword_items(
                    crawl.domain,
                    location_code,
                    language_code,
                    user=crawl.user,
                    limit=ONBOARDING_RANKED_KEYWORDS_LIMIT,
                    business_profile=crawl.business_profile,
                )
            except Exception as exc:
                logger.exception("[onboarding ranked fetch] ranked_keywords fetch crawl_id=%s", crawl_id)
                crawl.ranked_keywords_error = str(exc)[:2000]
        _persist_crawl_ranked_and_clusters(crawl, extracted, ranked_items)
    except Exception as exc:
        logger.exception("[onboarding ranked fetch] crawl id=%s failed", crawl_id)
        crawl.ranked_keywords_error = (crawl.ranked_keywords_error or str(exc))[:2000]
    crawl.ranked_keywords_fetch_status = OnboardingOnPageCrawl.RANKED_FETCH_COMPLETE
    crawl.save(
        update_fields=[
            "crawl_topic_seeds",
            "topic_clusters",
            "ranked_keywords",
            "ranked_keywords_error",
            "ranked_keywords_fetch_status",
            "updated_at",
        ],
    )
    logger.info(
        "[onboarding ranked fetch] crawl id=%s done ranked_count=%s",
        crawl_id,
        len(crawl.ranked_keywords or []),
    )


def execute_onboarding_onpage_crawl(crawl_id: int) -> None:
    """Run crawl + extraction, review topics, then enqueue Labs ranked_keywords in the background."""
    crawl = OnboardingOnPageCrawl.objects.filter(pk=crawl_id).first()
    if not crawl:
        logger.warning("[onboarding onpage] crawl id=%s not found", crawl_id)
        return

    crawl.status = OnboardingOnPageCrawl.STATUS_RUNNING
    crawl.save(update_fields=["status", "updated_at"])

    try:
        with usage_profile_context(crawl.business_profile):
            result = crawl_pages_for_onboarding(
                target_domain=crawl.domain,
                max_pages=min(crawl.max_pages, 10),
                timeout_seconds=150,
                business_profile=crawl.business_profile,
            )
            raw_pages: List[Dict[str, Any]] = result.get("pages") or []
            context = crawl.context if isinstance(crawl.context, dict) else {}
            extracted = [
                extract_onboarding_page_record(p, context) for p in raw_pages[: crawl.max_pages]
            ]
            crawl.pages = extracted
            crawl.task_id = str(result.get("task_id") or "")[:128]
            crawl.exit_reason = str(result.get("exit_reason") or "")[:64]
            if extracted:
                crawl.status = OnboardingOnPageCrawl.STATUS_COMPLETED
                crawl.error_message = ""
            else:
                crawl.status = OnboardingOnPageCrawl.STATUS_FAILED
                crawl.error_message = (
                    result.get("exit_reason")
                    or result.get("crawl_status")
                    or "no_pages"
                )[:2000]

        if not extracted:
            crawl.ranked_keywords_fetch_status = OnboardingOnPageCrawl.RANKED_FETCH_LEGACY
            crawl.save(
                update_fields=[
                    "pages",
                    "task_id",
                    "exit_reason",
                    "status",
                    "error_message",
                    "ranked_keywords_fetch_status",
                    "updated_at",
                ],
            )
            return

        # Crawl-only seeds/clusters (no Labs yet); ranked_keywords filled by background task.
        crawl.ranked_keywords_error = ""
        _persist_crawl_ranked_and_clusters(crawl, extracted, [])
        crawl.ranked_keywords_fetch_status = OnboardingOnPageCrawl.RANKED_FETCH_PENDING

        save_fields = [
            "pages",
            "task_id",
            "exit_reason",
            "status",
            "error_message",
            "crawl_topic_seeds",
            "topic_clusters",
            "ranked_keywords",
            "ranked_keywords_error",
            "ranked_keywords_fetch_status",
            "updated_at",
        ]
        crawl.save(update_fields=save_fields)

        from .onboarding_review_topics import generate_review_topics_for_domain

        rt_list, rt_err = generate_review_topics_for_domain(
            domain=crawl.domain,
            business_profile=crawl.business_profile,
        )
        crawl.review_topics = rt_list
        crawl.review_topics_error = (rt_err or "")[:2000]
        crawl.save(update_fields=["review_topics", "review_topics_error", "updated_at"])

        from .tasks import onboarding_ranked_keywords_fetch_task

        try:
            onboarding_ranked_keywords_fetch_task.delay(crawl.id)
        except Exception:
            logger.exception(
                "[onboarding onpage] ranked_keywords fetch enqueue failed crawl_id=%s",
                crawl.id,
            )

        logger.info(
            "[onboarding onpage] crawl id=%s status=%s pages=%s ranked_fetch=enqueued",
            crawl_id,
            crawl.status,
            len(extracted),
        )
    except Exception as exc:
        logger.exception("[onboarding onpage] crawl id=%s failed", crawl_id)
        crawl.status = OnboardingOnPageCrawl.STATUS_FAILED
        crawl.error_message = str(exc)[:2000]
        crawl.save(update_fields=["status", "error_message", "updated_at"])
