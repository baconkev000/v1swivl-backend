"""
Phase 5: actionable AEO recommendations from score + extraction snapshots.

Does not implement notifications, dashboards, or content automation.

Note: `accounts.openai_utils.generate_aeo_recommendations` is unrelated (legacy string-list helper for serializers).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from django.conf import settings

from ..models import (
    AEOExtractionSnapshot,
    AEORecommendationRun,
    AEOScoreSnapshot,
    BusinessProfile,
    OnboardingOnPageCrawl,
)
from ..openai_utils import _get_client, _get_model, chat_completion_create_logged
from .aeo_extraction_utils import brand_effectively_cited, canonical_registrable_domain
from .aeo_prompts import (
    AEO_RECOMMENDATION_NL_SYSTEM_PROMPT,
    AEO_RECOMMENDATION_TYPE_SYSTEM_PROMPT,
)
from .aeo_scoring_utils import (
    calculate_citation_share,
    calculate_visibility_score,
    calculate_weighted_position_score,
    latest_extraction_per_response,
)
from .aeo_utils import infer_city_from_address

logger = logging.getLogger(__name__)

DEFAULT_API_KEY_ENV = "OPEN_AI_SEO_API_KEY"

RECOMMENDATION_TYPE_CHOICES: frozenset[str] = frozenset(
    {
        "new_page",
        "faq_expansion",
        "schema_fix",
        "citation_target",
        "entity_alignment",
    }
)

# Country / macro-region strings that are too vague alone for AEO locality copy
_VAGUE_LOCALITY_ONLY: frozenset[str] = frozenset(
    {
        "united states",
        "usa",
        "us",
        "u.s.",
        "u.s.a.",
        "united kingdom",
        "uk",
        "great britain",
        "canada",
        "australia",
        "new zealand",
        "europe",
        "worldwide",
        "global",
    }
)


def _short_website_domain(profile: BusinessProfile) -> str:
    raw = (getattr(profile, "website_url", None) or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        host = (urlparse(raw).netloc or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _competitor_display_names(competitors: list[Any] | None, *, limit: int = 16) -> list[str]:
    """
    Turn extraction competitor entries into short display strings (name or url), never ``str(dict)``.
    """
    out: list[str] = []
    if not competitors or not isinstance(competitors, list):
        return out
    seen: set[str] = set()
    for c in competitors:
        if len(out) >= limit:
            break
        label = ""
        if isinstance(c, dict):
            name = str(c.get("name") or c.get("competitor") or "").strip()
            url = str(c.get("url") or c.get("domain") or "").strip()
            label = name or url
        elif c is not None:
            s = str(c).strip()
            if s.startswith("{") and "}" in s:
                continue
            label = s
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(label[:120])
    return out


def _region_label_for_profile(profile: BusinessProfile, city: str) -> str:
    """
    Human-readable locality for copy. Avoid using bare country names (e.g. only ``United States``)
    when we have no city—prefer industry or domain-scoped phrasing.
    """
    c = (city or "").strip()
    if c and c.lower() in _VAGUE_LOCALITY_ONLY:
        c = ""
    if c:
        return c
    addr = (profile.business_address or "").strip()
    if addr:
        parts = [p.strip() for p in addr.split(",") if p.strip()]
        if len(parts) == 1 and parts[0].lower() in _VAGUE_LOCALITY_ONLY:
            pass
        else:
            alt = infer_city_from_address(addr)
            if alt and alt.lower() not in _VAGUE_LOCALITY_ONLY:
                return alt
    industry = (profile.industry or "").strip()
    if industry:
        return f"the {industry} market"
    dom = _short_website_domain(profile)
    if dom:
        return f"searchers evaluating offerings like yours ({dom})"
    return "your market"


def _latest_completed_onboarding_crawl(profile: BusinessProfile) -> OnboardingOnPageCrawl | None:
    return (
        OnboardingOnPageCrawl.objects.filter(
            business_profile=profile,
            status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        )
        .order_by("-created_at", "-id")
        .first()
    )


def _build_onpage_crawl_summary(crawl: OnboardingOnPageCrawl | None) -> str:
    """
    Compact text for NL enrichment: topic seeds plus first few crawled pages (title, h1, meta, schema).
    """
    if crawl is None:
        return ""
    parts: list[str] = []
    seeds = crawl.crawl_topic_seeds or []
    if isinstance(seeds, list) and seeds:
        labels: list[str] = []
        for s in seeds[:10]:
            if isinstance(s, dict):
                lab = str(s.get("label") or "").strip()
                if lab:
                    labels.append(lab[:100])
            elif isinstance(s, str) and s.strip():
                labels.append(s.strip()[:100])
        if labels:
            dedup = list(dict.fromkeys(labels))[:8]
            parts.append("Topic seeds: " + "; ".join(dedup))
    pages = crawl.pages if isinstance(crawl.pages, list) else []
    for p in pages[:4]:
        if not isinstance(p, dict):
            continue
        title = str(p.get("page_title") or "").strip()[:80]
        h1 = str(p.get("h1") or "").strip()[:80]
        md = str(p.get("meta_description") or "").strip()
        md = re.sub(r"\s+", " ", md)[:140].rstrip()
        st = p.get("schema_types") or []
        st_str = ""
        if isinstance(st, list):
            st_str = ", ".join(str(x) for x in st[:10] if str(x).strip())
        bits: list[str] = []
        if title:
            bits.append(f"title «{title}»")
        if h1 and h1.lower() != title.lower():
            bits.append(f"h1 «{h1}»")
        if md:
            bits.append(f"meta «{md}»")
        if st_str:
            bits.append(f"schema [{st_str}]")
        if not bits:
            continue
        url = str(p.get("url") or "")[:72]
        prefix = f"{url}: " if url else ""
        parts.append(prefix + " | ".join(bits))
    return "\n".join(parts)[:8000]


def _priority_from_scores(visibility: float, citation_share: float) -> str:
    """
    Global tier from Phase 4-style percentages (transparent thresholds).

    High: Visibility < 20% OR Citation Share < 15%
    Medium: Visibility < 40% OR Citation Share < 30%
    Low: otherwise
    """
    if visibility < 20.0 or citation_share < 15.0:
        return "high"
    if visibility < 40.0 or citation_share < 30.0:
        return "medium"
    return "low"


def _first_extraction_ids_for_domain(
    extraction_snapshots: list[AEOExtractionSnapshot],
    domain: str,
) -> tuple[int | None, int | None]:
    d = (domain or "").strip().lower()
    if not d:
        return None, None
    for ex in extraction_snapshots:
        cites = ex.citations_json or []
        if not isinstance(cites, list):
            continue
        for raw in cites:
            if str(raw).strip().lower() == d:
                return ex.id, ex.response_snapshot_id
    return None, None


def _top_competitors_from_score(score: AEOScoreSnapshot | None, limit: int = 3) -> list[str]:
    if not score:
        return []
    dom = score.competitor_dominance_json or {}
    ranked = dom.get("ranked") or []
    out: list[str] = []
    if isinstance(ranked, list):
        for row in ranked[:limit]:
            if isinstance(row, dict):
                name = str(row.get("competitor") or "").strip()
                if name:
                    out.append(name)
    return out


def _url_identity_fields_from_extraction(
    ex: AEOExtractionSnapshot,
    *,
    canonical_domain: str,
) -> dict[str, Any]:
    """
    Per-extraction web-identity context for Phase 5 (wrong-domain live vs broken vs not flagged).

    ``canonical_domain`` should already be normalized (``canonical_registrable_domain``).
    """
    canon = (canonical_domain or "").strip().lower().rstrip(".")
    raw_status = getattr(ex, "brand_mentioned_url_status", None)
    status_str = str(raw_status).strip() if raw_status else ""
    if not status_str:
        status_str = "not_mentioned"

    cited_raw = (getattr(ex, "cited_domain_or_url", None) or "").strip()
    cited_list = [cited_raw] if cited_raw else []

    notes = getattr(ex, "url_verification_notes", None) or {}
    if not isinstance(notes, dict):
        notes = {}

    verify_parts: list[str] = []
    if notes.get("verification_disabled"):
        verify_parts.append("probe_skipped")
    elif notes:
        if "dns_ok" in notes:
            verify_parts.append("dns_" + ("ok" if notes.get("dns_ok") else "fail"))
        if "http_ok" in notes:
            verify_parts.append("http_" + ("ok" if notes.get("http_ok") else "fail"))
        sc = notes.get("status_code")
        if sc is not None:
            verify_parts.append(f"status_{sc}")
    verification_summary = ";".join(verify_parts)

    summary = ""
    if status_str == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE:
        other = cited_raw or "another domain"
        canon_disp = canon or "(no canonical domain on file)"
        summary = (
            f"Modeled answers tied your brand to {other} while your registered site is {canon_disp}; {other} resolves "
            f"over the network (likely a different entity—disambiguate in content and schema)."
        )
    elif status_str == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN:
        other = cited_raw or "a URL"
        summary = (
            f"Modeled answers tied your brand to {other}; that destination does not resolve reliably—treat it as a bad "
            f"citation and reinforce your official URL in schema, footer, and major listings."
        )

    return {
        "brand_mentioned_url_status": status_str,
        "canonical_domain": canon,
        "cited_domain_in_answer": cited_list,
        "url_identity_summary": summary,
        "verification_summary": verification_summary,
    }


def analyze_visibility_gaps(
    score_snapshot: AEOScoreSnapshot | None,
    extraction_snapshots: list[AEOExtractionSnapshot],
    *,
    tracked_website_url: str = "",
    canonical_domain: str = "",
) -> list[dict[str, Any]]:
    """
    Gaps where the target brand did not appear in the modeled answer for a prompt.

    Each gap carries ids for traceability to extraction / response rows, plus URL/identity context from the
    paired ``AEOExtractionSnapshot`` when available.
    """
    canon = (canonical_domain or "").strip().lower().rstrip(".") or canonical_registrable_domain(
        tracked_website_url or ""
    )
    gaps: list[dict[str, Any]] = []
    for ex in extraction_snapshots:
        if brand_effectively_cited(
            bool(ex.brand_mentioned),
            ex.competitors_json,
            tracked_website_url_or_domain=tracked_website_url,
        ):
            continue
        resp = ex.response_snapshot
        row: dict[str, Any] = {
            "gap_kind": "visibility_miss",
            "prompt_text": (resp.prompt_text or "").strip(),
            "extraction_snapshot_id": ex.id,
            "response_snapshot_id": resp.id,
            "competitors_in_answer": list(ex.competitors_json or [])
            if isinstance(ex.competitors_json, list)
            else [],
        }
        row.update(_url_identity_fields_from_extraction(ex, canonical_domain=canon))
        gaps.append(row)
    return gaps


def analyze_citation_gaps(
    score_snapshot: AEOScoreSnapshot | None,
    extraction_snapshots: list[AEOExtractionSnapshot],
    *,
    citation_share: float,
) -> list[dict[str, Any]]:
    """
    Gaps when citation-style share of mention-units is weak; suggests domains to prioritize.

    Uses aggregated citation domains from extractions plus competitor context from the score snapshot.
    """
    gaps: list[dict[str, Any]] = []
    cite = float(citation_share)
    if cite >= 30.0:
        return gaps

    dom_counter: Counter[str] = Counter()
    for ex in extraction_snapshots:
        cites = ex.citations_json or []
        if not isinstance(cites, list):
            continue
        for raw in cites:
            d = str(raw).strip().lower()
            if d and "." in d:
                dom_counter[d] += 1

    top_domains = [h for h, _ in dom_counter.most_common(5)]
    top_comp = _top_competitors_from_score(score_snapshot, limit=2)
    comp_label = top_comp[0] if top_comp else "competitors"

    for domain in top_domains[:3]:
        ex_id, resp_id = _first_extraction_ids_for_domain(extraction_snapshots, domain)
        gaps.append(
            {
                "gap_kind": "citation_share",
                "source_domain": domain,
                "citation_share_at_check": cite,
                "top_competitor_hint": comp_label,
                "extraction_snapshot_id": ex_id,
                "response_snapshot_id": resp_id,
            }
        )

    if not gaps and cite < 30.0:
        gaps.append(
            {
                "gap_kind": "citation_share_generic",
                "source_domain": None,
                "citation_share_at_check": cite,
                "top_competitor_hint": comp_label,
                "extraction_snapshot_id": None,
                "response_snapshot_id": None,
            }
        )
    return gaps


def _recommendation_nl_enrichment(
    profile: BusinessProfile,
    *,
    score: AEOScoreSnapshot | None = None,
) -> dict[str, Any]:
    """
    Context merged into gap dicts for NL templates and OpenAI; keeps guidance reusable and local-aware.
    """
    city = infer_city_from_address(profile.business_address or "")
    region_label = _region_label_for_profile(profile, city)
    crawl = _latest_completed_onboarding_crawl(profile)
    onpage_summary = _build_onpage_crawl_summary(crawl)
    site = (getattr(profile, "website_url", None) or "").strip()
    out: dict[str, Any] = {
        "business_name": (profile.business_name or "").strip(),
        "city": city,
        "region_label": region_label,
        "industry": (profile.industry or "").strip(),
        "services": "",  # placeholder when profile gains structured services
        "onpage_crawl_summary": onpage_summary,
        "canonical_domain": canonical_registrable_domain(site),
    }
    if score is not None:
        comps = _top_competitors_from_score(score, limit=5)
        if comps:
            out["competitors"] = ", ".join(comps)
    return out


def _prompt_short_label(prompt_text: str | None, *, max_words: int = 10) -> str:
    """Short intent phrase for copy and LLM JSON—never embed the full consumer prompt."""
    s = re.sub(r"\s+", " ", (prompt_text or "").strip())
    if not s:
        return "this type of question"
    words = s.split()
    if len(words) <= max_words:
        return s
    clip = " ".join(words[:max_words]).rstrip(".,;:!?")
    return clip + "…"


def _industry_snippet_for_copy(industry: str | None, *, max_len: int = 60) -> str:
    """First comma-separated segment, capped—avoids dumping long keyword-stuffed industry fields."""
    raw = (industry or "").strip()
    if not raw:
        return ""
    first = raw.split(",")[0].strip()
    if len(first) < 2:
        return ""
    if len(first) > max_len:
        return first[: max_len - 1].rstrip() + "…"
    return first


def generate_natural_language_recommendation(
    gap_object: dict[str, Any],
    *,
    business_profile: BusinessProfile | None = None,
) -> str:
    """
    Short human-readable line for a gap. Uses OpenAI only when
    settings.AEO_RECOMMENDATION_USE_OPENAI is true; otherwise template-based.
    """
    use_llm = bool(getattr(settings, "AEO_RECOMMENDATION_USE_OPENAI", False))
    if use_llm:
        try:
            return _nl_via_openai(gap_object, business_profile=business_profile)
        except Exception as exc:
            logger.warning("AEO recommendation NL OpenAI failed: %s", exc)
    return _nl_template_with_kinds(gap_object)


def _nl_template_with_kinds(gap: dict[str, Any]) -> str:
    kind = gap.get("gap_kind")
    if kind == "no_specific_gap":
        return _nl_template_no_gap(gap)
    return _nl_template(gap)


def _nl_template_no_gap(gap: dict[str, Any]) -> str:
    try:
        vis = float(gap.get("visibility") or 0)
        cite = float(gap.get("citation_share") or 0)
    except (TypeError, ValueError):
        vis, cite = 0.0, 0.0
    location = (gap.get("region_label") or gap.get("city") or "").strip() or "your area"
    bn = (gap.get("business_name") or "").strip() or "your business"
    return (
        f"No single prompt was flagged this pass—your headline visibility is about {vis:.0f}% and citation-style "
        f"share about {cite:.0f}%. You should align your homepage and top service page so the legal name of {bn} "
        f"matches your Google Business Profile, and add one FAQ pair that answers a common buyer question for {location} "
        f"so models have a citable snippet."
    )


def _nl_context_labels(gap: dict[str, Any]) -> tuple[str, str, str, str]:
    """(business_name, region, industry, services) with safe defaults for templates."""
    bn = (gap.get("business_name") or "").strip() or "your business"
    region = (gap.get("region_label") or "").strip() or (gap.get("city") or "").strip() or "your market"
    industry = (gap.get("industry") or "").strip()
    services = (gap.get("services") or "").strip()
    return bn, region, industry, services


def _nl_template(gap: dict[str, Any]) -> str:
    kind = gap.get("gap_kind")
    bn, region, industry, services = _nl_context_labels(gap)
    ind_snip = _industry_snippet_for_copy(industry)
    ind_lead = f"As a {ind_snip} operator, " if ind_snip else ""
    svc_tail = f" Where it is truthful, mention {services} on that page." if services else ""

    if kind == "visibility_miss":
        uid = (gap.get("url_identity_summary") or "").strip()
        st = (gap.get("brand_mentioned_url_status") or "").strip()
        if uid and st == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE:
            return (
                f"{ind_lead}{uid} "
                f"You should publish clear legal/trading-name copy, Organization or LocalBusiness JSON-LD with your true "
                f"url and sameAs to Google Business Profile and key profiles, plus a short disambiguation line so models "
                f"map {bn} to your canonical site—not the other domain.{svc_tail}"
            ).strip()
        if uid and st == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN:
            return (
                f"{ind_lead}{uid} "
                f"You should audit schema url fields, add a visible footer or FAQ “official website” line pointing to your "
                f"live domain, and ensure sitemap or listings expose that URL so models prefer it over dead links.{svc_tail}"
            ).strip()

        comps = gap.get("competitors_in_answer") or []
        comp_str = ", ".join(_competitor_display_names(comps if isinstance(comps, list) else [], limit=3))
        bench_raw = (gap.get("competitors") or "").strip()
        bench_list = [x.strip() for x in bench_raw.split(",") if x.strip()][:3]
        bench_disp = ", ".join(bench_list)
        if comp_str:
            return (
                f"{ind_lead}you should add or expand an on-page FAQ block plus matching FAQPage or Organization JSON-LD "
                f"on the service URL that fits this query intent in {region}, because modeled answers already cite "
                f"{comp_str} while {bn} is missing for this type of question.{svc_tail}"
            ).strip()
        if bench_disp:
            return (
                f"{ind_lead}you should strengthen the service page that matches this intent in {region} with clear "
                f"headings, a short FAQ, and schema that uses the same business name as your Google Business Profile so "
                f"{bn} can be extracted; your benchmarks often include {bench_disp}.{svc_tail}"
            ).strip()
        return (
            f"{ind_lead}you should add a visible FAQ section and Organization or LocalBusiness structured data on your "
            f"strongest commercial page for this intent in {region}, keeping entity naming aligned with live listings "
            f"so {bn} is eligible to be cited.{svc_tail}"
        ).strip()

    if kind == "citation_share":
        dom = gap.get("source_domain")
        try:
            cite = float(gap.get("citation_share_at_check") or 0)
        except (TypeError, ValueError):
            cite = 0.0
        comp = (gap.get("top_competitor_hint") or "named competitors").strip()
        if dom:
            cite_tail = f" Use categories and services you can verify." if services else ""
            return (
                f"Your modeled citation-style share is about {cite:.0f}%; you should claim or refresh a verifiable "
                f"profile or listing for {bn} on {dom} (and similar hubs in {region}), mirroring how {comp} appears "
                f"there without inventing attributes you do not offer.{cite_tail}"
            )
        return _nl_template({**gap, "gap_kind": "citation_share_generic"})

    if kind == "citation_share_generic":
        try:
            cite = float(gap.get("citation_share_at_check") or 0)
        except (TypeError, ValueError):
            cite = 0.0
        comp = (gap.get("top_competitor_hint") or "alternatives").strip()
        cite_tail = f" Keep wording factual." if services else ""
        return (
            f"Your modeled citation-style share is about {cite:.0f}%; you should line up trusted third-party listings "
            f"and data sources in {region} where {comp} already earns visibility, using consistent {bn} naming and "
            f"service categories across each profile.{cite_tail}"
        )

    return (
        f"You should tighten how {bn} appears on core service pages and in structured data for {region}, and keep "
        f"third-party mentions aligned so transactional and comparison-style answers can cite you consistently."
    )


def _infer_action_type_for_nl(g: dict[str, Any]) -> str:
    at = str(g.get("action_type") or "").strip()
    if at:
        return at
    k = g.get("gap_kind")
    if k == "visibility_miss":
        return "create_content"
    if k in ("citation_share", "citation_share_generic"):
        return "acquire_citation"
    return "review_visibility"


def _normalize_score_for_nl(gap_object: dict[str, Any]) -> dict[str, Any]:
    raw = gap_object.get("score")
    if isinstance(raw, dict) and raw:
        return {k: v for k, v in raw.items() if v is not None}
    out: dict[str, Any] = {}
    try:
        if gap_object.get("visibility_pct") is not None:
            out["visibility_pct"] = float(gap_object["visibility_pct"])
        elif gap_object.get("visibility") is not None:
            out["visibility_pct"] = float(gap_object["visibility"])
    except (TypeError, ValueError):
        pass
    try:
        if gap_object.get("citation_share_pct") is not None:
            out["citation_share_pct"] = float(gap_object["citation_share_pct"])
        elif gap_object.get("citation_share") is not None:
            out["citation_share_pct"] = float(gap_object["citation_share"])
        elif gap_object.get("citation_share_at_check") is not None:
            out["citation_share_pct"] = float(gap_object["citation_share_at_check"])
    except (TypeError, ValueError):
        pass
    try:
        if gap_object.get("weighted_position_pct") is not None:
            out["weighted_position_pct"] = float(gap_object["weighted_position_pct"])
    except (TypeError, ValueError):
        pass
    dom = gap_object.get("source_domain")
    if isinstance(dom, str) and dom.strip():
        out["citation_target_domain"] = dom.strip().lower()
    return out


def _competitor_strings_for_nl(gap_object: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add_one(s: str) -> None:
        t = (s or "").strip()[:160]
        if not t:
            return
        k = t.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(t)

    raw_list = gap_object.get("competitors_in_answer")
    if isinstance(raw_list, list):
        for x in _competitor_display_names(raw_list, limit=14):
            add_one(x)
    cs = gap_object.get("competitors")
    if isinstance(cs, str) and cs.strip():
        for part in cs.split(","):
            add_one(part)
    elif isinstance(cs, list):
        for x in _competitor_display_names(cs, limit=10):
            add_one(x)
    th = gap_object.get("top_competitor_hint")
    if isinstance(th, str) and th.strip():
        add_one(th.strip())
    return out


def _build_sanitized_nl_signals(gap_object: dict[str, Any]) -> dict[str, Any]:
    """
    Strip high-level ``reason`` / ``nl_explanation`` and ids; only raw signals for the model.
    Keys: prompt, action_type, competitors, business_name, region, gap_kind, score, crawl_summary;
    for ``visibility_miss`` also optional URL-identity keys (brand_mentioned_url_status, canonical_domain,
    cited_domain_in_answer, url_identity_summary, verification_summary).

    ``prompt`` is a short intent label only (not the full monitored question) so the LLM is not prompted to quote it.
    """
    full_pt = (gap_object.get("prompt_text") or gap_object.get("prompt") or "").strip()
    prompt = _prompt_short_label(full_pt, max_words=10)
    bn = (gap_object.get("business_name") or "").strip()
    region = (gap_object.get("region_label") or gap_object.get("region") or gap_object.get("city") or "").strip()
    crawl_summary = (gap_object.get("onpage_crawl_summary") or gap_object.get("crawl_summary") or "").strip()
    gk = gap_object.get("gap_kind")
    gap_kind = str(gk).strip() if gk is not None else ""
    out: dict[str, Any] = {
        "prompt": prompt,
        "action_type": _infer_action_type_for_nl(gap_object),
        "competitors": _competitor_strings_for_nl(gap_object),
        "business_name": bn,
        "region": region,
        "gap_kind": gap_kind,
        "score": _normalize_score_for_nl(gap_object),
        "crawl_summary": crawl_summary[:8000],
    }
    if gap_kind == "visibility_miss":
        cd = (gap_object.get("canonical_domain") or "").strip()
        if cd:
            out["canonical_domain"] = cd
        st = (gap_object.get("brand_mentioned_url_status") or "").strip()
        if st and st != "not_mentioned":
            out["brand_mentioned_url_status"] = st
        cited = gap_object.get("cited_domain_in_answer")
        if isinstance(cited, list) and cited:
            out["cited_domain_in_answer"] = [str(x).strip() for x in cited[:5] if str(x).strip()]
        uis = (gap_object.get("url_identity_summary") or "").strip()
        if uis:
            out["url_identity_summary"] = uis[:600]
        vs = (gap_object.get("verification_summary") or "").strip()
        if vs:
            out["verification_summary"] = vs[:200]
    return out


def _parse_recommendation_type_response(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "entity_alignment"
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            rt = str(obj.get("recommendation_type") or "").strip().lower()
            if rt in RECOMMENDATION_TYPE_CHOICES:
                return rt
    except json.JSONDecodeError:
        pass
    return "entity_alignment"


def _classify_recommendation_type(
    client: Any,
    sanitized: dict[str, Any],
    *,
    business_profile: BusinessProfile | None,
) -> str:
    payload = json.dumps(sanitized, ensure_ascii=False)[:4500]
    user_msg = (
        "Classify this gap for the next recommendation step. Output only valid JSON, one object.\n\n" + payload
    )
    completion = chat_completion_create_logged(
        client,
        operation="openai.chat.aeo_recommendation_type",
        business_profile=business_profile,
        model=_get_model(),
        messages=[
            {"role": "system", "content": AEO_RECOMMENDATION_TYPE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=100,
    )
    raw = (completion.choices[0].message.content or "").strip()
    return _parse_recommendation_type_response(raw)


def _nl_via_openai(
    gap_object: dict[str, Any],
    *,
    business_profile: BusinessProfile | None = None,
) -> str:
    api_env = getattr(settings, "AEO_RECOMMENDATION_API_KEY_ENV", DEFAULT_API_KEY_ENV)
    client = _get_client(api_env)
    sanitized = _build_sanitized_nl_signals(gap_object)
    if not sanitized.get("business_name") and business_profile is not None:
        sanitized["business_name"] = (business_profile.business_name or "").strip()
    rec_type = _classify_recommendation_type(
        client, sanitized, business_profile=business_profile
    )
    payload = json.dumps(sanitized, ensure_ascii=False)[:6000]
    user_msg = (
        f'recommendation_type: "{rec_type}"\n\n'
        "Gap signals (JSON only):\n"
        f"{payload}\n\n"
        "Write for the business owner. Second person (\"you\").\n"
        "Do not quote or repeat the consumer prompt text; say \"for this type of question\" or "
        "\"for this query intent\" only. The JSON \"prompt\" field is a short label—not text to paste.\n"
        "Sentence 1: specific actions (what to add or change on the site, or where to seek citations).\n"
        "Sentence 2: why answer engines are more likely to mention the business for this intent.\n"
        "If crawl_summary is non-empty, anchor at least one action to a page or topic from it.\n"
        "If crawl_summary is empty, still be specific; do not apologize.\n"
        "When brand_mentioned_url_status and domain fields are present: for mentioned_url_wrong_live, focus on entity "
        "disambiguation versus another business (canonical Organization/LocalBusiness name + url, sameAs, About/legal "
        "name clarity, GBP and major listing alignment)—do not treat the wrong domain as the client. For "
        "mentioned_url_wrong_broken, focus on strengthening discoverable official URLs and schema so models stop "
        "latching onto dead or hallucinated links (footer or FAQ “official site” only as a concrete tactic). For matched, "
        "do not invent URL problems.\n"
        "Avoid repeating the JSON wording verbatim."
    )
    completion = chat_completion_create_logged(
        client,
        operation="openai.chat.aeo_recommendation_nl",
        business_profile=business_profile,
        model=_get_model(),
        messages=[
            {"role": "system", "content": AEO_RECOMMENDATION_NL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=350,
    )
    return (
        (completion.choices[0].message.content or "").strip() or _nl_template_with_kinds(gap_object)
    )


def _build_create_content_recommendation(
    gap: dict[str, Any],
    *,
    score: AEOScoreSnapshot | None,
    priority: str,
    business_profile: BusinessProfile,
) -> dict[str, Any]:
    city = infer_city_from_address(business_profile.business_address or "")
    region = _region_label_for_profile(business_profile, city)
    bn = (business_profile.business_name or "").strip() or "This business"
    ind_snip = _industry_snippet_for_copy((business_profile.industry or "").strip())
    ind_bit = f" ({ind_snip})" if ind_snip else ""
    prompt_display = gap.get("prompt_text") or ""
    reason = (
        f"{bn} did not appear in the captured model answer for this monitored query{ind_bit}. "
        f"Add or tune a service-level page, FAQ block, and matching Organization/LocalBusiness schema for {region} "
        f"so this intent can cite {bn}."
    )
    uid = (gap.get("url_identity_summary") or "").strip()
    if uid:
        reason += f" {uid}"
    if score is not None and score.visibility_score < 20:
        reason += f" Overall visibility score for {bn} is {score.visibility_score:.1f}% (high priority)."
    elif score is not None:
        reason += f" Snapshot visibility score for {bn}: {score.visibility_score:.1f}%."
    comps = gap.get("competitors_in_answer") or []
    return {
        "action_type": "create_content",
        "prompt": prompt_display,
        "reason": reason,
        "priority": priority,
        "references": {
            "score_snapshot_id": score.id if score else None,
            "extraction_snapshot_id": gap.get("extraction_snapshot_id"),
            "response_snapshot_id": gap.get("response_snapshot_id"),
            "competitors": comps,
            "city_context": city or None,
        },
    }


def _build_acquire_citation_recommendation(
    gap: dict[str, Any],
    *,
    score: AEOScoreSnapshot | None,
    priority: str,
    business_profile: BusinessProfile,
) -> dict[str, Any]:
    city = infer_city_from_address(business_profile.business_address or "")
    region = _region_label_for_profile(business_profile, city)
    dom = gap.get("source_domain")
    try:
        cite = float(gap.get("citation_share_at_check") or 0)
    except (TypeError, ValueError):
        cite = 0.0
    comp = gap.get("top_competitor_hint")
    bn = (business_profile.business_name or "").strip() or "This business"
    if dom:
        reason = (
            f"Citation-style signal is weak ({cite:.1f}%); prioritize a verifiable listing or profile for {bn} on {dom} "
            f"and peer domains in {region}."
        )
    else:
        reason = (
            f"Citation-style signal is weak ({cite:.1f}%); add consistent {bn} profiles on category-trusted sites and "
            f"data sources models use in {region}."
        )
    if comp:
        reason += f" Answers often reference {comp}—match factual categories and proof patterns, not unverifiable claims."
    rec: dict[str, Any] = {
        "action_type": "acquire_citation",
        "reason": reason,
        "priority": priority,
        "references": {
            "score_snapshot_id": score.id if score else None,
            "extraction_snapshot_id": gap.get("extraction_snapshot_id"),
            "response_snapshot_id": gap.get("response_snapshot_id"),
            "competitors": _top_competitors_from_score(score),
        },
    }
    if dom:
        rec["source"] = dom
    return rec


def save_recommendation_run(
    business_profile: BusinessProfile,
    *,
    score_snapshot: AEOScoreSnapshot | None,
    recommendations: list[dict[str, Any]],
    visibility_score: float,
    weighted_position_score: float,
    citation_share: float,
) -> AEORecommendationRun:
    return AEORecommendationRun.objects.create(
        profile=business_profile,
        score_snapshot=score_snapshot,
        recommendations_json=recommendations,
        visibility_score_at_run=visibility_score,
        weighted_position_score_at_run=weighted_position_score,
        citation_share_at_run=citation_share,
    )


def generate_aeo_recommendations(
    business_profile: BusinessProfile,
    *,
    save: bool = True,
    enrich_with_nl: bool = True,
) -> dict[str, Any]:
    """
    Build recommendations from the latest score snapshot and latest extraction per response.

    When save=True, appends an AEORecommendationRun (historical, never overwrites prior runs).
    """
    score = (
        business_profile.aeo_score_snapshots.order_by("-created_at").select_related("profile").first()
    )
    extractions = latest_extraction_per_response(business_profile)
    site = (getattr(business_profile, "website_url", None) or "").strip()

    if score:
        visibility = float(score.visibility_score)
        weighted_pos = float(score.weighted_position_score)
        citation = float(score.citation_share)
    else:
        visibility = calculate_visibility_score(extractions, tracked_website_url=site)
        weighted_pos = calculate_weighted_position_score(extractions, tracked_website_url=site)
        citation = calculate_citation_share(extractions)

    priority = _priority_from_scores(visibility, citation)

    canonical_dom = canonical_registrable_domain(site)
    vis_gaps = analyze_visibility_gaps(
        score,
        extractions,
        tracked_website_url=site,
        canonical_domain=canonical_dom,
    )
    cite_gaps = analyze_citation_gaps(score, extractions, citation_share=citation)

    recommendations: list[dict[str, Any]] = []
    nl_ctx = _recommendation_nl_enrichment(business_profile, score=score)

    def _headline_score_dict(*, cite_gap: dict[str, Any] | None = None) -> dict[str, Any]:
        sc: dict[str, Any] = {
            "visibility_pct": visibility,
            "citation_share_pct": citation,
        }
        if score is not None:
            sc["weighted_position_pct"] = float(score.weighted_position_score)
        if cite_gap:
            dom = cite_gap.get("source_domain")
            if isinstance(dom, str) and dom.strip():
                sc["citation_target_domain"] = dom.strip().lower()
        return sc

    for gap in vis_gaps:
        rec = _build_create_content_recommendation(
            gap,
            score=score,
            priority=priority,
            business_profile=business_profile,
        )
        if enrich_with_nl:
            rec["nl_explanation"] = generate_natural_language_recommendation(
                {
                    **gap,
                    **nl_ctx,
                    "action_type": "create_content",
                    "score": _headline_score_dict(),
                },
                business_profile=business_profile,
            )
        recommendations.append(rec)

    for gap in cite_gaps:
        rec = _build_acquire_citation_recommendation(
            gap,
            score=score,
            priority=priority,
            business_profile=business_profile,
        )
        if enrich_with_nl:
            rec["nl_explanation"] = generate_natural_language_recommendation(
                {
                    **gap,
                    **nl_ctx,
                    "action_type": "acquire_citation",
                    "score": _headline_score_dict(cite_gap=gap),
                },
                business_profile=business_profile,
            )
        recommendations.append(rec)

    if not recommendations:
        bn = (business_profile.business_name or "").strip() or "This business"
        low_rec = {
            "action_type": "review_visibility",
            "reason": (
                f"No single prompt gap fired for {bn} this run (visibility {visibility:.1f}%, citation-style "
                f"{citation:.1f}%). Re-check after you update FAQs, schema, or key listings so the next snapshot can "
                f"catch shifts."
            ),
            "priority": "low" if priority == "low" else priority,
            "references": {
                "score_snapshot_id": score.id if score else None,
                "extraction_snapshot_id": None,
                "response_snapshot_id": None,
                "competitors": _top_competitors_from_score(score),
            },
        }
        if enrich_with_nl:
            low_rec["nl_explanation"] = generate_natural_language_recommendation(
                {
                    "gap_kind": "no_specific_gap",
                    "visibility": visibility,
                    "citation_share": citation,
                    **nl_ctx,
                    "action_type": "review_visibility",
                    "score": _headline_score_dict(),
                },
                business_profile=business_profile,
            )
        recommendations.append(low_rec)

    run_id: int | None = None
    if save:
        run = save_recommendation_run(
            business_profile,
            score_snapshot=score,
            recommendations=recommendations,
            visibility_score=visibility,
            weighted_position_score=weighted_pos,
            citation_share=citation,
        )
        run_id = run.id

    return {
        "recommendation_run_id": run_id,
        "score_snapshot_id": score.id if score else None,
        "extraction_count": len(extractions),
        "visibility_score": visibility,
        "weighted_position_score": weighted_pos,
        "citation_share": citation,
        "priority_tier": priority,
        "recommendations": recommendations,
    }
