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

ABSENCE_REASON_CHOICES: frozenset[str] = frozenset(
    {
        "competitor_authority",
        "missing_category_page",
        "entity_confusion",
        "missing_local_signal",
        "missing_trust_signal",
    }
)

INTENT_TYPE_CHOICES: frozenset[str] = frozenset(
    {
        "transactional",
        "trust",
        "comparison",
        "local",
        "informational",
    }
)

CONTENT_ANGLE_CHOICES: frozenset[str] = frozenset(
    {
        "service_offer",
        "trust_proof",
        "comparison",
        "local_availability",
        "brand_identity",
        "safety_authority",
    }
)

_TRUSTED_CITATION_DOMAIN_HINTS: tuple[str, ...] = (
    "yelp",
    "yellowpages",
    "bbb.",
    "angi",
    "homeadvisor",
    "thumbtack",
    "houzz",
    "tripadvisor",
    "trustpilot",
    "g2.",
    "clutch",
    "capterra",
    "chamberofcommerce",
    ".org",
    ".gov",
    "wikipedia",
    "linkedin",
    "facebook.com",
    "instagram.com",
    "youtube.com",
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


def _domain_from_any(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    if "://" not in s and "/" not in s:
        s = "https://" + s
    return canonical_registrable_domain(s) or ""


def _competitor_domains_from_extractions(extraction_snapshots: list[AEOExtractionSnapshot]) -> set[str]:
    out: set[str] = set()
    for ex in extraction_snapshots:
        comps = ex.competitors_json or []
        if not isinstance(comps, list):
            continue
        for c in comps:
            if not isinstance(c, dict):
                continue
            for key in ("url", "domain"):
                dom = _domain_from_any(str(c.get(key) or ""))
                if dom:
                    out.add(dom)
    return out


def _is_allowed_citation_target_domain(domain: str, *, excluded: set[str]) -> bool:
    d = (domain or "").strip().lower()
    if not d or "." not in d:
        return False
    if d in excluded:
        return False
    return any(h in d for h in _TRUSTED_CITATION_DOMAIN_HINTS)


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
            f"Answer results are pointing to {other} instead of your official site {canon_disp}. "
            f"Make your business name and official website explicit across your homepage, About page, and listings."
        )
    elif status_str == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN:
        other = cited_raw or "a URL"
        summary = (
            f"Answer results are pointing to {other}, but that link is broken or unreliable. "
            f"Reinforce your official website everywhere your business is listed so engines stop using bad links."
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

    own_domain = canonical_registrable_domain(
        getattr(score_snapshot, "profile", None).website_url if score_snapshot and getattr(score_snapshot, "profile", None) else ""
    )
    excluded_domains = _competitor_domains_from_extractions(extraction_snapshots)
    if own_domain:
        excluded_domains.add(own_domain)

    dom_counter: Counter[str] = Counter()
    for ex in extraction_snapshots:
        cites = ex.citations_json or []
        if not isinstance(cites, list):
            continue
        for raw in cites:
            d = _domain_from_any(str(raw))
            if d and _is_allowed_citation_target_domain(d, excluded=excluded_domains):
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


def _derive_absence_reason(gap_object: dict[str, Any]) -> str:
    raw = str(gap_object.get("absence_reason") or "").strip().lower()
    if raw in ABSENCE_REASON_CHOICES:
        return raw
    status = str(gap_object.get("brand_mentioned_url_status") or "").strip()
    if status in (
        AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE,
        AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN,
    ):
        return "entity_confusion"
    kind = str(gap_object.get("gap_kind") or "").strip()
    if kind in ("citation_share", "citation_share_generic"):
        return "competitor_authority"
    action_type = str(gap_object.get("action_type") or "").strip()
    if action_type == "create_content":
        return "missing_category_page"
    return ""


def _derive_intent_type(gap_object: dict[str, Any]) -> str:
    raw = str(gap_object.get("intent_type") or "").strip().lower()
    if raw in INTENT_TYPE_CHOICES:
        return raw
    p = str(gap_object.get("prompt_text") or gap_object.get("prompt") or "").strip().lower()
    if p:
        if any(tok in p for tok in ("near me", "nearby", "in ", "local", "city", "area")):
            return "local"
        if any(tok in p for tok in ("best", "top", "vs", "compare", "comparison", "alternative")):
            return "comparison"
        if any(tok in p for tok in ("price", "pricing", "cost", "quote", "book", "hire", "buy")):
            return "transactional"
        if any(tok in p for tok in ("review", "trusted", "safe", "licensed", "certified", "reliable")):
            return "trust"
    action_type = str(gap_object.get("action_type") or "").strip()
    if action_type == "acquire_citation":
        return "trust"
    if action_type == "create_content":
        return "transactional"
    return "informational"


def _derive_content_angle(gap_object: dict[str, Any]) -> str:
    raw = str(gap_object.get("content_angle") or "").strip().lower()
    if raw in CONTENT_ANGLE_CHOICES:
        return raw
    absence_reason = _derive_absence_reason(gap_object)
    intent_type = _derive_intent_type(gap_object)
    status = str(gap_object.get("brand_mentioned_url_status") or "").strip()
    if status in (
        AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE,
        AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN,
    ) or absence_reason == "entity_confusion":
        return "brand_identity"
    if absence_reason == "missing_local_signal" or intent_type == "local":
        return "local_availability"
    if absence_reason == "missing_trust_signal" or intent_type == "trust":
        return "trust_proof"
    if absence_reason == "competitor_authority":
        return "safety_authority"
    if intent_type == "comparison":
        return "comparison"
    return "service_offer"


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
        f"Update your homepage and top service page so {bn} is written the same way as your business listings, then add one short FAQ for buyers in {location}. "
        f"This gives answer engines clearer details to trust, which helps your business appear more often in AI-generated answers."
    )


def _nl_context_labels(gap: dict[str, Any]) -> tuple[str, str, str, str]:
    """(business_name, region, industry, services) with safe defaults for templates."""
    bn = (gap.get("business_name") or "").strip() or "your business"
    region = (gap.get("region_label") or "").strip() or (gap.get("city") or "").strip() or "your market"
    industry = (gap.get("industry") or "").strip()
    services = (gap.get("services") or "").strip()
    return bn, region, industry, services


def _nl_template(gap: dict[str, Any]) -> str:
    kind = str(gap.get("gap_kind") or "").strip()
    bn, region, _industry, services = _nl_context_labels(gap)
    absence_reason = _derive_absence_reason(gap)
    intent_type = _derive_intent_type(gap)
    content_angle = _derive_content_angle(gap)
    status = str(gap.get("brand_mentioned_url_status") or "").strip()
    cited = ""
    cited_raw = gap.get("cited_domain_in_answer")
    if isinstance(cited_raw, list) and cited_raw:
        cited = str(cited_raw[0] or "").strip()
    canonical = str(gap.get("canonical_domain") or "").strip()

    if status == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE:
        action = (
            f"Add a short About section that says {canonical or 'your website'} is your official website and keep the same business name on your site and listings"
        )
        why = (
            f"This helps answer engines stop confusing you with {cited or 'another business'} and mention {bn} more often in relevant answers."
        )
        return f"{action}. {why}"
    if status == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN:
        action = (
            "Add an \"Official website\" line in your footer and contact page, then update your main listings to the same live URL"
        )
        why = (
            "This gives answer engines one trusted website to reference instead of broken links, so your business appears more consistently."
        )
        return f"{action}. {why}"

    if content_angle == "brand_identity":
        action = (
            "Clarify your business identity on the homepage and About page, including your exact business name and official website"
        )
    elif content_angle == "local_availability":
        action = (
            f"Add service-area copy for {region} on your main service pages and match that same location wording across your business listings"
        )
    elif content_angle == "trust_proof":
        action = (
            "Add a short FAQ with proof points such as credentials, reviews, and project examples on the most relevant page"
        )
    elif content_angle == "comparison":
        action = (
            "Publish a comparison section that explains how your service differs and add concrete proof like certifications, partnerships, or project outcomes"
        )
    elif content_angle == "safety_authority":
        action = (
            "Add an authority section with certifications, safety standards, and real project examples on your key service page"
        )
    elif absence_reason == "missing_category_page" or intent_type == "transactional":
        action = (
            f"Create a dedicated service page for this offer in {region} with clear pricing, scope, and a strong contact section"
        )
    elif absence_reason == "missing_local_signal" or intent_type == "local":
        action = (
            f"Add service-area copy for {region} on your main service pages and match that same location wording across your business listings"
        )
    elif absence_reason == "missing_trust_signal" or intent_type == "trust":
        action = (
            "Add a short FAQ with proof points such as credentials, reviews, and project examples on the most relevant page"
        )
    elif absence_reason == "competitor_authority" or intent_type == "comparison":
        action = (
            "Publish a comparison section that explains how your service differs and add concrete proof like certifications, partnerships, or project outcomes"
        )
    else:
        action = (
            "Add a concise Q&A section that answers common buyer questions on the page most related to this intent"
        )

    if kind in ("citation_share", "citation_share_generic"):
        dom = str(gap.get("source_domain") or "").strip()
        if dom:
            action = (
                f"Update your profile on {dom} and similar trusted directories with the same business name, services, and website you use on your own site"
            )
        else:
            action = (
                "Update your top third-party listings so your business name, services, and website are consistent everywhere"
            )

    if services:
        action += f", including services like {services}"

    why = (
        "This gives answer engines clearer, verifiable business details they can trust when choosing who to mention for this type of question."
    )
    return f"{action}. {why}"


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
    absence_reason = _derive_absence_reason(gap_object)
    intent_type = _derive_intent_type(gap_object)
    content_angle = _derive_content_angle(gap_object)
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
    if absence_reason:
        out["absence_reason"] = absence_reason
    if intent_type:
        out["intent_type"] = intent_type
    if content_angle:
        out["content_angle"] = content_angle
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
        "Write for a non-technical business owner. Second person (\"you\").\n"
        "Do not quote or repeat the consumer prompt text; say \"for this type of question\" only.\n"
        "Use exactly two short sentences.\n"
        "Sentence 1: exact action to take.\n"
        "Sentence 2: simple reason this helps the business appear more often in AI-generated answers.\n"
        "Never begin with: \"As a business owner\", \"As an operator\", or \"As {business_name} operator\".\n"
        "Never use these terms: modeled answers, canonical, entity graph, disambiguation, attribution, citation share, gap score.\n"
        "Prefer plain language; if schema is needed, explain it simply as structured business details/search markup.\n"
        "Use content_angle and absence_reason to choose emphasis before writing.\n"
        "If crawl_summary is non-empty, anchor at least one action to a page or topic from it.\n"
        "If crawl_summary is empty, still be specific; do not apologize.\n"
        "When brand_mentioned_url_status and domain fields are present: for mentioned_url_wrong_live, focus on making the "
        "official business name and website clear across homepage/About/listings; do not treat the wrong domain as the client. "
        "For mentioned_url_wrong_broken, focus on reinforcing one live official website across site and listings. For matched, "
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
    grouped_prompts = gap.get("_grouped_prompts") if isinstance(gap.get("_grouped_prompts"), list) else []
    matched_prompt_texts = [str(p).strip() for p in grouped_prompts if str(p).strip()]
    grouped_resp = gap.get("_grouped_response_ids") if isinstance(gap.get("_grouped_response_ids"), list) else []
    matched_response_snapshot_ids: list[int] = []
    for rid in grouped_resp:
        try:
            v = int(rid)
        except (TypeError, ValueError):
            continue
        if v not in matched_response_snapshot_ids:
            matched_response_snapshot_ids.append(v)
    references: dict[str, Any] = {
        "score_snapshot_id": score.id if score else None,
        "extraction_snapshot_id": gap.get("extraction_snapshot_id"),
        "response_snapshot_id": gap.get("response_snapshot_id"),
        "competitors": comps,
        "city_context": city or None,
    }
    if matched_prompt_texts:
        references["matched_prompt_texts"] = matched_prompt_texts
    if matched_response_snapshot_ids:
        references["matched_response_snapshot_ids"] = matched_response_snapshot_ids
    return {
        "action_type": "create_content",
        "prompt": prompt_display,
        "reason": reason,
        "priority": priority,
        "references": references,
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
    grouped_prompts = gap.get("_grouped_prompts") if isinstance(gap.get("_grouped_prompts"), list) else []
    matched_prompt_texts = [str(p).strip() for p in grouped_prompts if str(p).strip()]
    grouped_resp = gap.get("_grouped_response_ids") if isinstance(gap.get("_grouped_response_ids"), list) else []
    matched_response_snapshot_ids: list[int] = []
    for rid in grouped_resp:
        try:
            v = int(rid)
        except (TypeError, ValueError):
            continue
        if v not in matched_response_snapshot_ids:
            matched_response_snapshot_ids.append(v)
    references: dict[str, Any] = {
        "score_snapshot_id": score.id if score else None,
        "extraction_snapshot_id": gap.get("extraction_snapshot_id"),
        "response_snapshot_id": gap.get("response_snapshot_id"),
        "competitors": _top_competitors_from_score(score),
    }
    if matched_prompt_texts:
        references["matched_prompt_texts"] = matched_prompt_texts
    if matched_response_snapshot_ids:
        references["matched_response_snapshot_ids"] = matched_response_snapshot_ids
    rec: dict[str, Any] = {
        "action_type": "acquire_citation",
        "reason": reason,
        "priority": priority,
        "references": references,
    }
    if dom:
        rec["source"] = dom
    return rec


def _group_gap_objects_for_recommendations(
    gaps: list[dict[str, Any]],
    *,
    action_type: str,
) -> list[dict[str, Any]]:
    """
    Consolidate repeated advice: group by (absence_reason, content_angle, action_type) so multiple
    prompts with the same root issue produce one stronger client-facing recommendation.
    """
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for g in gaps:
        ar = _derive_absence_reason(g)
        ca = _derive_content_angle(g)
        key = (ar, ca, action_type)
        base = grouped.get(key)
        if base is None:
            base = dict(g)
            base["absence_reason"] = ar
            base["content_angle"] = ca
            base["_grouped_prompt_count"] = 0
            base["_grouped_prompts"] = []
            grouped[key] = base

        prompt = str(g.get("prompt_text") or "").strip()
        if prompt:
            seen = {str(x).strip().lower() for x in base["_grouped_prompts"]}
            if prompt.lower() not in seen and len(base["_grouped_prompts"]) < 8:
                base["_grouped_prompts"].append(prompt)
        base["_grouped_prompt_count"] += 1

        if not base.get("source_domain") and g.get("source_domain"):
            base["source_domain"] = g.get("source_domain")
        if base.get("extraction_snapshot_id") is None and g.get("extraction_snapshot_id") is not None:
            base["extraction_snapshot_id"] = g.get("extraction_snapshot_id")
        if base.get("response_snapshot_id") is None and g.get("response_snapshot_id") is not None:
            base["response_snapshot_id"] = g.get("response_snapshot_id")
        if "_grouped_response_ids" not in base:
            base["_grouped_response_ids"] = []
        rid = g.get("response_snapshot_id")
        try:
            rid_int = int(rid) if rid is not None else None
        except (TypeError, ValueError):
            rid_int = None
        if rid_int is not None and rid_int not in base["_grouped_response_ids"] and len(base["_grouped_response_ids"]) < 50:
            base["_grouped_response_ids"].append(rid_int)

        comp_base = base.get("competitors_in_answer")
        comp_add = g.get("competitors_in_answer")
        if isinstance(comp_base, list) and isinstance(comp_add, list):
            merged = comp_base + comp_add
            base["competitors_in_answer"] = merged[:20]

    out: list[dict[str, Any]] = []
    for row in grouped.values():
        prompts = row.get("_grouped_prompts") or []
        if isinstance(prompts, list) and prompts:
            if len(prompts) == 1:
                row["prompt_text"] = prompts[0]
            else:
                row["prompt_text"] = f"{prompts[0]} (+{len(prompts) - 1} similar prompts)"
        row.pop("_grouped_prompts", None)
        row.pop("_grouped_prompt_count", None)
        out.append(row)
    return out


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
    vis_gaps_raw = analyze_visibility_gaps(
        score,
        extractions,
        tracked_website_url=site,
        canonical_domain=canonical_dom,
    )
    cite_gaps_raw = analyze_citation_gaps(score, extractions, citation_share=citation)
    vis_gaps = _group_gap_objects_for_recommendations(vis_gaps_raw, action_type="create_content")
    cite_gaps = _group_gap_objects_for_recommendations(cite_gaps_raw, action_type="acquire_citation")

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
