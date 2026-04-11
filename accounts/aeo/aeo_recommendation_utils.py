"""
Phase 5: actionable AEO recommendations from score + extraction snapshots.

Does not implement notifications, dashboards, or content automation.

Note: `accounts.openai_utils.generate_aeo_recommendations` is unrelated (legacy string-list helper for serializers).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Iterable
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

AEO_RECOMMENDATION_VERBOSITY_COMPACT = "compact"
AEO_RECOMMENDATION_VERBOSITY_EXPANDED = "expanded"
AEO_RECOMMENDATION_VERBOSITY_CHOICES: frozenset[str] = frozenset(
    {AEO_RECOMMENDATION_VERBOSITY_COMPACT, AEO_RECOMMENDATION_VERBOSITY_EXPANDED}
)

# Multi-angle recommendation facets (API / UI); distinct from Phase-5 content_angle taxonomy.
ANGLE_CONTENT = "content"
ANGLE_ON_PAGE = "on_page"
ANGLE_SCHEMA = "schema"
ANGLE_ENTITY_LOCATION = "entity_location"
ANGLE_COMPETITIVE_PARITY = "competitive_parity"
ANGLE_PRESENCE_LISTINGS = "presence_listings"
# Single bucket for UI: no Schema / Content / Entity section headings.
ANGLE_FLAT_TODO = "todo"

# Maximum recommendation leaves per run (short prioritized to-do list).
AEO_RECOMMENDATION_MAX_LEAVES = 3

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
    verbosity: str = AEO_RECOMMENDATION_VERBOSITY_COMPACT,
) -> list[dict[str, Any]]:
    """
    Gaps when citation-style share of mention-units is weak; suggests domains to prioritize.

    Uses aggregated citation domains from extractions plus competitor context from the score snapshot.
    ``verbosity=expanded`` keeps more directory targets (less lossy vs prompt count).
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

    top_domains = [h for h, _ in dom_counter.most_common(8)]
    top_comp = _top_competitors_from_score(score_snapshot, limit=2)
    comp_label = top_comp[0] if top_comp else "competitors"

    n_dom = 5 if verbosity == AEO_RECOMMENDATION_VERBOSITY_EXPANDED else 3
    for domain in top_domains[:n_dom]:
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


def _analyze_citation_gaps_per_response(
    score_snapshot: AEOScoreSnapshot | None,
    extraction_snapshots: list[AEOExtractionSnapshot],
    *,
    citation_share: float,
    excluded_domains: set[str],
) -> list[dict[str, Any]]:
    """
    One citation gap per (response_snapshot, strongest trusted directory domain) when share is weak.
    Surfaces prompt-level attachment ids without replacing aggregate citation gaps.
    """
    cite = float(citation_share)
    if cite >= 30.0:
        return []
    top_comp = _top_competitors_from_score(score_snapshot, limit=2)
    comp_label = top_comp[0] if top_comp else "competitors"
    gaps: list[dict[str, Any]] = []
    seen_pair: set[tuple[int, str]] = set()
    for ex in extraction_snapshots:
        resp = getattr(ex, "response_snapshot", None)
        if resp is None:
            continue
        try:
            rid = int(resp.id)
        except (TypeError, ValueError, AttributeError):
            continue
        cites = ex.citations_json or []
        if not isinstance(cites, list):
            continue
        dom_counter: Counter[str] = Counter()
        for raw in cites:
            d = _domain_from_any(str(raw))
            if d and _is_allowed_citation_target_domain(d, excluded=excluded_domains):
                dom_counter[d] += 1
        if not dom_counter:
            continue
        best_dom = dom_counter.most_common(1)[0][0]
        key = (rid, best_dom)
        if key in seen_pair:
            continue
        seen_pair.add(key)
        pt = (getattr(resp, "prompt_text", None) or "").strip()
        gaps.append(
            {
                "gap_kind": "citation_share",
                "source_domain": best_dom,
                "citation_share_at_check": cite,
                "top_competitor_hint": comp_label,
                "extraction_snapshot_id": ex.id,
                "response_snapshot_id": rid,
                "prompt_text": pt,
            }
        )
    return gaps


def _merge_citation_gaps_deduped(
    aggregate_gaps: list[dict[str, Any]],
    per_response_gaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer per-response rows first so prompt-level ids win; drop duplicate (response_id, domain)."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[int | None, str | None]] = set()

    def _key(g: dict[str, Any]) -> tuple[int | None, str | None]:
        try:
            rs = int(g["response_snapshot_id"]) if g.get("response_snapshot_id") is not None else None
        except (TypeError, ValueError):
            rs = None
        dom = g.get("source_domain")
        dom_s = str(dom).strip().lower() if dom else None
        return (rs, dom_s)

    for g in per_response_gaps + aggregate_gaps:
        k = _key(g)
        if k in seen:
            continue
        seen.add(k)
        out.append(g)
    return out


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
    _ = vis, cite
    return (
        f"Update your homepage and main service page so {bn} is spelled the same on your site as on your listings.\n"
        f"Add one short FAQ with real buyer questions for people in {location}.\n"
        f"This gives AI assistants simple facts to trust so they can mention you more often."
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
    why_common = (
        "This helps AI assistants match real customer questions to your business and mention you more often."
    )

    if status == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE:
        return (
            f"On your About page, state clearly that {canonical or 'your website'} is {bn}'s official website.\n"
            f"Use the same business name on your site and on listings so you are not mixed up with {cited or 'another site'}.\n"
            f"{why_common}"
        )
    if status == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN:
        return (
            "Add a line in your footer and contact page with your one working official website link.\n"
            "Update Google Business Profile and other main listings to that same live link.\n"
            f"{why_common}"
        )

    line1 = ""
    line2 = ""

    if content_angle == "brand_identity":
        line1 = f"On your homepage and About page, show the exact name people should use for {bn} and your official website."
        line2 = "Keep the same spelling everywhere buyers might look."
    elif content_angle == "local_availability":
        line1 = f"On your main service pages, say clearly which areas near {region} you serve."
        line2 = "Use the same city and neighborhood names on your site as on your business listings."
    elif content_angle == "trust_proof":
        line1 = "Add a short FAQ on the page that fits this topic with real proof—reviews, photos, or credentials."
        line2 = "Answer the worries buyers voice for this type of question in plain words."
    elif content_angle == "comparison":
        line1 = "Add a short section that states how your service differs and what a buyer gets."
        line2 = "Include one or two facts you can prove, such as years in business or a guarantee."
    elif content_angle == "safety_authority":
        line1 = "On your main service page, list safety steps, licenses, or training you use."
        line2 = "Add a simple example or outcome buyers can picture."
    elif absence_reason == "missing_category_page" or intent_type == "transactional":
        line1 = f"Add or refresh a service page for {region} with what you offer, typical pricing or ranges, and how to contact you."
        line2 = "Put the next step (call, book, or visit) near the top so busy buyers see it first."
    elif absence_reason == "missing_local_signal" or intent_type == "local":
        line1 = f"State your service area for {region} on the page that matches local questions."
        line2 = "Repeat the same area wording on your main online listings."
    elif absence_reason == "missing_trust_signal" or intent_type == "trust":
        line1 = "Add a short FAQ that answers trust questions for this topic—who you are, how you work, and what others say."
        line2 = "Use real reviews or project notes buyers can verify."
    elif absence_reason == "competitor_authority" or intent_type == "comparison":
        line1 = "Add a simple comparison-style section: what you include, how you price, and what makes you a safe choice."
        line2 = "Back claims with facts you can show, not empty slogans."
    else:
        line1 = "Add a short Q&A on the page that best matches this type of question."
        line2 = "Use the same words customers use when they ask for help."

    if kind in ("citation_share", "citation_share_generic"):
        dom = str(gap.get("source_domain") or "").strip()
        if dom:
            line1 = f"Update your {dom} profile (and similar trusted sites) so your name, phone, services, and website match your own site."
        else:
            line1 = "Update your top online listings so your name, services, phone, and website match your website."
        line2 = "Fill every section the site offers; add recent photos if you can."

    if services:
        line2 = f"{line2} Mention services such as {services}.".strip()

    return f"{line1}\n{line2}\n{why_common}"


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
        "Write for a non-technical business owner. Use \"you\". "
        "Do not quote the full consumer prompt; say \"this type of question\" at most once.\n"
        "Output exactly 3 lines separated by newlines (no bullets, no numbers, no headings):\n"
        "Line 1: what to do.\n"
        "Line 2: where (page or listing) and what to write or add.\n"
        "Line 3: one sentence on why this helps you show up in AI answers.\n"
        "Never begin with: \"As a business owner\" or similar.\n"
        "Never use: JSON-LD, schema, sameAs, entity, canonical, markup, competitive parity, leverage, optimize, signals, job-to-be-done.\n"
        "Tie the advice to intent_type and this type of question (pricing, local, trust, comparison, etc.).\n"
        "If crawl_summary is non-empty, mention one real page or topic from it on line 2.\n"
        "If brand_mentioned_url_status is mentioned_url_wrong_live or wrong_broken, focus on one clear official website and matching listings.\n"
        "Avoid copying JSON keys verbatim."
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


def _stable_parent_group_id(root_issue_key: str) -> str:
    h = hashlib.sha256(root_issue_key.encode("utf-8")).hexdigest()[:14]
    return f"aeo_grp_{h}"


def _root_issue_key(gap: dict[str, Any], action_type: str) -> str:
    gk = str(gap.get("gap_kind") or "")
    if action_type == "acquire_citation":
        dom = str(gap.get("source_domain") or "generic").strip().lower()
        return f"{action_type}|{gk}|{dom}"
    ar = _derive_absence_reason(gap)
    ca = _derive_content_angle(gap)
    rs = gap.get("response_snapshot_id")
    return f"{action_type}|{gk}|{ar}|{ca}|rs:{rs}"


def _action_count_for_verbosity(verbosity: str) -> int:
    # One focused step per recommendation; detail lives in nl_explanation (3 short lines).
    return 2 if verbosity == AEO_RECOMMENDATION_VERBOSITY_EXPANDED else 1


def _primary_action_rows_from_nl(nl: str) -> list[dict[str, str]]:
    """Map 3-line NL text into a single card row (title + body)."""
    lines = [ln.strip() for ln in (nl or "").splitlines() if ln.strip()]
    if not lines:
        return []
    title = lines[0]
    body = "\n".join(lines[1:]).strip()
    return [{"title": title, "description": body, "priority": "high"}]


def _apply_nl_as_primary_actions(rec: dict[str, Any]) -> None:
    nl = str(rec.get("nl_explanation") or "")
    lines = [ln.strip() for ln in nl.splitlines() if ln.strip()]
    if len(lines) >= 2:
        rows = _primary_action_rows_from_nl(nl)
        if rows:
            rec["actions"] = rows


def _angles_for_visibility_gap(gap: dict[str, Any], *, verbosity: str) -> list[str]:
    status = str(gap.get("brand_mentioned_url_status") or "").strip()
    intent = _derive_intent_type(gap)
    absence = _derive_absence_reason(gap)
    ca = _derive_content_angle(gap)
    ordered: list[str] = []
    if status in (
        AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE,
        AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN,
    ) or absence == "entity_confusion":
        ordered = [ANGLE_ENTITY_LOCATION, ANGLE_ON_PAGE, ANGLE_SCHEMA, ANGLE_CONTENT, ANGLE_COMPETITIVE_PARITY]
    elif intent == "local" or absence == "missing_local_signal":
        ordered = [ANGLE_ENTITY_LOCATION, ANGLE_SCHEMA, ANGLE_CONTENT, ANGLE_ON_PAGE, ANGLE_COMPETITIVE_PARITY]
    elif intent == "comparison" or ca == "comparison":
        ordered = [ANGLE_CONTENT, ANGLE_COMPETITIVE_PARITY, ANGLE_ON_PAGE, ANGLE_SCHEMA, ANGLE_ENTITY_LOCATION]
    elif intent == "trust" or absence in ("missing_trust_signal", "competitor_authority"):
        ordered = [ANGLE_CONTENT, ANGLE_SCHEMA, ANGLE_COMPETITIVE_PARITY, ANGLE_ON_PAGE, ANGLE_ENTITY_LOCATION]
    else:
        ordered = [ANGLE_CONTENT, ANGLE_SCHEMA, ANGLE_ENTITY_LOCATION, ANGLE_ON_PAGE, ANGLE_COMPETITIVE_PARITY]
    seen: set[str] = set()
    uniq: list[str] = []
    for a in ordered:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    cap = 6 if verbosity == AEO_RECOMMENDATION_VERBOSITY_EXPANDED else 3
    return uniq[:cap]


def _angles_for_citation_gap(*, verbosity: str) -> list[str]:
    base = [ANGLE_PRESENCE_LISTINGS, ANGLE_CONTENT, ANGLE_SCHEMA, ANGLE_COMPETITIVE_PARITY, ANGLE_ENTITY_LOCATION]
    cap = 5 if verbosity == AEO_RECOMMENDATION_VERBOSITY_EXPANDED else 3
    return base[:cap]


def _applies_to_prompts_and_response_ids(gap: dict[str, Any]) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    mpt = gap.get("matched_prompt_texts")
    if isinstance(mpt, list) and mpt:
        texts = [str(p).strip() for p in mpt if str(p).strip()]
    if not texts:
        gp = gap.get("_grouped_prompts") if isinstance(gap.get("_grouped_prompts"), list) else []
        if gp:
            texts = [str(p).strip() for p in gp if str(p).strip()]
    if not texts:
        pt = str(gap.get("prompt_text") or gap.get("prompt") or "").strip()
        if pt:
            texts = [pt]
    resp_ids: list[int] = []
    gr = gap.get("_grouped_response_ids") if isinstance(gap.get("_grouped_response_ids"), list) else []
    for rid in gr:
        try:
            v = int(rid)
        except (TypeError, ValueError):
            continue
        if v not in resp_ids:
            resp_ids.append(v)
    if not resp_ids:
        rs = gap.get("response_snapshot_id")
        try:
            ri = int(rs) if rs is not None else None
        except (TypeError, ValueError):
            ri = None
        if ri is not None:
            resp_ids.append(ri)
    return texts, resp_ids


def _competitor_names_phrase(gap: dict[str, Any], *, limit: int = 3) -> str:
    names = _competitor_display_names(gap.get("competitors_in_answer"), limit=limit)
    if not names:
        th = gap.get("top_competitor_hint")
        if isinstance(th, str) and th.strip():
            return th.strip()
        return "competitors named in AI answers"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and others"


def _cluster_summary_line(gap: dict[str, Any]) -> str:
    """Short, human phrase for applies_to (not internal taxonomy labels)."""
    short = _prompt_short_label(str(gap.get("prompt_text") or gap.get("prompt") or ""))
    it = _derive_intent_type(gap)
    if short and short != "this type of question":
        return f'Questions like "{short}" ({it}).'
    return f"Questions in the {it} category."


def _build_angle_summary(
    gap: dict[str, Any],
    angle: str,
    action_type: str,
    business_profile: BusinessProfile,
) -> str:
    bn = (business_profile.business_name or "").strip() or "Your business"
    city = infer_city_from_address(business_profile.business_address or "")
    region = _region_label_for_profile(business_profile, city)
    intent_phrase = _prompt_short_label(str(gap.get("prompt_text") or gap.get("prompt") or ""))
    comp_phr = _competitor_names_phrase(gap)
    dom = str(gap.get("source_domain") or "").strip()

    if action_type == "acquire_citation":
        if dom:
            if angle == ANGLE_PRESENCE_LISTINGS:
                return (
                    f"Strengthen {bn}'s verified presence on {dom} (and peers) so answers that cite directories can surface you alongside {comp_phr}."
                )
            if angle == ANGLE_CONTENT:
                return (
                    f"Echo the same services, areas, and proof points from your {dom} profile on a page you control so AI answers see consistent facts for questions like «{intent_phrase}»."
                )
            if angle == ANGLE_SCHEMA:
                return (
                    f"Add the same clear business details on your site (hours, what you offer, official website) that you show on {dom}, so {bn} matches everywhere."
                )
            if angle == ANGLE_COMPETITIVE_PARITY:
                return (
                    f"See what {comp_phr} show on {dom} (photos, services, service area) and add the same kind of clear facts for {bn} in {region}."
                )
            return (
                f"Tie {region} service-area and contact facts on your site to the same wording used on {dom} for {bn}."
            )

    # create_content / visibility
    if angle == ANGLE_CONTENT:
        return (
            f"Publish or tighten copy that directly answers «{intent_phrase}» for {bn} in {region}, citing specifics that differentiate you from {comp_phr}."
        )
    if angle == ANGLE_ON_PAGE:
        return (
            f"Improve headings, internal links, and a short FAQ on the page that best matches «{intent_phrase}» so crawlers and answer engines see a clear primary topic for {bn}."
        )
    if angle == ANGLE_SCHEMA:
        return (
            f"Put clear business details on the page that fits «{intent_phrase}»—hours, contact, services, and service area—so it matches what you say elsewhere online."
        )
    if angle == ANGLE_ENTITY_LOCATION:
        return (
            f"Align business name, address, phone, and service area with your live listings so {bn} is unambiguous for local questions in {region}."
        )
    if angle == ANGLE_COMPETITIVE_PARITY:
        return (
            f"Show the same kind of simple proof {comp_phr} show—certifications, years in business, or real outcomes—on your site for the same buyer questions."
        )
    return (
        f"Give answer engines clearer, verifiable details about {bn} for questions like «{intent_phrase}» in {region}."
    )


def _build_structured_actions(
    gap: dict[str, Any],
    angle: str,
    action_type: str,
    business_profile: BusinessProfile,
    nl_ctx: dict[str, Any],
    *,
    max_actions: int,
) -> list[dict[str, str]]:
    bn = (business_profile.business_name or "").strip() or "your business"
    region = (nl_ctx.get("region_label") or "").strip() or "your market"
    dom = str(gap.get("source_domain") or "").strip()
    comp_phr = _competitor_names_phrase(gap)

    def _pri(i: int) -> str:
        return "high" if i == 0 else ("medium" if i < 3 else "low")

    actions: list[dict[str, str]] = []

    if action_type == "acquire_citation":
        if angle == ANGLE_PRESENCE_LISTINGS:
            actions = [
                {
                    "title": f"Audit and complete your {dom or 'priority directory'} profile" if dom else "Audit top directory profiles",
                    "description": (
                        f"Match business name, phone, website, categories, and service area to what appears on {bn}'s site; fix outdated fields."
                        if dom
                        else f"Pick the top 3 directory domains that appear in AI answers for your niche and align {bn}'s listings with your website."
                    ),
                    "priority": "high",
                },
                {
                    "title": "Add proof-rich attributes",
                    "description": f"Fill attributes competitors use (licensing, years in business, service radius) so {bn} is comparable to {comp_phr}.",
                    "priority": "medium",
                },
                {
                    "title": "Request reviews on that platform",
                    "description": "Encourage recent, specific reviews that mention services you want to be known for in AI answers.",
                    "priority": "medium",
                },
            ]
        elif angle == ANGLE_CONTENT:
            actions = [
                {
                    "title": "Mirror listing facts on your site",
                    "description": f"Create a short block on your primary service page listing the same hours, areas, and services as {dom or 'your directory profiles'} for {bn}.",
                    "priority": "high",
                },
                {
                    "title": "Add a comparison or “why us” section",
                    "description": f"Explain differentiators vs {comp_phr} with concrete proof (certifications, process, guarantees)—avoid unverifiable superlatives.",
                    "priority": "medium",
                },
                {
                    "title": "Link out to authoritative profiles",
                    "description": f"From your footer or About page, link to the verified {dom or 'directory'} profile so models see the connection.",
                    "priority": "low",
                },
            ]
        elif angle == ANGLE_SCHEMA:
            actions = [
                {
                    "title": "Match business details to your listings",
                    "description": f"On your site, list the same business name, phone, website, hours, and service area as on {dom or 'your main listings'} for {bn}.",
                    "priority": "high",
                },
                {
                    "title": "Add a short FAQ buyers would ask",
                    "description": "Write 3–5 questions and answers about pricing, what you include, and where you work—use the same wording buyers use.",
                    "priority": "medium",
                },
                {
                    "title": "Double-check one page for mistakes",
                    "description": "Remove outdated phone numbers or old service names so every line matches your live listings.",
                    "priority": "low",
                },
            ]
        elif angle == ANGLE_COMPETITIVE_PARITY:
            actions = [
                {
                    "title": f"Benchmark {comp_phr} on {dom or 'key directories'}",
                    "description": "List attributes, media, and categories they use that you lack; prioritize the top two gaps.",
                    "priority": "high",
                },
                {
                    "title": "Close the highest-impact gap first",
                    "description": f"Implement the missing proof or category for {bn} before expanding to secondary platforms.",
                    "priority": "medium",
                },
                {
                    "title": "Document sources for claims",
                    "description": "Tie certifications and awards to verifiable pages or issuer sites.",
                    "priority": "low",
                },
            ]
        else:
            actions = [
                {
                    "title": "Unify NAP across site and listings",
                    "description": f"Ensure name/address/phone strings match exactly between {bn}'s website and {dom or 'trusted profiles'} in {region}.",
                    "priority": "high",
                },
                {
                    "title": "Clarify service area wording",
                    "description": "Use the same neighborhood/city names on site copy as in listings so local intent aligns.",
                    "priority": "medium",
                },
            ]
    else:
        # create_content
        if angle == ANGLE_CONTENT:
            actions = [
                {
                    "title": "Draft a page section that answers the buyer question directly",
                    "description": f"Lead with the outcome buyers want, then list scope, pricing signals, and next step for {bn} in {region}.",
                    "priority": "high",
                },
                {
                    "title": "Add evidence blocks",
                    "description": f"Include certifications, timelines, or case bullets that counter generic answers favoring {comp_phr}.",
                    "priority": "medium",
                },
                {
                    "title": "Refresh stale copy",
                    "description": "Update dates, offers, and service names so the page matches how people ask today.",
                    "priority": "medium",
                },
            ]
        elif angle == ANGLE_ON_PAGE:
            actions = [
                {
                    "title": "Rewrite the main heading and opening lines",
                    "description": f"Make the first heading and intro say plainly what you offer and for whom in {region}.",
                    "priority": "high",
                },
                {
                    "title": "Add internal links from related pages",
                    "description": f"Link from adjacent services or location pages so the target page is clearly central for {bn}.",
                    "priority": "medium",
                },
                {
                    "title": "Insert a tight FAQ (visible HTML)",
                    "description": "Three to five questions mirroring how people ask assistants, each with a factual answer.",
                    "priority": "medium",
                },
            ]
        elif angle == ANGLE_SCHEMA:
            actions = [
                {
                    "title": "Put clear business facts on the page",
                    "description": f"Show name, phone, hours, services, and service area in plain text on the page that fits this topic for {bn}.",
                    "priority": "high",
                },
                {
                    "title": "Link to your official profiles",
                    "description": "From your footer or About page, link to the Google Business Profile or other profiles you keep updated.",
                    "priority": "medium",
                },
                {
                    "title": "Remove duplicate or old blocks",
                    "description": "If the same business details appear twice with different info, keep one correct version.",
                    "priority": "low",
                },
            ]
        elif angle == ANGLE_ENTITY_LOCATION:
            actions = [
                {
                    "title": "Standardize business name everywhere",
                    "description": f"Pick one legal/trade style for {bn} and use it on site, footer, and listings.",
                    "priority": "high",
                },
                {
                    "title": "Publish service area explicitly",
                    "description": f"List cities/neighborhoods served—match wording used in local prompts for {region}.",
                    "priority": "medium",
                },
                {
                    "title": "Reconcile map pins and citations",
                    "description": "Fix mismatched addresses or closed locations that could confuse answer engines.",
                    "priority": "medium",
                },
            ]
        elif angle == ANGLE_COMPETITIVE_PARITY:
            actions = [
                {
                    "title": "Inventory what competitors showcase",
                    "description": f"From AI answers, note proof types {comp_phr} cite (licenses, awards, data); list gaps for {bn}.",
                    "priority": "high",
                },
                {
                    "title": "Add comparable proof",
                    "description": "Publish the same class of evidence (not copy) so assistants have parallel reasons to mention you.",
                    "priority": "medium",
                },
                {
                    "title": "Quantify outcomes where possible",
                    "description": "Use specific metrics, ranges, or timelines instead of vague quality claims.",
                    "priority": "low",
                },
            ]
        else:
            actions = [
                {
                    "title": "Clarify the primary conversion page",
                    "description": f"Ensure one page clearly owns this topic for {bn} in {region}.",
                    "priority": "high",
                },
            ]

    for i, row in enumerate(actions[:max_actions]):
        row["priority"] = _pri(i)
    return actions[:max_actions]


def _make_rec_id(
    action_type: str, gap: dict[str, Any], index: int, *, angle: str | None = None
) -> str:
    """Stable id for API/UI; include ``angle`` when emitting multi-angle leaves."""
    rs = gap.get("response_snapshot_id")
    ex = gap.get("extraction_snapshot_id")
    dom = str(gap.get("source_domain") or "").strip().lower().replace(":", "_")
    try:
        rs_i = int(rs) if rs is not None else -1
    except (TypeError, ValueError):
        rs_i = -1
    try:
        ex_i = int(ex) if ex is not None else -1
    except (TypeError, ValueError):
        ex_i = -1
    base = f"{action_type}:{rs_i}:{ex_i}:{dom}:{index}"
    if angle:
        return f"{base}:{angle}"
    return base


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
        f"Add or refresh a clear service page and a short FAQ for {region} with matching business details "
        f"so this type of question can mention {bn}."
    )
    uid = (gap.get("url_identity_summary") or "").strip()
    if uid:
        reason += f" {uid}"
    if score is not None and score.visibility_score < 20:
        reason += f" Overall visibility score for {bn} is {score.visibility_score:.1f}% (high priority)."
    elif score is not None:
        reason += f" Snapshot visibility score for {bn}: {score.visibility_score:.1f}%."
    comps = gap.get("competitors_in_answer") or []
    mpt_top = gap.get("matched_prompt_texts")
    if isinstance(mpt_top, list) and mpt_top:
        matched_prompt_texts = [str(p).strip() for p in mpt_top if str(p).strip()]
    else:
        grouped_prompts = gap.get("_grouped_prompts") if isinstance(gap.get("_grouped_prompts"), list) else []
        if grouped_prompts:
            matched_prompt_texts = [str(p).strip() for p in grouped_prompts if str(p).strip()]
        else:
            pt = str(gap.get("prompt_text") or gap.get("prompt") or "").strip()
            matched_prompt_texts = [pt] if pt else []
    grouped_resp = gap.get("_grouped_response_ids") if isinstance(gap.get("_grouped_response_ids"), list) else []
    matched_response_snapshot_ids: list[int] = []
    for rid in grouped_resp:
        try:
            v = int(rid)
        except (TypeError, ValueError):
            continue
        if v not in matched_response_snapshot_ids:
            matched_response_snapshot_ids.append(v)
    if not matched_response_snapshot_ids:
        rid_single = gap.get("response_snapshot_id")
        try:
            rid_int = int(rid_single) if rid_single is not None else None
        except (TypeError, ValueError):
            rid_int = None
        if rid_int is not None:
            matched_response_snapshot_ids.append(rid_int)
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
    mpt_top = gap.get("matched_prompt_texts")
    if isinstance(mpt_top, list) and mpt_top:
        matched_prompt_texts = [str(p).strip() for p in mpt_top if str(p).strip()]
    else:
        grouped_prompts = gap.get("_grouped_prompts") if isinstance(gap.get("_grouped_prompts"), list) else []
        if grouped_prompts:
            matched_prompt_texts = [str(p).strip() for p in grouped_prompts if str(p).strip()]
        else:
            matched_prompt_texts = []
    grouped_resp = gap.get("_grouped_response_ids") if isinstance(gap.get("_grouped_response_ids"), list) else []
    matched_response_snapshot_ids: list[int] = []
    for rid in grouped_resp:
        try:
            v = int(rid)
        except (TypeError, ValueError):
            continue
        if v not in matched_response_snapshot_ids:
            matched_response_snapshot_ids.append(v)
    if not matched_response_snapshot_ids:
        rid_single = gap.get("response_snapshot_id")
        try:
            rid_int = int(rid_single) if rid_single is not None else None
        except (TypeError, ValueError):
            rid_int = None
        if rid_int is not None:
            matched_response_snapshot_ids.append(rid_int)
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

    Used only when ``generate_aeo_recommendations(..., group_gaps=True)`` (e.g. summary export).
    Prompt-coverage and Actions use ungrouped gaps by default (``group_gaps=False``).
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
            row["matched_prompt_texts"] = [str(p).strip() for p in prompts if str(p).strip()]
            if len(prompts) == 1:
                row["prompt_text"] = prompts[0]
            else:
                row["prompt_text"] = f"{prompts[0]} (+{len(prompts) - 1} similar prompts)"
        row.pop("_grouped_prompts", None)
        row.pop("_grouped_prompt_count", None)
        out.append(row)
    return out


def _finalize_angle_recommendation(
    base: dict[str, Any],
    *,
    gap: dict[str, Any],
    angle: str,
    rec_id: str,
    parent_group_id: str,
    summary: str,
    actions: list[dict[str, str]],
    applies_prompts: list[str],
    applies_resp_ids: list[int],
    verbosity: str,
    nl_explanation: str,
) -> dict[str, Any]:
    rec = dict(base)
    old_refs = rec.get("references")
    if isinstance(old_refs, dict):
        rec["references"] = dict(old_refs)
    else:
        rec["references"] = {}
    rec["schema_version"] = 2
    rec["id"] = rec_id
    rec["rec_id"] = rec_id
    rec["parent_group_id"] = parent_group_id
    rec["summary"] = summary
    rec["angle"] = angle
    rec["verbosity"] = verbosity
    rec["actions"] = actions
    p_count = len(applies_prompts)
    if p_count == 0 and applies_resp_ids:
        p_count = len(applies_resp_ids)
    rec["applies_to"] = {
        "prompt_count": p_count,
        "prompt_examples": applies_prompts[:8],
        "response_snapshot_ids": applies_resp_ids[:50],
        "cluster_summary": _cluster_summary_line(gap),
    }
    refs = rec["references"]
    if applies_prompts:
        refs["matched_prompt_texts"] = applies_prompts[:50]
    if applies_resp_ids:
        refs["matched_response_snapshot_ids"] = applies_resp_ids[:50]
    rec["nl_explanation"] = nl_explanation
    return rec


_LOW_VALUE_ACTION_SNIPPETS: tuple[str, ...] = (
    "tie work to an existing",
    "crawl context",
    "topic seeds",
    "use crawl context",
    "onpage crawl",
)

# Higher wins when choosing which angle keeps a duplicate action title.
_ANGLE_PRECEDENCE_FOR_MERGE: dict[str, int] = {
    ANGLE_SCHEMA: 60,
    ANGLE_ON_PAGE: 50,
    ANGLE_ENTITY_LOCATION: 45,
    ANGLE_CONTENT: 40,
    ANGLE_COMPETITIVE_PARITY: 35,
    ANGLE_PRESENCE_LISTINGS: 30,
}


def _action_title_norm_key(title: str) -> str:
    s = re.sub(r"[^a-z0-9\s]+", " ", (title or "").lower())
    return " ".join(s.split())


def _is_low_value_action_row(action: dict[str, Any]) -> bool:
    t = f"{action.get('title') or ''} {action.get('description') or ''}".lower()
    return any(frag in t for frag in _LOW_VALUE_ACTION_SNIPPETS)


def _strategy_action_priority_rank(p: str | None) -> int:
    x = (p or "").lower()
    if x == "high":
        return 3
    if x == "medium":
        return 2
    if x == "low":
        return 1
    return 0


def _parse_cluster_kv(summaries: Iterable[str], prefix: str) -> set[str]:
    out: set[str] = set()
    pref = prefix.lower()
    for raw in summaries:
        if not isinstance(raw, str) or not raw.strip():
            continue
        for part in raw.split(";"):
            part = part.strip()
            low = part.lower()
            if low.startswith(pref):
                out.add(part.split(":", 1)[-1].strip().lower())
    return out


def _collect_strategy_signals(
    members: list[dict[str, Any]],
) -> tuple[Counter[str], Counter[str], set[str], set[str]]:
    action_ctr: Counter[str] = Counter()
    angle_ctr: Counter[str] = Counter()
    cluster_lines: list[str] = []
    for r in members:
        action_ctr[str(r.get("action_type") or "unknown")] += 1
        ang = r.get("angle")
        if isinstance(ang, str) and ang.strip():
            angle_ctr[ang.strip()] += 1
        ap = r.get("applies_to") if isinstance(r.get("applies_to"), dict) else {}
        cs = ap.get("cluster_summary")
        if isinstance(cs, str) and cs.strip():
            cluster_lines.append(cs.strip())
    intents = _parse_cluster_kv(cluster_lines, "intent:")
    focuses = _parse_cluster_kv(cluster_lines, "focus:")
    for r in members:
        ap = r.get("applies_to") if isinstance(r.get("applies_to"), dict) else {}
        for p in ap.get("prompt_examples") or []:
            if not isinstance(p, str) or not p.strip():
                continue
            faux: dict[str, Any] = {"prompt_text": p.strip()}
            intents.add(_derive_intent_type(faux))
            focuses.add(_derive_content_angle(faux))
    return action_ctr, angle_ctr, intents, focuses


def _buyer_topic_hints_from_members(members: list[dict[str, Any]], *, limit: int = 4) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in members:
        ap = r.get("applies_to") if isinstance(r.get("applies_to"), dict) else {}
        for p in ap.get("prompt_examples") or []:
            if not isinstance(p, str) or not p.strip():
                continue
            short = _prompt_short_label(p.strip(), max_words=5)
            if not short or short == "this type of question":
                continue
            k = short.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(short)
            if len(out) >= limit:
                return out
    return out


def _cap_words(title: str, max_words: int = 10) -> str:
    words = (title or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _strategy_short_title(
    members: list[dict[str, Any]],
    *,
    business_profile: BusinessProfile | None,
    action_ctr: Counter[str],
    angle_ctr: Counter[str],
    intents: set[str],
    focuses: set[str],
) -> str:
    bn = (business_profile.business_name or "").strip() if business_profile else ""
    hints = _buyer_topic_hints_from_members(members)
    topic = hints[0] if hints else ""

    cite_n = int(action_ctr.get("acquire_citation", 0))
    create_n = int(action_ctr.get("create_content", 0))
    dom = ""
    for r in members:
        if r.get("action_type") == "acquire_citation":
            s = r.get("source")
            if isinstance(s, str) and s.strip():
                dom = s.strip().split(".")[0].title()
                break

    top_angles = [a for a, _ in angle_ctr.most_common(3)]

    if cite_n > 0 and cite_n >= create_n:
        if dom:
            return _cap_words(f"Strengthen {dom} and Similar Listings", 10)
        return _cap_words("Strengthen Trusted Directory Profiles", 10)

    if ANGLE_SCHEMA in top_angles[:2] or angle_ctr.get(ANGLE_SCHEMA, 0) >= 2:
        if topic:
            return _cap_words(f"Clarify business details for {topic}", 10)
        return _cap_words("Clarify business details on key pages", 10)

    if ANGLE_ENTITY_LOCATION in top_angles[:2]:
        return _cap_words("Match your business name and contact info everywhere", 10)

    if ANGLE_COMPETITIVE_PARITY in top_angles[:2]:
        return _cap_words("Show proof like competitors do", 10)

    if ANGLE_ON_PAGE in top_angles[:2]:
        return _cap_words("Sharpen Headings and FAQs for Buyers", 10)

    if "local" in intents or "local_availability" in focuses:
        if topic:
            return _cap_words(f"Improve Local Content Around {topic}", 10)
        return _cap_words("Improve Content for Local Buyer Searches", 10)

    if "comparison" in intents or "comparison" in focuses:
        if topic:
            return _cap_words(f"Improve Comparison Content on {topic}", 10)
        return _cap_words("Improve Comparison and Options Content", 10)

    if "trust" in intents or "trust_proof" in focuses:
        return _cap_words("Add Trust and Credentials Buyers Can Verify", 10)

    if "transactional" in intents:
        if topic:
            return _cap_words(f"Clarify Offers and Next Steps for {topic}", 10)
        return _cap_words("Clarify Services Offers and Next Steps", 10)

    if topic:
        return _cap_words(f"Improve Content Covering {topic}", 10)

    if bn:
        return _cap_words(f"Help {bn} Show Up in More Answers", 10)
    return _cap_words("Improve Key Pages Buyers Ask About", 10)


def _strategy_outcome_summary(
    members: list[dict[str, Any]],
    *,
    business_profile: BusinessProfile | None,
    title: str,
    action_ctr: Counter[str],
    angle_ctr: Counter[str],
    intents: set[str],
) -> str:
    bn = (business_profile.business_name or "").strip() if business_profile else "your business"
    if not bn:
        bn = "your business"
    parts: list[str] = []

    if action_ctr.get("acquire_citation", 0) > 0 and action_ctr.get("create_content", 0) == 0:
        parts.append(
            f"When trusted directories list consistent facts about {bn}, assistants are more likely to mention you alongside competitors."
        )
    elif ANGLE_SCHEMA in angle_ctr:
        parts.append(
            f"When your pages spell out clear, matching details about {bn}, assistants can connect answers to the right business."
        )
    elif "local" in intents or ANGLE_ENTITY_LOCATION in angle_ctr:
        parts.append(
            f"Aligned name, area, and contact information make it easier for local questions to surface {bn}."
        )
    elif "comparison" in intents or ANGLE_COMPETITIVE_PARITY in angle_ctr:
        parts.append(
            f"Simple proof and clear differences give assistants reasons to mention {bn} when people compare options."
        )
    else:
        parts.append(
            f"Focused pages and FAQs that mirror how people ask help assistants confidently mention {bn}."
        )

    second = "Do the steps below in order; small steady updates usually help more than one big rush."
    return f"{parts[0]} {second}"


def _merge_action_rows(a: dict[str, str], b: dict[str, str], *, intent_note: str) -> dict[str, str]:
    out = dict(a)
    da = (out.get("description") or "").strip()
    db = (b.get("description") or "").strip()
    if len(db) > len(da):
        out["description"] = db
    elif intent_note and intent_note.lower() not in da.lower():
        out["description"] = f"{da.rstrip('.')}. Tailor this for {intent_note}.".strip()
    if _strategy_action_priority_rank(b.get("priority")) > _strategy_action_priority_rank(out.get("priority")):
        out["priority"] = b.get("priority") or out.get("priority") or "medium"
    return out


def _dedupe_and_bucket_actions_for_strategy(
    members: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """
    Returns angle -> deduped actions. Drops low-value rows; merges duplicate titles; picks angle by precedence.
    """
    raw_pairs: list[tuple[str, dict[str, str]]] = []
    intent_notes: list[str] = []
    for r in members:
        ang = str(r.get("angle") or "general").strip() or "general"
        acts = r.get("actions")
        if not isinstance(acts, list):
            continue
        ap = r.get("applies_to") if isinstance(r.get("applies_to"), dict) else {}
        cs = ap.get("cluster_summary")
        note = ""
        if isinstance(cs, str) and "intent:" in cs.lower():
            for seg in cs.split(";"):
                seg = seg.strip()
                if seg.lower().startswith("intent:"):
                    note = seg.split(":", 1)[-1].strip()
                    break
        for a in acts:
            if not isinstance(a, dict):
                continue
            title = str(a.get("title") or "").strip()
            desc = str(a.get("description") or "").strip()
            pr = str(a.get("priority") or "medium").strip()
            if not title or _is_low_value_action_row(a):
                continue
            raw_pairs.append((ang, {"title": title, "description": desc, "priority": pr}))
            if note:
                intent_notes.append(note)

    intent_hint = intent_notes[0] if intent_notes else ""

    # Merge by normalized title; keep higher angle precedence + richer description.
    merged: dict[str, dict[str, Any]] = {}
    order_keys: list[str] = []
    for ang, row in raw_pairs:
        k = _action_title_norm_key(row["title"])
        if not k:
            continue
        prec = _ANGLE_PRECEDENCE_FOR_MERGE.get(ang, 0)
        if k not in merged:
            merged[k] = {"angle": ang, "row": dict(row), "prec": prec}
            order_keys.append(k)
            continue
        cur = merged[k]
        new_prec = _ANGLE_PRECEDENCE_FOR_MERGE.get(ang, 0)
        if new_prec > cur["prec"]:
            cur["row"] = _merge_action_rows(dict(row), cur["row"], intent_note=intent_hint)
            cur["angle"] = ang
            cur["prec"] = new_prec
        else:
            cur["row"] = _merge_action_rows(cur["row"], dict(row), intent_note=intent_hint)

    by_angle: dict[str, list[dict[str, str]]] = defaultdict(list)
    for k in order_keys:
        bundle = merged.get(k)
        if not bundle:
            continue
        ang = str(bundle["angle"])
        row = dict(bundle["row"])
        by_angle[ang].append(row)

    return dict(by_angle)


def _aggregate_applies_to_for_strategy(
    members: list[dict[str, Any]],
    *,
    monitored_prompt_count: int | None,
) -> dict[str, Any]:
    prompts: list[str] = []
    resp_ids: list[int] = []
    seen_p: set[str] = set()
    seen_r: set[int] = set()
    max_pc = 0
    for r in members:
        ap = r.get("applies_to") if isinstance(r.get("applies_to"), dict) else {}
        try:
            pc = int(ap.get("prompt_count") or 0)
        except (TypeError, ValueError):
            pc = 0
        max_pc = max(max_pc, pc)
        for p in ap.get("prompt_examples") or []:
            if not isinstance(p, str) or not p.strip():
                continue
            k = " ".join(p.split()).strip().lower()
            if k in seen_p:
                continue
            seen_p.add(k)
            prompts.append(p.strip())
        for x in ap.get("response_snapshot_ids") or []:
            try:
                v = int(x)
            except (TypeError, ValueError):
                continue
            if v not in seen_r:
                seen_r.add(v)
                resp_ids.append(v)
    unique_prompt_n = len(seen_p)
    if unique_prompt_n == 0 and max_pc > 0:
        unique_prompt_n = max_pc
    elif unique_prompt_n < max_pc:
        unique_prompt_n = max(unique_prompt_n, max_pc)

    out: dict[str, Any] = {
        "prompt_count": unique_prompt_n,
        "prompt_examples": prompts[:3],
        "response_snapshot_ids": resp_ids[:40],
    }
    if monitored_prompt_count and monitored_prompt_count > 0 and unique_prompt_n > 0:
        out["coverage_fraction"] = round(min(1.0, unique_prompt_n / monitored_prompt_count), 3)
    return out


def _strategy_priority_label(members: list[dict[str, Any]]) -> str:
    best = "low"
    for r in members:
        p = str(r.get("priority") or "medium").lower()
        if p == "high":
            return "high"
        if p == "medium" and best == "low":
            best = "medium"
    return best


def build_recommendation_strategies_from_flat(
    flat_recommendations: list[dict[str, Any]],
    *,
    business_profile: BusinessProfile | None = None,
    monitored_prompt_count: int | None = None,
) -> list[dict[str, Any]]:
    """
    UI-ready hierarchical strategies: one entry per ``parent_group_id``, deduped actions,
    short titles, plain-language summaries. Granular rows stay in ``flat_recommendations``.
    """
    if not flat_recommendations:
        return []

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in flat_recommendations:
        if not isinstance(rec, dict):
            continue
        pid = str(rec.get("parent_group_id") or "").strip()
        if not pid:
            rid = str(rec.get("rec_id") or rec.get("id") or "").strip()
            pid = f"aeo_single::{rid}" if rid else "aeo_single::unknown"
        by_parent[pid].append(rec)

    strategies: list[dict[str, Any]] = []
    for strategy_id, members in by_parent.items():
        if not members:
            continue
        action_ctr, angle_ctr, intents, focuses = _collect_strategy_signals(members)
        title = _strategy_short_title(
            members,
            business_profile=business_profile,
            action_ctr=action_ctr,
            angle_ctr=angle_ctr,
            intents=intents,
            focuses=focuses,
        )
        summary = _strategy_outcome_summary(
            members,
            business_profile=business_profile,
            title=title,
            action_ctr=action_ctr,
            angle_ctr=angle_ctr,
            intents=intents,
        )
        angle_buckets = _dedupe_and_bucket_actions_for_strategy(members)
        flat_actions: list[dict[str, str]] = []
        seen_t: set[str] = set()
        for ang, acts in sorted(
            angle_buckets.items(),
            key=lambda kv: _ANGLE_PRECEDENCE_FOR_MERGE.get(kv[0], 0),
            reverse=True,
        ):
            for act in acts:
                k = _action_title_norm_key(str(act.get("title") or ""))
                if not k or k in seen_t:
                    continue
                seen_t.add(k)
                flat_actions.append(dict(act))
                if len(flat_actions) >= 12:
                    break
            if len(flat_actions) >= 12:
                break
        angles_out = (
            [{"angle": ANGLE_FLAT_TODO, "actions": flat_actions}] if flat_actions else []
        )
        applies = _aggregate_applies_to_for_strategy(members, monitored_prompt_count=monitored_prompt_count)
        leaf_ids = []
        for r in members:
            rid = r.get("rec_id") or r.get("id")
            if rid is not None and str(rid).strip():
                leaf_ids.append(str(rid).strip())
        strategies.append(
            {
                "strategy_id": strategy_id,
                "title": title,
                "summary": summary,
                "priority": _strategy_priority_label(members),
                "action_types": sorted(action_ctr.keys()),
                "applies_to": applies,
                "angles": angles_out,
                "source_leaf_ids": leaf_ids[:80],
            }
        )

    strategies.sort(
        key=lambda s: (
            -_strategy_action_priority_rank(str(s.get("priority"))),
            -int((s.get("applies_to") or {}).get("prompt_count") or 0),
            s.get("title") or "",
        )
    )
    return strategies


def save_recommendation_run(
    business_profile: BusinessProfile,
    *,
    score_snapshot: AEOScoreSnapshot | None,
    recommendations: list[dict[str, Any]],
    strategies: list[dict[str, Any]] | None = None,
    visibility_score: float,
    weighted_position_score: float,
    citation_share: float,
) -> AEORecommendationRun:
    return AEORecommendationRun.objects.create(
        profile=business_profile,
        score_snapshot=score_snapshot,
        recommendations_json=recommendations,
        strategies_json=strategies if strategies is not None else [],
        visibility_score_at_run=visibility_score,
        weighted_position_score_at_run=weighted_position_score,
        citation_share_at_run=citation_share,
    )


def generate_aeo_recommendations(
    business_profile: BusinessProfile,
    *,
    save: bool = True,
    enrich_with_nl: bool = True,
    group_gaps: bool = False,
    verbosity: str = AEO_RECOMMENDATION_VERBOSITY_COMPACT,
    multi_angle: bool = False,
) -> dict[str, Any]:
    """
    Build a short prioritized to-do list (max :data:`AEO_RECOMMENDATION_MAX_LEAVES` items) from the
    latest score snapshot and latest extraction per response.

    Default output is one row per gap with plain-language copy (multi-angle fan-out is opt-in).
    Strategies flatten actions into a single list for the UI (no Schema/Content section headings).

    OpenAI NL runs when ``enrich_with_nl`` is True (capped by the max-leaves limit).
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

    verb = (
        verbosity
        if verbosity in AEO_RECOMMENDATION_VERBOSITY_CHOICES
        else AEO_RECOMMENDATION_VERBOSITY_EXPANDED
    )

    canonical_dom = canonical_registrable_domain(site)
    vis_gaps_raw = analyze_visibility_gaps(
        score,
        extractions,
        tracked_website_url=site,
        canonical_domain=canonical_dom,
    )
    cite_gaps_raw = analyze_citation_gaps(
        score, extractions, citation_share=citation, verbosity=verb
    )
    if verb == AEO_RECOMMENDATION_VERBOSITY_EXPANDED and citation < 30.0:
        excluded_domains = _competitor_domains_from_extractions(extractions)
        own_domain = canonical_registrable_domain(site)
        if own_domain:
            excluded_domains.add(own_domain)
        per_resp_cites = _analyze_citation_gaps_per_response(
            score,
            extractions,
            citation_share=citation,
            excluded_domains=excluded_domains,
        )
        cite_gaps_raw = _merge_citation_gaps_deduped(cite_gaps_raw, per_resp_cites)

    if group_gaps:
        vis_gaps = _group_gap_objects_for_recommendations(vis_gaps_raw, action_type="create_content")
        cite_gaps = _group_gap_objects_for_recommendations(cite_gaps_raw, action_type="acquire_citation")
    else:
        vis_gaps = vis_gaps_raw
        cite_gaps = cite_gaps_raw

    recommendations: list[dict[str, Any]] = []
    nl_ctx = _recommendation_nl_enrichment(business_profile, score=score)
    use_openai_nl = bool(enrich_with_nl)
    rec_seq = 0

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

    for _vis_i, gap in enumerate(vis_gaps):
        if len(recommendations) >= AEO_RECOMMENDATION_MAX_LEAVES:
            break
        base = _build_create_content_recommendation(
            gap,
            score=score,
            priority=priority,
            business_profile=business_profile,
        )
        root_key = _root_issue_key(gap, "create_content")
        parent_id = _stable_parent_group_id(root_key)
        applies_prompts, applies_resp_ids = _applies_to_prompts_and_response_ids(gap)
        n_act = _action_count_for_verbosity(verb)

        if multi_angle:
            for angle in _angles_for_visibility_gap(gap, verbosity=verb):
                summary = _build_angle_summary(gap, angle, "create_content", business_profile)
                actions = _build_structured_actions(
                    gap,
                    angle,
                    "create_content",
                    business_profile,
                    nl_ctx,
                    max_actions=n_act,
                )
                rid = _make_rec_id("create_content", gap, rec_seq, angle=angle)
                rec_seq += 1
                nl_line = summary
                fin = _finalize_angle_recommendation(
                    base,
                    gap=gap,
                    angle=angle,
                    rec_id=rid,
                    parent_group_id=parent_id,
                    summary=summary,
                    actions=actions,
                    applies_prompts=applies_prompts,
                    applies_resp_ids=applies_resp_ids,
                    verbosity=verb,
                    nl_explanation=nl_line,
                )
                _apply_nl_as_primary_actions(fin)
                recommendations.append(fin)
                if len(recommendations) >= AEO_RECOMMENDATION_MAX_LEAVES:
                    break
        else:
            rid = _make_rec_id("create_content", gap, rec_seq)
            rec_seq += 1
            rec = dict(base)
            br = rec.get("references")
            if isinstance(br, dict):
                rec["references"] = dict(br)
            rec["schema_version"] = 2
            rec["id"] = rid
            rec["rec_id"] = rid
            rec["parent_group_id"] = parent_id
            rec["angle"] = ANGLE_CONTENT
            rec["verbosity"] = verb
            rec["summary"] = _build_angle_summary(gap, ANGLE_CONTENT, "create_content", business_profile)
            rec["actions"] = _build_structured_actions(
                gap,
                ANGLE_CONTENT,
                "create_content",
                business_profile,
                nl_ctx,
                max_actions=n_act,
            )
            rec["applies_to"] = {
                "prompt_count": len(applies_prompts),
                "prompt_examples": applies_prompts[:8],
                "response_snapshot_ids": applies_resp_ids[:50],
                "cluster_summary": _cluster_summary_line(gap),
            }
            if use_openai_nl:
                rec["nl_explanation"] = generate_natural_language_recommendation(
                    {
                        **gap,
                        **nl_ctx,
                        "action_type": "create_content",
                        "score": _headline_score_dict(),
                    },
                    business_profile=business_profile,
                )
            else:
                rec["nl_explanation"] = rec["summary"]
            _apply_nl_as_primary_actions(rec)
            recommendations.append(rec)
            if len(recommendations) >= AEO_RECOMMENDATION_MAX_LEAVES:
                break

    for _cite_i, gap in enumerate(cite_gaps):
        if len(recommendations) >= AEO_RECOMMENDATION_MAX_LEAVES:
            break
        base = _build_acquire_citation_recommendation(
            gap,
            score=score,
            priority=priority,
            business_profile=business_profile,
        )
        root_key = _root_issue_key(gap, "acquire_citation")
        parent_id = _stable_parent_group_id(root_key)
        applies_prompts, applies_resp_ids = _applies_to_prompts_and_response_ids(gap)
        n_act = _action_count_for_verbosity(verb)

        if multi_angle:
            for angle in _angles_for_citation_gap(verbosity=verb):
                summary = _build_angle_summary(gap, angle, "acquire_citation", business_profile)
                actions = _build_structured_actions(
                    gap,
                    angle,
                    "acquire_citation",
                    business_profile,
                    nl_ctx,
                    max_actions=n_act,
                )
                rid = _make_rec_id("acquire_citation", gap, rec_seq, angle=angle)
                rec_seq += 1
                fin = _finalize_angle_recommendation(
                    base,
                    gap=gap,
                    angle=angle,
                    rec_id=rid,
                    parent_group_id=parent_id,
                    summary=summary,
                    actions=actions,
                    applies_prompts=applies_prompts,
                    applies_resp_ids=applies_resp_ids,
                    verbosity=verb,
                    nl_explanation=summary,
                )
                _apply_nl_as_primary_actions(fin)
                recommendations.append(fin)
                if len(recommendations) >= AEO_RECOMMENDATION_MAX_LEAVES:
                    break
        else:
            rid = _make_rec_id("acquire_citation", gap, rec_seq)
            rec_seq += 1
            rec = dict(base)
            br = rec.get("references")
            if isinstance(br, dict):
                rec["references"] = dict(br)
            rec["schema_version"] = 2
            rec["id"] = rid
            rec["rec_id"] = rid
            rec["parent_group_id"] = parent_id
            rec["angle"] = ANGLE_PRESENCE_LISTINGS
            rec["verbosity"] = verb
            rec["summary"] = _build_angle_summary(
                gap, ANGLE_PRESENCE_LISTINGS, "acquire_citation", business_profile
            )
            rec["actions"] = _build_structured_actions(
                gap,
                ANGLE_PRESENCE_LISTINGS,
                "acquire_citation",
                business_profile,
                nl_ctx,
                max_actions=n_act,
            )
            rec["applies_to"] = {
                "prompt_count": len(applies_prompts),
                "prompt_examples": applies_prompts[:8],
                "response_snapshot_ids": applies_resp_ids[:50],
                "cluster_summary": _cluster_summary_line(gap),
            }
            if use_openai_nl:
                rec["nl_explanation"] = generate_natural_language_recommendation(
                    {
                        **gap,
                        **nl_ctx,
                        "action_type": "acquire_citation",
                        "score": _headline_score_dict(cite_gap=gap),
                    },
                    business_profile=business_profile,
                )
            else:
                rec["nl_explanation"] = rec["summary"]
            _apply_nl_as_primary_actions(rec)
            recommendations.append(rec)
            if len(recommendations) >= AEO_RECOMMENDATION_MAX_LEAVES:
                break

    recommendations = recommendations[:AEO_RECOMMENDATION_MAX_LEAVES]

    if not recommendations:
        bn = (business_profile.business_name or "").strip() or "This business"
        reason = (
            f"No single prompt gap fired for {bn} this run (visibility {visibility:.1f}%, citation-style "
            f"{citation:.1f}%). After you update FAQs, business details on key pages, or main listings, run another check."
        )
        low_rec: dict[str, Any] = {
            "action_type": "review_visibility",
            "reason": reason,
            "priority": "low" if priority == "low" else priority,
            "references": {
                "score_snapshot_id": score.id if score else None,
                "extraction_snapshot_id": None,
                "response_snapshot_id": None,
                "competitors": _top_competitors_from_score(score),
            },
        }
        low_summary = reason
        if use_openai_nl:
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
        else:
            low_rec["nl_explanation"] = low_summary
        _apply_nl_as_primary_actions(low_rec)
        low_id = "review_visibility:0"
        low_rec["rec_id"] = low_id
        low_rec["id"] = low_id
        low_rec["schema_version"] = 2
        low_rec["parent_group_id"] = _stable_parent_group_id("review_visibility|fallback")
        low_rec["summary"] = low_summary
        low_rec["angle"] = ANGLE_ON_PAGE
        low_rec["verbosity"] = verb
        low_rec["actions"] = [
            {
                "title": "Refresh FAQs on your top service pages",
                "description": "Add 3–5 questions buyers actually ask; keep answers factual and aligned with listings.",
                "priority": "high",
            },
            {
                "title": "Check your business details everywhere",
                "description": "Make sure name, address, phone, and services read the same on your site and on major listings.",
                "priority": "medium",
            },
            {
                "title": "Run another check after you update",
                "description": "When your pages and listings are updated, run monitoring again to see new AI answers.",
                "priority": "low",
            },
        ]
        low_rec["applies_to"] = {
            "prompt_count": 0,
            "prompt_examples": [],
            "response_snapshot_ids": [],
            "cluster_summary": "no_specific_gap",
        }
        recommendations.append(low_rec)

    monitored_n = len(
        [str(x).strip() for x in (business_profile.selected_aeo_prompts or []) if str(x).strip()]
    )
    strategies = build_recommendation_strategies_from_flat(
        recommendations,
        business_profile=business_profile,
        monitored_prompt_count=monitored_n or None,
    )

    run_id: int | None = None
    if save:
        run = save_recommendation_run(
            business_profile,
            score_snapshot=score,
            recommendations=recommendations,
            strategies=strategies,
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
        "strategies": strategies,
        "verbosity": verb,
        "multi_angle": multi_angle,
    }
