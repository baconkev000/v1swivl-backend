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
from .aeo_extraction_utils import brand_effectively_cited
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


def analyze_visibility_gaps(
    score_snapshot: AEOScoreSnapshot | None,
    extraction_snapshots: list[AEOExtractionSnapshot],
    *,
    tracked_website_url: str = "",
) -> list[dict[str, Any]]:
    """
    Gaps where the target brand did not appear in the modeled answer for a prompt.

    Each gap carries ids for traceability to extraction / response rows.
    """
    gaps: list[dict[str, Any]] = []
    for ex in extraction_snapshots:
        if brand_effectively_cited(
            bool(ex.brand_mentioned),
            ex.competitors_json,
            tracked_website_url_or_domain=tracked_website_url,
        ):
            continue
        resp = ex.response_snapshot
        gaps.append(
            {
                "gap_kind": "visibility_miss",
                "prompt_text": (resp.prompt_text or "").strip(),
                "extraction_snapshot_id": ex.id,
                "response_snapshot_id": resp.id,
                "competitors_in_answer": list(ex.competitors_json or [])
                if isinstance(ex.competitors_json, list)
                else [],
            }
        )
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
    out: dict[str, Any] = {
        "business_name": (profile.business_name or "").strip(),
        "city": city,
        "region_label": region_label,
        "industry": (profile.industry or "").strip(),
        "services": "",  # placeholder when profile gains structured services
        "onpage_crawl_summary": onpage_summary,
    }
    if score is not None:
        comps = _top_competitors_from_score(score, limit=5)
        if comps:
            out["competitors"] = ", ".join(comps)
    return out


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
        f"No single prompt gap fired on the latest run; visibility is about {vis:.0f}% and citation-style "
        f"share about {cite:.0f}%. Keep monitoring and refresh entity + FAQ signals for {location} so "
        f"{bn} stays easy for answer engines to surface."
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
    ind_suffix = f" in the {industry} category" if industry else ""
    svc_clause = f" Mention {services} plainly where it matches the query." if services else ""

    if kind == "visibility_miss":
        prompt = (gap.get("prompt_text") or "this type of question").strip()
        if len(prompt) > 140:
            prompt = prompt[:137].rstrip() + "…"
        comps = gap.get("competitors_in_answer") or []
        comp_str = ", ".join(_competitor_display_names(comps if isinstance(comps, list) else [], limit=8)[:3])
        if comp_str:
            return (
                f"For “{prompt}”{ind_suffix}, modeled answers in {region} already name {comp_str}; tighten FAQs, "
                f"service pages, and entity/schema so {bn} is cited for the same intent.{svc_clause}"
            )
        bench = (gap.get("competitors") or "").strip()
        if bench:
            if len(bench) > 85:
                bench = bench[:82].rstrip() + "…"
            return (
                f"For “{prompt}”{ind_suffix}, publish concise, location-grounded copy in {region} (offers, FAQs, "
                f"Organization/LocalBusiness signals) so {bn} can appear when this question is answered—"
                f"your score snapshot also surfaces {bench} as frequent comparables.{svc_clause}"
            )
        return (
            f"For “{prompt}”{ind_suffix}, publish concise, location-grounded copy in {region} (offers, FAQs, "
            f"Organization/LocalBusiness signals) so {bn} can appear when this question is answered.{svc_clause}"
        )

    if kind == "citation_share":
        dom = gap.get("source_domain")
        try:
            cite = float(gap.get("citation_share_at_check") or 0)
        except (TypeError, ValueError):
            cite = 0.0
        comp = (gap.get("top_competitor_hint") or "named competitors").strip()
        if dom:
            cite_tail = f" Reflect {services} in public profiles where accurate." if services else ""
            return (
                f"Citation-style share is about {cite:.0f}%—pursue verifiable mentions on {dom} and similar trusted "
                f"sources for {region}, where {comp} may already earn more model-visible citations than {bn}.{cite_tail}"
            )
        return _nl_template({**gap, "gap_kind": "citation_share_generic"})

    if kind == "citation_share_generic":
        try:
            cite = float(gap.get("citation_share_at_check") or 0)
        except (TypeError, ValueError):
            cite = 0.0
        comp = (gap.get("top_competitor_hint") or "alternatives").strip()
        cite_tail = f" Reflect {services} in titles or blurbs where truthful." if services else ""
        return (
            f"Citation-style share is about {cite:.0f}%—earn listings, profiles, or editorial mentions on authoritative "
            f"domains tied to {region} and your sector so models cite {bn} alongside {comp}.{cite_tail}"
        )

    return (
        f"Review latest AEO snapshots for {region}: strengthen on-page clarity, third-party mentions, and entity "
        f"consistency so {bn} stays visible across transactional, trust, and comparison-style queries."
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
    Keys: prompt, action_type, competitors, business_name, region, gap_kind, score, crawl_summary.
    """
    prompt = (gap_object.get("prompt_text") or gap_object.get("prompt") or "").strip()
    bn = (gap_object.get("business_name") or "").strip()
    region = (gap_object.get("region_label") or gap_object.get("region") or gap_object.get("city") or "").strip()
    crawl_summary = (gap_object.get("onpage_crawl_summary") or gap_object.get("crawl_summary") or "").strip()
    gk = gap_object.get("gap_kind")
    gap_kind = str(gk).strip() if gk is not None else ""
    return {
        "prompt": prompt,
        "action_type": _infer_action_type_for_nl(gap_object),
        "competitors": _competitor_strings_for_nl(gap_object),
        "business_name": bn,
        "region": region,
        "gap_kind": gap_kind,
        "score": _normalize_score_for_nl(gap_object),
        "crawl_summary": crawl_summary[:8000],
    }


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
        "Write exactly 2 sentences:\n"
        "1) specific website/content action\n"
        "2) why this closes the AEO gap for this query\n"
        "Avoid repeating the JSON wording."
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
    industry = (business_profile.industry or "").strip()
    ind_bit = f" ({industry})" if industry else ""
    prompt_display = gap.get("prompt_text") or ""
    reason = (
        f"Visibility gap: {bn} was not mentioned in the captured answer for this query{ind_bit}. "
        f"Improve content, entity consistency, and local relevance for {region} so transactional and trust queries "
        f"can surface {bn} alongside named alternatives."
    )
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
    dom = gap.get("source_domain")
    try:
        cite = float(gap.get("citation_share_at_check") or 0)
    except (TypeError, ValueError):
        cite = 0.0
    comp = gap.get("top_competitor_hint")
    bn = (business_profile.business_name or "").strip() or "This business"
    if dom:
        reason = (
            f"Citation gap: modeled citation-style share is {cite:.1f}%; earn verifiable mentions on {dom} "
            f"and comparable authoritative domains so answer engines can cite {bn}."
        )
    else:
        reason = (
            f"Citation gap: modeled citation-style share is {cite:.1f}%; pursue trusted third-party listings, "
            f"profiles, and editorial references your category relies on so models can cite {bn} in AI answers."
        )
    if comp:
        reason += f" Recent answers often name {comp}—mirror the citation patterns those sources reward so {bn} is included."
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

    vis_gaps = analyze_visibility_gaps(score, extractions, tracked_website_url=site)
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
                f"No prompt-level visibility or citation gaps were flagged on this pass for {bn}; headline scores are "
                f"visibility {visibility:.1f}% and citation-style share {citation:.1f}%. Re-run after content or "
                f"off-site citation updates so regressions surface early."
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
