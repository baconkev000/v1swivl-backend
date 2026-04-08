"""
Phase 4: deterministic AEO metrics from extraction snapshots.

Formulas are explicit (no hidden multipliers). Imports only flatten helpers from
``aeo_extraction_utils`` for competitor JSON shapes.

Phase 2 can create multiple ``AEOResponseSnapshot`` rows per prompt (OpenAI, Gemini, Perplexity when configured).
Headline Phase-4 scores use **OpenAI responses only** (``platform="openai"``) so prompts are not double-counted;
Gemini and Perplexity extractions still power prompt-coverage UI and share-of-voice when using
``latest_extraction_per_response(..., response_platform=None)``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Protocol, Sequence

from django.db.models import Prefetch

from ..models import (
    AEOPromptExecutionAggregate,
    AEOExtractionSnapshot,
    AEOScoreSnapshot,
    AEOResponseSnapshot,
    BusinessProfile,
)
from .aeo_extraction_utils import (
    _normalize_for_brand_match,
    brand_effectively_cited,
    parse_competitor_raw_item,
    registered_root_domains_match,
    root_domain_from_fragment,
)

PRIMARY_AEO_SCORING_PLATFORM = "openai"


def _competitor_entry_name(raw: Any) -> str:
    """Display name from a stored competitor (dict, JSON/repr string, or legacy string)."""
    return parse_competitor_raw_item(raw)["name"]


def _competitor_entry_url(raw: Any) -> str:
    return parse_competitor_raw_item(raw)["url"][:2048]


class _ExtractionLike(Protocol):
    brand_mentioned: bool
    mention_position: str
    mention_count: int
    competitors_json: list


_POSITION_WEIGHTS: dict[str, float] = {
    AEOExtractionSnapshot.MENTION_TOP: 1.0,
    AEOExtractionSnapshot.MENTION_MIDDLE: 0.5,
    AEOExtractionSnapshot.MENTION_BOTTOM: 0.25,
    AEOExtractionSnapshot.MENTION_NONE: 0.0,
}


def calculate_visibility_score(
    extractions: Sequence[_ExtractionLike],
    *,
    tracked_website_url: str = "",
) -> float:
    """
    Percentage of prompts where the target brand was mentioned.

    Formula: (count where brand_mentioned) / (total prompts) × 100

    Counts a row when ``brand_mentioned`` is true or the profile website matches a competitor URL
    in ``competitors_json`` (models often list the target only there).

    Rows with ``brand_mentioned_url_status`` in ``mentioned_url_wrong_*`` still count as *not* cited
    here (``brand_mentioned`` remains false) until product opts into partial credit; the status field
    is for UI/diagnostics only.
    """
    n = len(extractions)
    if n == 0:
        return 0.0
    mentioned = sum(
        1
        for e in extractions
        if brand_effectively_cited(
            bool(getattr(e, "brand_mentioned", False)),
            getattr(e, "competitors_json", None),
            tracked_website_url_or_domain=tracked_website_url,
        )
    )
    return round(100.0 * mentioned / n, 4)


def calculate_weighted_position_score(
    extractions: Sequence[_ExtractionLike],
    *,
    tracked_website_url: str = "",
) -> float:
    """
    Average positional weight across prompts, expressed as a percentage of the max (1.0).

    Weights (only applied when brand_mentioned is true; otherwise 0 for that row):
        top = 1.0, middle = 0.5, bottom = 0.25, none = 0

    Formula: (sum of per-prompt weights) / (total prompts) × 100
    """
    n = len(extractions)
    if n == 0:
        return 0.0
    total_w = 0.0
    for e in extractions:
        if not brand_effectively_cited(
            bool(getattr(e, "brand_mentioned", False)),
            getattr(e, "competitors_json", None),
            tracked_website_url_or_domain=tracked_website_url,
        ):
            continue
        pos = (e.mention_position or AEOExtractionSnapshot.MENTION_NONE).lower()
        total_w += _POSITION_WEIGHTS.get(pos, 0.0)
    return round(100.0 * total_w / n, 4)


def calculate_citation_share(extractions: Sequence[_ExtractionLike]) -> float:
    """
    Target share of modeled \"named business\" mention units across extractions.

    Numerator: sum of mention_count (target brand mentions from extraction).
    Denominator: that sum plus one unit per competitor entry listed per extraction.

    Formula: (target mention units) / (target + competitor mention units) × 100

    Returns 0 when the denominator is 0.
    """
    target_units = 0
    competitor_units = 0
    for e in extractions:
        try:
            target_units += max(0, int(e.mention_count))
        except (TypeError, ValueError):
            pass
        comps = getattr(e, "competitors_json", None) or []
        if isinstance(comps, list):
            competitor_units += len(comps)
    total = target_units + competitor_units
    if total == 0:
        return 0.0
    return round(100.0 * target_units / total, 4)


def calculate_competitor_dominance(extractions: Sequence[_ExtractionLike]) -> dict[str, Any]:
    """
    Frequency of named competitors across extractions (case-insensitive key, display name preserved).

    Returns:
        ranked: list of {\"competitor\": str, \"count\": int} descending by count
        total_competitor_mentions: int (sum of counts)
    """
    counter: Counter[str] = Counter()
    display: dict[str, str] = {}
    for e in extractions:
        comps = getattr(e, "competitors_json", None) or []
        if not isinstance(comps, list):
            continue
        for raw in comps:
            name = _competitor_entry_name(raw)
            if not name:
                continue
            key = name.casefold()
            counter[key] += 1
            display.setdefault(key, name)
    ranked = [
        {"competitor": display[k], "count": c}
        for k, c in counter.most_common()
    ]
    return {
        "ranked": ranked,
        "total_competitor_mentions": int(sum(counter.values())),
    }


_COMP_SOV_COLORS = ("#1d1d1f", "#636366", "#9CA3AF")


def aggregate_aeo_share_of_voice_from_extractions(
    extractions: Sequence[_ExtractionLike],
    *,
    business_display_name: str,
    business_website_url: str = "",
) -> dict[str, Any]:
    """
    Share of named-mention units across latest extractions (one per prompt).

    Denominator matches ``calculate_citation_share``: your_units = sum(mention_count);
    competitor units = one per entry in each extraction's ``competitors_json`` list.

    Returns UI rows: you, top 3 competitors by mention count, then Others (remaining
    competitor mentions). Percentages sum to ~100 when total_units > 0.
    """
    n = len(extractions)
    site = (business_website_url or "").strip()
    target_root = root_domain_from_fragment(site) or ""
    your_units = 0
    for e in extractions:
        try:
            mc = max(0, int(e.mention_count))
        except (TypeError, ValueError):
            mc = 0
        your_units += mc
        if mc == 0 and brand_effectively_cited(
            bool(getattr(e, "brand_mentioned", False)),
            getattr(e, "competitors_json", None),
            tracked_website_url_or_domain=site,
        ):
            your_units += 1

    counter: Counter[str] = Counter()
    display: dict[str, str] = {}
    url_by_name_key: dict[str, str] = {}
    for e in extractions:
        comps = getattr(e, "competitors_json", None) or []
        if not isinstance(comps, list):
            continue
        for raw in comps:
            parsed = parse_competitor_raw_item(raw)
            name = (parsed.get("name") or "").strip()
            if not name:
                continue
            dom = root_domain_from_fragment(parsed.get("url") or "") or ""
            if target_root and dom and registered_root_domains_match(target_root, dom):
                continue
            key = name.casefold()
            counter[key] += 1
            display.setdefault(key, name)
            u = (parsed.get("url") or "").strip()[:2048]
            if u and key not in url_by_name_key:
                url_by_name_key[key] = u

    competitor_total = int(sum(counter.values()))
    total_units = your_units + competitor_total
    top3 = counter.most_common(3)
    top3_keys = {k for k, _ in top3}
    others_units = sum(cnt for k, cnt in counter.items() if k not in top3_keys)

    def pct(units: int) -> float:
        if total_units <= 0:
            return 0.0
        return round(100.0 * float(units) / float(total_units), 1)

    label_you = (business_display_name or "").strip() or "Your business"
    you_url = (business_website_url or "").strip()[:2048]

    rows: list[dict[str, Any]] = [
        {
            "name": label_you,
            "url": you_url,
            "pct": pct(your_units),
            "units": your_units,
            "you": True,
            "color": "#000000",
        }
    ]
    for i, (key, cnt) in enumerate(top3):
        rows.append(
            {
                "name": display[key],
                "url": url_by_name_key.get(key, ""),
                "pct": pct(cnt),
                "units": int(cnt),
                "you": False,
                "color": _COMP_SOV_COLORS[i],
            }
        )
    if others_units > 0:
        rows.append(
            {
                "name": "Others",
                "url": "",
                "pct": pct(others_units),
                "units": others_units,
                "you": False,
                "color": "#E5E7EB",
            }
        )

    return {
        "total_prompts": n,
        "total_mention_units": total_units,
        "your_mention_units": your_units,
        "competitor_mention_units": competitor_total,
        "has_data": total_units > 0,
        "rows": rows,
    }


def aggregate_aeo_share_of_voice(business_profile: BusinessProfile) -> dict[str, Any]:
    # Share-of-voice reflects all monitored providers (OpenAI, Gemini, Perplexity, …).
    # Phase-4 headline scoring remains OpenAI-primary via default platform filtering elsewhere.
    extractions = latest_extraction_per_response(business_profile, response_platform=None)
    name = (getattr(business_profile, "business_name", None) or "").strip() or "Your business"
    site = (getattr(business_profile, "website_url", None) or "").strip()
    return aggregate_aeo_share_of_voice_from_extractions(
        extractions,
        business_display_name=name,
        business_website_url=site,
    )


def _visibility_brand_key(name: str) -> str:
    """Collapse brand variants using the same normalization as entity / brand matching."""
    return _normalize_for_brand_match((name or "").strip())


def _pick_three_onboarding_competitor_rows(
    full_sorted: list[dict[str, Any]],
    target_idx: int,
) -> list[dict[str, Any]]:
    """
    Pick up to 3 rows: rules from onboarding Search Analytics spec (target always included).

    ``full_sorted`` is all brands sorted by appearances descending (ties: name).
    ``target_idx`` is the index of the onboarded business in that list.
    """
    n = len(full_sorted)
    if n == 0:
        return []
    if n == 1:
        return [full_sorted[0]]
    tr = target_idx + 1
    if tr == 1:
        return full_sorted[: min(3, n)]
    if tr == 2:
        if n >= 3:
            return [full_sorted[0], full_sorted[1], full_sorted[2]]
        return [full_sorted[0], full_sorted[1]]
    if n >= 3:
        return [full_sorted[0], full_sorted[1], full_sorted[target_idx]]
    return full_sorted


def _aeo_onboarding_response_extractions(
    business_profile: BusinessProfile,
) -> list[tuple[AEOResponseSnapshot, AEOExtractionSnapshot | None]]:
    """
    One row per stored LLM response (every platform). Includes responses with no extraction yet.
    """
    rsp_qs = business_profile.aeo_response_snapshots.all()
    responses = list(
        rsp_qs.prefetch_related(
            Prefetch(
                "extraction_snapshots",
                queryset=AEOExtractionSnapshot.objects.select_related("response_snapshot").order_by(
                    "-created_at",
                ),
            ),
        )
    )
    return [(resp, _pick_extraction_for_response(resp)) for resp in responses]


def _target_citation_units_for_onboarding(
    extraction: AEOExtractionSnapshot | None,
    *,
    tracked_website_url_or_domain: str,
) -> int:
    """Mention units for visibility numerator (aligned with share-of-voice when mention_count is set)."""
    if extraction is None:
        return 0
    if not brand_effectively_cited(
        bool(getattr(extraction, "brand_mentioned", False)),
        getattr(extraction, "competitors_json", None),
        tracked_website_url_or_domain=tracked_website_url_or_domain,
    ):
        return 0
    try:
        mc = max(0, int(getattr(extraction, "mention_count", 0)))
    except (TypeError, ValueError):
        mc = 0
    return mc if mc > 0 else 1


def aeo_onboarding_competitors_visibility(business_profile: BusinessProfile) -> dict[str, Any]:
    """
    Competitor + target visibility for onboarding Search Analytics (all LLM platforms).

    Visibility % = (total citation events for the brand) / (total LLM response slots) × 100.
    Denominator is every ``AEOResponseSnapshot`` row (each prompt × each platform run).

    Target citation units: sum of ``mention_count`` when the brand is cited, or 1 when cited only
    via ``brand_mentioned`` / profile URL in ``competitors_json`` (no positive mention_count).

    Competitors: one count per matching entry in ``competitors_json`` (repeats in the same answer
    count separately). Brands are keyed with ``_normalize_for_brand_match`` on extracted names.
    """
    slots = _aeo_onboarding_response_extractions(business_profile)
    total = len(slots)
    if total == 0:
        return {
            "has_data": False,
            "total_prompts": 0,
            "rows": [],
        }

    display_name = (getattr(business_profile, "business_name", None) or "").strip() or "Your business"
    target_key = _visibility_brand_key(display_name)
    site = (getattr(business_profile, "website_url", None) or "").strip()
    target_domain = root_domain_from_fragment(site) or ""

    target_appearances = sum(
        _target_citation_units_for_onboarding(ex, tracked_website_url_or_domain=site) for _, ex in slots
    )

    # Competitors: count every list row (same brand twice in one response = two cites)
    competitor_appearances: dict[str, int] = {}
    competitor_display: dict[str, str] = {}
    competitor_domain: dict[str, str] = {}
    for _resp, e in slots:
        if e is None:
            continue
        comps = getattr(e, "competitors_json", None) or []
        if not isinstance(comps, list):
            continue
        for raw in comps:
            parsed = parse_competitor_raw_item(raw)
            name = parsed["name"]
            if not name:
                continue
            key = _visibility_brand_key(name)
            if not key:
                continue
            if target_key and key == target_key:
                continue
            dom = root_domain_from_fragment(parsed["url"]) or ""
            if target_domain and dom and registered_root_domains_match(target_domain, dom):
                continue
            competitor_appearances[key] = competitor_appearances.get(key, 0) + 1
            competitor_display.setdefault(key, name.strip())
            if dom and key not in competitor_domain:
                competitor_domain[key] = dom

    def pct(appearances: int) -> float:
        return round(100.0 * float(appearances) / float(total), 1)

    rows_build: list[dict[str, Any]] = []
    for key, app in competitor_appearances.items():
        rows_build.append(
            {
                "brand_key": key,
                "brand": competitor_display.get(key, key),
                "appearances": app,
                "visibility_pct": pct(app),
                "is_target": False,
                "domain": competitor_domain.get(key, ""),
                "sentiment": None,
            }
        )

    rows_build.append(
        {
            "brand_key": target_key or "__target__",
            "brand": display_name,
            "appearances": target_appearances,
            "visibility_pct": pct(target_appearances),
            "is_target": True,
            "domain": target_domain,
            "sentiment": None,
        }
    )

    rows_build.sort(key=lambda r: (-r["appearances"], str(r["brand"]).lower()))

    for i, r in enumerate(rows_build):
        r["rank"] = i + 1

    target_idx = next((i for i, r in enumerate(rows_build) if r["is_target"]), 0)
    chosen = _pick_three_onboarding_competitor_rows(rows_build, target_idx)

    # Target sentiment: mode across extractions where the brand is cited
    sents: list[str] = []
    for _resp, e in slots:
        if e is None:
            continue
        if not brand_effectively_cited(
            bool(getattr(e, "brand_mentioned", False)),
            getattr(e, "competitors_json", None),
            tracked_website_url_or_domain=site,
        ):
            continue
        s = str(getattr(e, "sentiment", "") or "").strip().lower()
        if s in (
            AEOExtractionSnapshot.SENTIMENT_POSITIVE,
            AEOExtractionSnapshot.SENTIMENT_NEUTRAL,
            AEOExtractionSnapshot.SENTIMENT_NEGATIVE,
        ):
            sents.append(s)
    target_sentiment = None
    if sents:
        target_sentiment = Counter(sents).most_common(1)[0][0]

    for r in chosen:
        if r["is_target"]:
            r["sentiment"] = target_sentiment

    out_rows: list[dict[str, Any]] = []
    for r in sorted(chosen, key=lambda x: int(x["rank"])):
        out_rows.append(
            {
                "position": int(r["rank"]),
                "brand": r["brand"],
                "visibility_pct": float(r["visibility_pct"]),
                "sentiment": r["sentiment"],
                "domain": r.get("domain") or "",
                "is_target": bool(r["is_target"]),
            }
        )

    return {
        "has_data": True,
        "total_prompts": total,
        "rows": out_rows,
    }


def latest_extraction_per_response(
    business_profile: BusinessProfile,
    *,
    response_platform: str | None = PRIMARY_AEO_SCORING_PLATFORM,
) -> list[AEOExtractionSnapshot]:
    """
    One extraction per response snapshot: prefer latest successful parse, else latest row.

    Default ``response_platform`` is OpenAI so dual-provider runs do not double-count prompts
    in Phase 4 aggregates. Use ``None`` to consider every ``AEOResponseSnapshot`` row.
    """
    out: list[AEOExtractionSnapshot] = []
    rsp_qs = business_profile.aeo_response_snapshots.all()
    if response_platform is not None:
        rsp_qs = rsp_qs.filter(platform=response_platform)
    responses = rsp_qs.prefetch_related(
        Prefetch(
            "extraction_snapshots",
            queryset=AEOExtractionSnapshot.objects.select_related("response_snapshot").order_by(
                "-created_at",
            ),
        ),
    ).all()
    for resp in responses:
        chosen = _pick_extraction_for_response(resp)
        if chosen is not None:
            out.append(chosen)
    return out


def latest_extraction_per_response_in_window(
    business_profile: BusinessProfile,
    *,
    start: datetime,
    end: datetime,
    response_platform: str | None,
) -> list[AEOExtractionSnapshot]:
    """
    Like ``latest_extraction_per_response``, but only ``AEOResponseSnapshot`` rows with
    ``created_at`` in ``[start, end)`` (half-open). Used for monthly AEO visibility trends.
    """
    rsp_qs = business_profile.aeo_response_snapshots.filter(
        created_at__gte=start,
        created_at__lt=end,
    )
    if response_platform is not None:
        rsp_qs = rsp_qs.filter(platform=response_platform)
    responses = rsp_qs.prefetch_related(
        Prefetch(
            "extraction_snapshots",
            queryset=AEOExtractionSnapshot.objects.order_by("-created_at"),
        ),
    ).order_by("id")
    out: list[AEOExtractionSnapshot] = []
    for resp in responses:
        chosen = _pick_extraction_for_response(resp)
        if chosen is not None:
            out.append(chosen)
    return out


def _pick_extraction_for_response(resp: AEOResponseSnapshot) -> AEOExtractionSnapshot | None:
    # Prefetch uses order_by("-created_at"); model Meta ordering matches when uncached.
    rows = list(resp.extraction_snapshots.all())
    if not rows:
        return None
    for ex in rows:
        if not ex.extraction_parse_failed:
            return ex
    return rows[0]


def composite_aeo_score_from_snapshot(snap: AEOScoreSnapshot) -> int:
    """
    Single 0–100 headline score from a Phase-4 AEOScoreSnapshot.

    Equal-weight blend of visibility %, weighted average position %, and citation share %.
    """
    v = max(0.0, min(100.0, float(snap.visibility_score)))
    w = max(0.0, min(100.0, float(snap.weighted_position_score)))
    c = max(0.0, min(100.0, float(snap.citation_share)))
    return max(0, min(100, int(round((v + w + c) / 3.0))))


def save_aeo_score_snapshot(
    business_profile: BusinessProfile,
    *,
    visibility_score: float,
    weighted_position_score: float,
    citation_share: float,
    competitor_dominance: dict[str, Any],
    total_prompts: int,
    total_mentions: int,
    score_layer: str = AEOScoreSnapshot.LAYER_CONFIDENCE,
    execution_run_id: int | None = None,
) -> AEOScoreSnapshot:
    """Append-only score row for trend tracking."""
    return AEOScoreSnapshot.objects.create(
        profile=business_profile,
        score_layer=score_layer,
        execution_run_id=execution_run_id,
        visibility_score=visibility_score,
        weighted_position_score=weighted_position_score,
        citation_share=citation_share,
        competitor_dominance_json=competitor_dominance,
        total_prompts=max(0, int(total_prompts)),
        total_mentions=max(0, int(total_mentions)),
    )


def calculate_layered_scores_from_aggregates(
    business_profile: BusinessProfile,
    *,
    execution_run_id: int,
    score_layer: str,
    save: bool = True,
) -> dict[str, Any]:
    """
    Lightweight layered score from canonical prompt aggregates.
    - sample: phase-1 minimum coverage signal
    - confidence: all passes/providers collected
    """
    aggs = AEOPromptExecutionAggregate.objects.filter(
        profile=business_profile,
        execution_run_id=execution_run_id,
    )
    if score_layer == AEOScoreSnapshot.LAYER_SAMPLE:
        aggs = aggs.filter(total_pass_count__gte=2)  # one pass per provider in phase 1
    else:
        aggs = aggs.filter(openai_pass_count__gte=2, gemini_pass_count__gte=2)

    rows = list(aggs)
    total = len(rows)
    if total == 0:
        values = {
            "visibility_score": 0.0,
            "weighted_position_score": 0.0,
            "citation_share": 0.0,
            "competitor_dominance": {"ranked": [], "total_competitor_mentions": 0},
            "total_prompts": 0,
            "total_mentions": 0,
            "snapshot_id": None,
        }
        return values

    cited_prompts = sum(1 for a in rows if a.total_brand_cited_count > 0)
    visibility = round(100.0 * cited_prompts / total, 4)
    weighted = visibility
    total_mentions = sum(int(a.total_brand_cited_count or 0) for a in rows)
    total_passes = sum(int(a.total_pass_count or 0) for a in rows)
    citation_share = round(100.0 * total_mentions / max(1, total_passes), 4)
    comp_mentions = 0
    for a in rows:
        comp_mentions += len(a.last_openai_competitors_json or [])
        comp_mentions += len(a.last_gemini_competitors_json or [])

    snap_id = None
    if save:
        snap = save_aeo_score_snapshot(
            business_profile,
            visibility_score=visibility,
            weighted_position_score=weighted,
            citation_share=citation_share,
            competitor_dominance={"ranked": [], "total_competitor_mentions": int(comp_mentions)},
            total_prompts=total,
            total_mentions=total_mentions,
            score_layer=score_layer,
            execution_run_id=execution_run_id,
        )
        snap_id = snap.id
    return {
        "snapshot_id": snap_id,
        "visibility_score": visibility,
        "weighted_position_score": weighted,
        "citation_share": citation_share,
        "competitor_dominance": {"ranked": [], "total_competitor_mentions": int(comp_mentions)},
        "total_prompts": total,
        "total_mentions": total_mentions,
    }


def calculate_aeo_scores_for_business(
    business_profile: BusinessProfile,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """
    Load latest extractions for the profile, compute metrics, optionally persist AEOScoreSnapshot.

    Returns computed values plus snapshot_id when saved.
    """
    site = (getattr(business_profile, "website_url", None) or "").strip()
    extractions = latest_extraction_per_response(business_profile)
    visibility = calculate_visibility_score(extractions, tracked_website_url=site)
    weighted_pos = calculate_weighted_position_score(extractions, tracked_website_url=site)
    cite_share = calculate_citation_share(extractions)
    comp_dom = calculate_competitor_dominance(extractions)

    target_units = 0
    for e in extractions:
        try:
            target_units += max(0, int(e.mention_count))
        except (TypeError, ValueError):
            pass
    competitor_units = comp_dom.get("total_competitor_mentions", 0)
    total_mentions = int(target_units + competitor_units)

    snapshot_id: int | None = None
    if save:
        snap = save_aeo_score_snapshot(
            business_profile,
            visibility_score=visibility,
            weighted_position_score=weighted_pos,
            citation_share=cite_share,
            competitor_dominance=comp_dom,
            total_prompts=len(extractions),
            total_mentions=total_mentions,
        )
        snapshot_id = snap.id

    return {
        "snapshot_id": snapshot_id,
        "visibility_score": visibility,
        "weighted_position_score": weighted_pos,
        "citation_share": cite_share,
        "competitor_dominance": comp_dom,
        "total_prompts": len(extractions),
        "total_mentions": total_mentions,
    }
