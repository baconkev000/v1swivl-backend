"""
Phase 4: deterministic AEO metrics from extraction snapshots.

Formulas are explicit (no hidden multipliers). Does not import extraction or execution modules.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol, Sequence

from django.db.models import Prefetch

from ..models import AEOExtractionSnapshot, AEOScoreSnapshot, AEOResponseSnapshot, BusinessProfile


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


def calculate_visibility_score(extractions: Sequence[_ExtractionLike]) -> float:
    """
    Percentage of prompts where the target brand was mentioned.

    Formula: (count where brand_mentioned) / (total prompts) × 100
    """
    n = len(extractions)
    if n == 0:
        return 0.0
    mentioned = sum(1 for e in extractions if e.brand_mentioned)
    return round(100.0 * mentioned / n, 4)


def calculate_weighted_position_score(extractions: Sequence[_ExtractionLike]) -> float:
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
        if not e.brand_mentioned:
            continue
        pos = (e.mention_position or AEOExtractionSnapshot.MENTION_NONE).lower()
        total_w += _POSITION_WEIGHTS.get(pos, 0.0)
    return round(100.0 * total_w / n, 4)


def calculate_citation_share(extractions: Sequence[_ExtractionLike]) -> float:
    """
    Target share of modeled \"named business\" mention units across extractions.

    Numerator: sum of mention_count (target brand mentions from extraction).
    Denominator: that sum plus one unit per competitor string listed per extraction.

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
            name = str(raw).strip()
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


def latest_extraction_per_response(
    business_profile: BusinessProfile,
) -> list[AEOExtractionSnapshot]:
    """
    One extraction per response snapshot: prefer latest successful parse, else latest row.
    """
    out: list[AEOExtractionSnapshot] = []
    responses = business_profile.aeo_response_snapshots.prefetch_related(
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


def _pick_extraction_for_response(resp: AEOResponseSnapshot) -> AEOExtractionSnapshot | None:
    # Prefetch uses order_by("-created_at"); model Meta ordering matches when uncached.
    rows = list(resp.extraction_snapshots.all())
    if not rows:
        return None
    for ex in rows:
        if not ex.extraction_parse_failed:
            return ex
    return rows[0]


def save_aeo_score_snapshot(
    business_profile: BusinessProfile,
    *,
    visibility_score: float,
    weighted_position_score: float,
    citation_share: float,
    competitor_dominance: dict[str, Any],
    total_prompts: int,
    total_mentions: int,
) -> AEOScoreSnapshot:
    """Append-only score row for trend tracking."""
    return AEOScoreSnapshot.objects.create(
        profile=business_profile,
        visibility_score=visibility_score,
        weighted_position_score=weighted_position_score,
        citation_share=citation_share,
        competitor_dominance_json=competitor_dominance,
        total_prompts=max(0, int(total_prompts)),
        total_mentions=max(0, int(total_mentions)),
    )


def calculate_aeo_scores_for_business(
    business_profile: BusinessProfile,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """
    Load latest extractions for the profile, compute metrics, optionally persist AEOScoreSnapshot.

    Returns computed values plus snapshot_id when saved.
    """
    extractions = latest_extraction_per_response(business_profile)
    visibility = calculate_visibility_score(extractions)
    weighted_pos = calculate_weighted_position_score(extractions)
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
