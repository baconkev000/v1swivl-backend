from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

from django.db import transaction

from ..models import AEOPromptExecutionAggregate, AEOExtractionSnapshot, AEOResponseSnapshot, BusinessProfile
from .aeo_execution_utils import PLATFORM_GEMINI, PLATFORM_OPENAI, hash_prompt, normalize_aeo_prompt_dict

PHASE1_TOTAL_CALLS = 10
PHASE1_CATEGORIES = ("authority", "comparison", "pricing", "trust", "service")
PHASE1_CALLS_PER_CATEGORY = 2
PASSES_PER_PROVIDER_TARGET = 2

_WRONG_URL_STATUSES = {
    AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE,
    AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN,
}


def classify_prompt_category(prompt_obj: Mapping[str, Any]) -> str:
    spec = normalize_aeo_prompt_dict(prompt_obj)
    text = str(spec.get("prompt") or "").strip().lower()
    ptype = str(spec.get("type") or "").strip().lower()
    if ptype in {"authority", "comparison", "trust"}:
        return ptype
    if any(tok in text for tok in ("price", "pricing", "cost", "quote", "afford")):
        return "pricing"
    if any(tok in text for tok in ("service", "near me", "provider", "company", "business")):
        return "service"
    # deterministic fallback pool balance
    stable = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) if text else 0
    return ("authority", "comparison", "trust", "service", "pricing")[stable % 5]


def _stable_sort_key(prompt_obj: Mapping[str, Any]) -> tuple[str, str]:
    spec = normalize_aeo_prompt_dict(prompt_obj)
    text = str(spec.get("prompt") or "").strip()
    return (hash_prompt(text), text.lower())


def build_phase1_provider_batches(prompt_set: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Deterministic phase-1 selection:
    - 2 prompts/category for 5 categories (10 total)
    - provider-balanced: openai gets 1/category, gemini gets 1/category
    """
    categorized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in prompt_set:
        spec = normalize_aeo_prompt_dict(raw)
        if not spec.get("prompt"):
            continue
        cat = classify_prompt_category(spec)
        spec2 = dict(spec)
        spec2["_aeo_category"] = cat
        categorized[cat].append(spec2)

    # deterministic ordering in every bucket
    for cat in categorized:
        categorized[cat].sort(key=_stable_sort_key)

    openai_batch: list[dict[str, Any]] = []
    gemini_batch: list[dict[str, Any]] = []
    used_hashes: set[str] = set()

    for cat in PHASE1_CATEGORIES:
        bucket = categorized.get(cat, [])
        picked: list[dict[str, Any]] = []
        for item in bucket:
            h = hash_prompt(str(item.get("prompt") or ""))
            if h in used_hashes:
                continue
            picked.append(item)
            used_hashes.add(h)
            if len(picked) >= PHASE1_CALLS_PER_CATEGORY:
                break
        # backfill from any other prompts if category sparse
        if len(picked) < PHASE1_CALLS_PER_CATEGORY:
            remainder = sorted(
                (
                    dict(normalize_aeo_prompt_dict(p), _aeo_category=cat)
                    for p in prompt_set
                    if str(normalize_aeo_prompt_dict(p).get("prompt") or "").strip()
                ),
                key=_stable_sort_key,
            )
            for item in remainder:
                h = hash_prompt(str(item.get("prompt") or ""))
                if h in used_hashes:
                    continue
                picked.append(item)
                used_hashes.add(h)
                if len(picked) >= PHASE1_CALLS_PER_CATEGORY:
                    break
        if picked:
            openai_batch.append(picked[0])
        if len(picked) > 1:
            gemini_batch.append(picked[1])

    return {
        PLATFORM_OPENAI: openai_batch[: len(PHASE1_CATEGORIES)],
        PLATFORM_GEMINI: gemini_batch[: len(PHASE1_CATEGORIES)],
    }


def _material_set(v: Any) -> set[str]:
    if not isinstance(v, list):
        return set()
    out: set[str] = set()
    for item in v:
        if isinstance(item, dict):
            out.add(str(item.get("name") or item.get("url") or "").strip().casefold())
        else:
            out.add(str(item or "").strip().casefold())
    return {x for x in out if x}


def _provider_stability_from_history(history: list[bool]) -> tuple[str, bool]:
    """
    Provider-local stability rule:
    - <2 passes => pending
    - 2 passes equal => stable
    - 2 passes differ => unstable + third-pass required
    - >=3 passes => if first two differed and third creates 2-of-3 majority => stabilized_after_third
      else unstable (fallback deterministic rule).
    """
    h = [bool(x) for x in (history or [])]
    if len(h) < 2:
        return AEOPromptExecutionAggregate.STABILITY_PENDING, False
    if h[0] == h[1]:
        return AEOPromptExecutionAggregate.STABILITY_STABLE, False
    if len(h) == 2:
        return AEOPromptExecutionAggregate.STABILITY_UNSTABLE, True
    window = h[:3]
    true_n = sum(1 for x in window if x)
    false_n = len(window) - true_n
    if true_n >= 2 or false_n >= 2:
        return AEOPromptExecutionAggregate.STABILITY_STABILIZED_AFTER_THIRD, False
    return AEOPromptExecutionAggregate.STABILITY_UNSTABLE, False


def recompute_stability(agg: AEOPromptExecutionAggregate) -> tuple[str, list[str]]:
    openai_status, _ = _provider_stability_from_history(list(agg.openai_brand_mention_history or []))
    gemini_status, _ = _provider_stability_from_history(list(agg.gemini_brand_mention_history or []))
    reasons: list[str] = []
    if openai_status == AEOPromptExecutionAggregate.STABILITY_PENDING or gemini_status == AEOPromptExecutionAggregate.STABILITY_PENDING:
        reasons.append("missing_second_provider_or_pass")
    if agg.openai_last_wrong_url_status in _WRONG_URL_STATUSES or agg.gemini_last_wrong_url_status in _WRONG_URL_STATUSES:
        reasons.append("wrong_url_attribution_present")
    if openai_status == AEOPromptExecutionAggregate.STABILITY_UNSTABLE or gemini_status == AEOPromptExecutionAggregate.STABILITY_UNSTABLE:
        reasons.append("provider_unstable")
    if bool(agg.last_openai_brand_mentioned) != bool(agg.last_gemini_brand_mentioned):
        reasons.append("brand_mention_changed_across_provider")
    if _material_set(agg.last_openai_citations_json) != _material_set(agg.last_gemini_citations_json):
        reasons.append("citation_set_changed")
    if _material_set(agg.last_openai_competitors_json) != _material_set(agg.last_gemini_competitors_json):
        reasons.append("competitor_set_changed")
    if not reasons:
        return AEOPromptExecutionAggregate.STABILITY_STABLE, []
    if "missing_second_provider_or_pass" in reasons and len(reasons) == 1:
        return AEOPromptExecutionAggregate.STABILITY_PENDING, reasons
    if (
        openai_status in {AEOPromptExecutionAggregate.STABILITY_STABLE, AEOPromptExecutionAggregate.STABILITY_STABILIZED_AFTER_THIRD}
        and gemini_status in {AEOPromptExecutionAggregate.STABILITY_STABLE, AEOPromptExecutionAggregate.STABILITY_STABILIZED_AFTER_THIRD}
        and "wrong_url_attribution_present" not in reasons
    ):
        return AEOPromptExecutionAggregate.STABILITY_STABLE, reasons
    return AEOPromptExecutionAggregate.STABILITY_UNSTABLE, reasons


@transaction.atomic
def update_prompt_aggregate_from_extraction(
    *,
    profile: BusinessProfile,
    execution_run_id: int | None,
    response_snapshot: AEOResponseSnapshot,
    extraction_snapshot: AEOExtractionSnapshot,
    prompt_category: str = "",
) -> AEOPromptExecutionAggregate:
    agg, _ = AEOPromptExecutionAggregate.objects.select_for_update().get_or_create(
        profile=profile,
        execution_run_id=execution_run_id,
        prompt_hash=response_snapshot.prompt_hash,
        defaults={
            "prompt_text": response_snapshot.prompt_text,
            "prompt_type": response_snapshot.prompt_type,
            "prompt_category": prompt_category or "",
        },
    )
    if prompt_category and not agg.prompt_category:
        agg.prompt_category = prompt_category
    if response_snapshot.platform == PLATFORM_OPENAI:
        agg.openai_pass_count += 1
        openai_pass_idx = int(agg.openai_pass_count or 0)
        agg.openai_brand_mention_history = list(agg.openai_brand_mention_history or []) + [
            bool(extraction_snapshot.brand_mentioned)
        ]
        agg.openai_pass_history_json = list(agg.openai_pass_history_json or []) + [
            {
                "pass_index": openai_pass_idx,
                "provider": PLATFORM_OPENAI,
                "brand_mentioned": bool(extraction_snapshot.brand_mentioned),
                "competitors": list(extraction_snapshot.competitors_json or []),
                "citations": list(extraction_snapshot.citations_json or []),
                "wrong_url_status": str(extraction_snapshot.brand_mentioned_url_status or ""),
            }
        ]
        if extraction_snapshot.brand_mentioned:
            agg.openai_brand_cited_count += 1
        if extraction_snapshot.brand_mentioned_url_status in _WRONG_URL_STATUSES:
            agg.openai_wrong_url_count += 1
        agg.last_openai_response_snapshot = response_snapshot
        agg.last_openai_competitors_json = extraction_snapshot.competitors_json or []
        agg.last_openai_citations_json = extraction_snapshot.citations_json or []
        agg.last_openai_brand_mentioned = bool(extraction_snapshot.brand_mentioned)
        agg.openai_last_wrong_url_status = str(extraction_snapshot.brand_mentioned_url_status or "")
        openai_status, openai_need_third = _provider_stability_from_history(list(agg.openai_brand_mention_history or []))
        agg.openai_stability_status = openai_status
        agg.openai_third_pass_required = bool(openai_need_third and agg.openai_pass_count < 3)
        agg.openai_third_pass_ran = agg.openai_pass_count >= 3
    elif response_snapshot.platform == PLATFORM_GEMINI:
        agg.gemini_pass_count += 1
        gemini_pass_idx = int(agg.gemini_pass_count or 0)
        agg.gemini_brand_mention_history = list(agg.gemini_brand_mention_history or []) + [
            bool(extraction_snapshot.brand_mentioned)
        ]
        agg.gemini_pass_history_json = list(agg.gemini_pass_history_json or []) + [
            {
                "pass_index": gemini_pass_idx,
                "provider": PLATFORM_GEMINI,
                "brand_mentioned": bool(extraction_snapshot.brand_mentioned),
                "competitors": list(extraction_snapshot.competitors_json or []),
                "citations": list(extraction_snapshot.citations_json or []),
                "wrong_url_status": str(extraction_snapshot.brand_mentioned_url_status or ""),
            }
        ]
        if extraction_snapshot.brand_mentioned:
            agg.gemini_brand_cited_count += 1
        if extraction_snapshot.brand_mentioned_url_status in _WRONG_URL_STATUSES:
            agg.gemini_wrong_url_count += 1
        agg.last_gemini_response_snapshot = response_snapshot
        agg.last_gemini_competitors_json = extraction_snapshot.competitors_json or []
        agg.last_gemini_citations_json = extraction_snapshot.citations_json or []
        agg.last_gemini_brand_mentioned = bool(extraction_snapshot.brand_mentioned)
        agg.gemini_last_wrong_url_status = str(extraction_snapshot.brand_mentioned_url_status or "")
        gemini_status, gemini_need_third = _provider_stability_from_history(list(agg.gemini_brand_mention_history or []))
        agg.gemini_stability_status = gemini_status
        agg.gemini_third_pass_required = bool(gemini_need_third and agg.gemini_pass_count < 3)
        agg.gemini_third_pass_ran = agg.gemini_pass_count >= 3

    agg.total_pass_count = agg.openai_pass_count + agg.gemini_pass_count
    agg.total_brand_cited_count = agg.openai_brand_cited_count + agg.gemini_brand_cited_count
    status, reasons = recompute_stability(agg)
    agg.stability_status = status
    agg.stability_reasons = reasons
    agg.save()
    return agg

