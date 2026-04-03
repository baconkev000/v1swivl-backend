"""
Heuristic scoring + OpenAI pass for onboarding DataForSEO ranked keywords (AEO prompt use).
"""
from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Tuple

from django.conf import settings

from .onboarding_topic_clusters import tokenize

logger = logging.getLogger(__name__)

AEO_ONBOARDING_KEYWORD_MIN_COUNT = 10

# OpenAI user prompt ({{keywords}} substituted at runtime).
AEO_KEYWORD_FILTER_PROMPT_TEMPLATE = """You are filtering SEO ranked keywords so they become AEO-ready topic phrases for local business prompt generation.

Your job is not only to remove weak keywords, but also to improve remaining keywords so they reflect how a real consumer would naturally think about the topic.

Keep keywords only if they:

* reflect services the business likely offers
* include local intent, trust intent, transactional intent, cost intent, or comparison intent
* can realistically lead to a consumer question

Remove keywords that:

* are too generic
* are broad informational blog traffic
* are weakly connected to the business
* are mostly branded lookup fragments without consumer decision value

Critical mutation rule:
Do not simply return raw SEO phrases when they sound mechanical.
Convert kept keywords into cleaner AEO topic phrases that preserve intent but sound more natural.

Examples:
"professional teeth whitening near me" → "professional teeth whitening salt lake city"
"dental implant cost utah" → "dental implant pricing utah"
"salt lake dental reviews" → "dentist reviews salt lake city"

Intent expansion rule:
Prefer phrases that sound like the beginning of a real consumer decision, trust, comparison, or cost topic.

Deduplicate aggressively:
Treat singular/plural, reordered words, city/state variants, and minor wording differences as one topic.

Return at least {{min_count}} keyword entries.

For each returned item provide:

* keyword
* category
* reason

Input keywords:
{{keywords}}

Return only JSON array.
"""

# OpenAI user prompt when we already have filtered topics but need more to reach minimum count.
# Placeholders: {{business_name}}, {{location}}, {{current}}, {{need}}
AEO_KEYWORD_ENRICH_PROMPT_TEMPLATE = """You are proposing additional AEO-ready topic phrases for a local business.

Business: {{business_name}}
Location: {{location}}

We already have these topics (do not repeat or trivially rephrase them; each new phrase must be meaningfully different):

{{current}}

Propose exactly {{need}} new topic phrases that:

* match services or problems a real consumer would ask about for this business in this area
* favor local intent, trust, comparison, cost, or transactional angles where natural
* sound like natural search or voice-query phrasing, not SEO keyword stuffing

For each item provide:

* keyword — the phrase (string)
* category — one of: service, trust, transactional, comparison, informational-support
* reason — one short sentence why it fits this business

Return only a JSON array of objects with keys keyword, category, reason. No more than {{need}} items.
"""

_TRUST_PATTERNS = re.compile(
    r"\b(best|reviews?|recommended|recommend|cost|price|pricing|vs\.?|versus|compare|comparison|"
    r"top[-\s]?rated|how\s+much|worth\s+it|affordable|cheap|cheapest|near\s+me)\b",
    re.I,
)

# Broad informational / weak-intent tokens (not exhaustive; OpenAI refines further).
_GENERIC_TOKENS = frozenset({
    "toothpaste", "floss", "mouthwash", "swelling", "headache", "fever", "symptoms",
    "causes", "definition", "meaning", "history", "facts", "pictures", "images", "memes",
    "wiki", "wikipedia", "reddit", "jobs", "salary", "career", "degree", "university",
})

_GENERIC_PREFIX = re.compile(
    r"^\s*(what\s+is|what\s+are|who\s+is|define|meaning\s+of)\b",
    re.I,
)

_VALID_CATEGORIES = frozenset({
    "service",
    "trust",
    "transactional",
    "comparison",
    "informational-support",
})


def _union_seed_tokens(seeds: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for s in seeds:
        for t in s.get("tokens") or []:
            if isinstance(t, str) and len(t) >= 3:
                out.add(t.lower())
    return out


def _location_tokens(location: str) -> Set[str]:
    raw = (location or "").strip().lower()
    if not raw:
        return set()
    parts = re.split(r"[,/]|(?:\band\b)", raw)
    toks: Set[str] = set()
    for p in parts:
        for w in re.findall(r"[a-z][a-z0-9-]*", p):
            if len(w) >= 3 and w not in {"the", "and", "united", "states", "county"}:
                toks.add(w)
    return toks


def _business_name_tokens(business_name: str) -> Set[str]:
    name = (business_name or "").strip().lower()
    if not name:
        return set()
    return {w for w in re.findall(r"[a-z0-9][a-z0-9-]*", name) if len(w) >= 2}


def _has_service_intent(keyword: str, seed_tokens: Set[str]) -> bool:
    kt = tokenize(keyword)
    if not kt:
        return False
    if kt & seed_tokens:
        return True
    # Common local-service modifiers without seed overlap
    if re.search(
        r"\b(dentist|dental|doctor|clinic|lawyer|attorney|plumber|hvac|roofing|salon|spa|"
        r"repair|contractor|insurance|real\s+estate|restaurant|gym|fitness)\b",
        keyword,
        re.I,
    ):
        return True
    return False


def _has_location_intent(keyword: str, loc_tokens: Set[str]) -> bool:
    kl = keyword.lower()
    if "near me" in kl or re.search(r"\b(in|at)\s+[a-z][a-z\s,-]{2,40}\s+(city|county)\b", kl):
        return True
    kt = tokenize(keyword)
    if loc_tokens and kt & loc_tokens:
        return True
    return False


def _has_trust_or_comparison_intent(keyword: str) -> bool:
    return bool(_TRUST_PATTERNS.search(keyword))


def _too_generic_penalty(keyword: str) -> bool:
    kl = keyword.lower()
    if _GENERIC_PREFIX.search(kl):
        return True
    kt = tokenize(keyword)
    if kt & _GENERIC_TOKENS:
        return True
    return False


def _branded_only_weak(keyword: str, brand_tokens: Set[str], seed_tokens: Set[str]) -> bool:
    if not brand_tokens:
        return False
    kt = tokenize(keyword)
    if not kt:
        return False
    if kt & seed_tokens or _has_trust_or_comparison_intent(keyword):
        return False
    if len(kt & brand_tokens) >= max(1, len(kt)-1):
        return True
    # Keyword is almost only brand (e.g. "acme dental" when brand is "acme dental")
    if len(kt - brand_tokens) <= 0:
        return True
    return False


def score_keyword_for_aeo(
    keyword: str,
    *,
    seed_tokens: Set[str],
    location_tokens: Set[str],
    brand_tokens: Set[str],
) -> int:
    """
    Score 1–5 (clamped). Keep keywords with score >= 3 in the heuristic layer.
    """
    score = 1
    if _has_service_intent(keyword, seed_tokens):
        score += 2
    if _has_location_intent(keyword, location_tokens):
        score += 2
    if _has_trust_or_comparison_intent(keyword):
        score += 1
    if _too_generic_penalty(keyword):
        score -= 2
    if _branded_only_weak(keyword, brand_tokens, seed_tokens):
        score -= 2
    return max(1, min(5, score))


_PHRASE_NORMALIZE_SUBS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwisdom\s+teeth\b", re.I), "wisdom tooth"),
    (re.compile(r"\bwisdom\s+tooth\b", re.I), "wisdom tooth"),
    (re.compile(r"\bteeth\b", re.I), "tooth"),
    (re.compile(r"\bimplants\b", re.I), "implant"),
    (re.compile(r"\bveneers\b", re.I), "veneer"),
    (re.compile(r"\bbraces\b", re.I), "brace"),
)


def _normalize_phrase_for_match(keyword: str) -> str:
    t = keyword.lower().strip()
    for rx, rep in _PHRASE_NORMALIZE_SUBS:
        t = rx.sub(rep, t)
    t = re.sub(r"[\s,]+", " ", t).strip()
    t = re.sub(
        r"\b(usa|us|united states|county)\b$",
        "",
        t,
        flags=re.I,
    ).strip()
    return t


def _token_jaccard_strings(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb) or 1
    return inter / union


def _token_jaccard(a: str, b: str) -> float:
    na, nb = _normalize_phrase_for_match(a), _normalize_phrase_for_match(b)
    return _token_jaccard_strings(na, nb)


def _near_duplicate(a: str, b: str) -> bool:
    if a.lower().strip() == b.lower().strip():
        return True
    na, nb = _normalize_phrase_for_match(a), _normalize_phrase_for_match(b)
    if na == nb:
        return True
    if SequenceMatcher(None, na, nb).ratio() >= 0.86:
        return True
    tja = _token_jaccard_strings(na, nb)
    if tja >= 0.78:
        return True
    ta, tb = tokenize(na), tokenize(nb)
    if ta and tb:
        smaller, larger = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if smaller <= larger and len(smaller) >= 4 and len(smaller) / max(len(larger), 1) >= 0.72:
            return True
    if SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.9:
        return True
    return False


def dedupe_near_duplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop near-duplicates, keeping higher search_volume then better rank."""
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            -int(r.get("search_volume") or 0),
            0 if r.get("rank") is not None else 1,
            int(r.get("rank") or 9999),
        ),
    )
    kept: List[Dict[str, Any]] = []
    for row in sorted_rows:
        kw = str(row.get("keyword") or "").strip()
        if not kw:
            continue
        if any(_near_duplicate(kw, str(k.get("keyword") or "")) for k in kept):
            continue
        kept.append(row)
    return kept


def heuristic_filter_ranked_rows(
    rows: List[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    seeds: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Dedupe, score, keep score >= 3; attach ``aeo_score`` on each row.
    """
    seed_tokens = _union_seed_tokens(seeds)
    loc_tokens = _location_tokens(str(context.get("location") or ""))
    brand_tokens = _business_name_tokens(str(context.get("business_name") or ""))

    deduped = dedupe_near_duplicate_rows(rows)
    out: List[Dict[str, Any]] = []
    for row in deduped:
        kw = str(row.get("keyword") or "").strip()
        if not kw:
            continue
        sc = score_keyword_for_aeo(
            kw,
            seed_tokens=seed_tokens,
            location_tokens=loc_tokens,
            brand_tokens=brand_tokens,
        )
        if sc < 3:
            continue
        new_row = dict(row)
        new_row["aeo_score"] = sc
        out.append(new_row)
    return out


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_ai_keyword_items(raw: str) -> Tuple[List[Dict[str, Any]], str | None]:
    if not raw:
        return [], "empty OpenAI response"
    try:
        data = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError as exc:
        logger.warning("[onboarding AEO keyword filter] JSON parse failed: %s raw=%s", exc, raw[:500])
        return [], f"invalid JSON: {exc}"
    if isinstance(data, dict) and "keywords" in data:
        data = data["keywords"]
    if not isinstance(data, list):
        return [], "OpenAI JSON was not a list"
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kw = str(item.get("keyword") or "").strip()
        if not kw:
            continue
        cat = str(item.get("category") or "service").strip().lower().replace(" ", "-")
        if cat not in _VALID_CATEGORIES:
            if cat == "informational_support":
                cat = "informational-support"
            elif "informational" in cat and "support" in cat:
                cat = "informational-support"
            elif "transaction" in cat:
                cat = "transactional"
            elif "compar" in cat:
                cat = "comparison"
            else:
                cat = "service"
        reason = str(item.get("reason") or "").strip()[:500]
        out.append({"keyword": kw, "category": cat, "reason": reason})
    return out, None


def _openai_chat_user_only(user_content: str, *, temperature: float, timeout: float) -> Tuple[str | None, str | None]:
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except ImportError:
        return None, "openai package not available"
    try:
        from .openai_utils import _get_client, _get_model, chat_completion_create_logged
    except Exception as exc:
        return None, str(exc)
    client = _get_client()
    model = _get_model()
    try:
        completion = chat_completion_create_logged(
            client,
            operation="openai.chat.onboarding_aeo_keyword_filter",
            business_profile=None,
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=temperature,
            timeout=timeout,
        )
    except (APIConnectionError, APITimeoutError, RateLimitError, Exception) as exc:
        logger.warning("[onboarding AEO keyword filter] OpenAI call failed: %s", exc)
        return None, str(exc)
    raw = (completion.choices[0].message.content or "").strip()
    return raw, None


def openai_filter_keywords_for_aeo(
    keywords: List[str],
    *,
    timeout: float | None = None,
    min_count: int = AEO_ONBOARDING_KEYWORD_MIN_COUNT,
) -> Tuple[List[Dict[str, Any]], str | None]:
    """
    Call OpenAI with the AEO filter prompt. Returns (rows, error_message).
    """
    if not keywords:
        return [], None

    payload = json.dumps(keywords, ensure_ascii=False)
    user_content = (
        AEO_KEYWORD_FILTER_PROMPT_TEMPLATE.replace("{{keywords}}", payload).replace(
            "{{min_count}}",
            str(min_count),
        )
    )
    to = timeout if timeout is not None else float(getattr(settings, "AEO_OPENAI_TIMEOUT", 60.0))
    raw, err = _openai_chat_user_only(user_content, temperature=0.2, timeout=to)
    if err or raw is None:
        return [], err or "empty response"
    return _parse_ai_keyword_items(raw)


def openai_enrich_keywords_for_minimum(
    current_keywords: List[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    need: int,
    timeout: float | None = None,
) -> Tuple[List[Dict[str, Any]], str | None]:
    """Ask OpenAI for ``need`` additional non-duplicate phrases."""
    if need <= 0:
        return [], None
    biz = str(context.get("business_name") or "").strip() or "Unknown business"
    loc = str(context.get("location") or "").strip() or "Unknown location"
    payload = json.dumps(
        [{"keyword": r.get("keyword"), "category": r.get("aeo_category"), "reason": r.get("aeo_reason")} for r in current_keywords],
        ensure_ascii=False,
    )
    user_content = (
        AEO_KEYWORD_ENRICH_PROMPT_TEMPLATE.replace("{{business_name}}", biz)
        .replace("{{location}}", loc)
        .replace("{{current}}", payload)
        .replace("{{need}}", str(need))
    )
    to = timeout if timeout is not None else float(getattr(settings, "AEO_OPENAI_TIMEOUT", 60.0))
    raw, err = _openai_chat_user_only(user_content, temperature=0.35, timeout=to)
    if err or raw is None:
        return [], err or "empty response"
    return _parse_ai_keyword_items(raw)


def _merge_ai_rows_into_heuristic(
    ai_rows: List[Dict[str, Any]],
    by_kw_lower: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for ai in ai_rows:
        ak = str(ai.get("keyword") or "").strip()
        if not ak:
            continue
        base = by_kw_lower.get(ak.lower())
        if base is None:
            for lk, row in by_kw_lower.items():
                if _near_duplicate(ak, lk) or ak.lower() == lk:
                    base = row
                    break
        if base is None:
            m = {
                "keyword": ak,
                "search_volume": 0,
                "rank": None,
                "rank_group": None,
                "aeo_score": 3,
            }
        else:
            m = dict(base)
            m["keyword"] = ak
        kl = str(m["keyword"]).lower().strip()
        if any(_near_duplicate(m["keyword"], str(o.get("keyword") or "")) for o in merged):
            continue
        if kl in seen:
            continue
        seen.add(kl)
        m["aeo_category"] = ai.get("category") or "service"
        m["aeo_reason"] = str(ai.get("reason") or "").strip()[:500]
        merged.append(m)
    return merged


def _backfill_from_ranked_norm(
    merged: List[Dict[str, Any]],
    ranked_norm: List[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    seeds: List[Dict[str, Any]],
    min_count: int,
) -> List[Dict[str, Any]]:
    out = dedupe_near_duplicate_rows(merged)
    existing_kws = [str(r.get("keyword") or "") for r in out]
    seed_tokens = _union_seed_tokens(seeds)
    loc_tokens = _location_tokens(str(context.get("location") or ""))
    brand_tokens = _business_name_tokens(str(context.get("business_name") or ""))
    sorted_pool = sorted(
        ranked_norm,
        key=lambda r: (
            -int(r.get("search_volume") or 0),
            r.get("rank") is not None,
            int(r.get("rank") or 9999),
        ),
    )
    for row in sorted_pool:
        if len(out) >= min_count:
            break
        kw = str(row.get("keyword") or "").strip()
        if not kw:
            continue
        if any(_near_duplicate(kw, ex) for ex in existing_kws):
            continue
        if score_keyword_for_aeo(kw, seed_tokens=seed_tokens, location_tokens=loc_tokens, brand_tokens=brand_tokens) < 2:
            continue
        new_row = dict(row)
        new_row["aeo_score"] = max(int(new_row.get("aeo_score") or 0), 2)
        new_row.setdefault("aeo_category", "service")
        new_row.setdefault("aeo_reason", "Included to reach minimum topic count.")
        out.append(new_row)
        existing_kws.append(kw)
    return out


def apply_aeo_keyword_pipeline(
    ranked_norm: List[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    seeds: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Heuristic layer (dedupe, score >= 3), then OpenAI filter + dedupe; ensure minimum count.
    """
    min_n = AEO_ONBOARDING_KEYWORD_MIN_COUNT
    heur = heuristic_filter_ranked_rows(ranked_norm, context=context, seeds=seeds)
    if not heur:
        return []

    by_kw_lower = {str(r["keyword"]).lower().strip(): r for r in heur}
    kws = [str(r["keyword"]) for r in heur]

    ai_rows, err = openai_filter_keywords_for_aeo(kws, min_count=min_n)
    if err:
        logger.info("[onboarding AEO keyword filter] using heuristic-only (%s)", err[:200])
        merged = dedupe_near_duplicate_rows(heur)
    else:
        merged = _merge_ai_rows_into_heuristic(ai_rows, by_kw_lower)
        merged = dedupe_near_duplicate_rows(merged)
        if not merged:
            merged = dedupe_near_duplicate_rows(heur)

    if len(merged) < min_n:
        need = min_n - len(merged)
        enrich, err2 = openai_enrich_keywords_for_minimum(merged, context=context, need=need)
        if not err2 and enrich:
            for item in enrich:
                ak = str(item.get("keyword") or "").strip()
                if not ak:
                    continue
                if any(_near_duplicate(ak, str(x.get("keyword") or "")) for x in merged):
                    continue
                row = {
                    "keyword": ak,
                    "search_volume": 0,
                    "rank": None,
                    "rank_group": None,
                    "aeo_score": 3,
                    "aeo_category": item.get("category") or "service",
                    "aeo_reason": str(item.get("reason") or "")[:500],
                    "aeo_suggested": True,
                }
                merged.append(row)
                if len(merged) >= min_n:
                    break
        merged = dedupe_near_duplicate_rows(merged)

    if len(merged) < min_n:
        merged = _backfill_from_ranked_norm(merged, ranked_norm, context=context, seeds=seeds, min_count=min_n)

    return merged


def ranked_rows_as_labs_api_shape(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape filtered rows like DataForSEO Labs items for ``build_topic_clusters``."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        kw = str(r.get("keyword") or "").strip()
        if not kw:
            continue
        out.append(
            {
                "keyword_data": {
                    "keyword": kw,
                    "keyword_info": {"search_volume": int(r.get("search_volume") or 0)},
                },
                "rank_absolute": r.get("rank"),
                "rank_group": r.get("rank_group"),
            },
        )
    return out
