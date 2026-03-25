"""
Phase 3: structured extraction from raw AEO answers (second-pass OpenAI JSON).

Keeps execution (Phase 2) separate; prompt strings for extraction live in aeo_prompts.py.

Optional settings:
    AEO_EXTRACTION_PARSER_MODEL — default: same as OPENAI_MODEL / _get_model()
    AEO_EXTRACTION_TEMPERATURE — default 0.1
    AEO_EXTRACTION_MAX_TOKENS — default 800
    AEO_EXTRACTION_TIMEOUT — HTTP timeout seconds, default 60
    AEO_EXTRACTION_API_KEY_ENV — default OPEN_AI_SEO_API_KEY
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from django.conf import settings

from ..models import AEOExtractionSnapshot, AEOResponseSnapshot, BusinessProfile
from ..openai_utils import _get_client, _get_model
from .aeo_prompts import (
    AEO_STRUCTURED_EXTRACTION_RETRY_SUFFIX,
    AEO_STRUCTURED_EXTRACTION_SYSTEM_PROMPT,
    AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE,
    GENERIC_COMPETITOR_TOKENS,
)

logger = logging.getLogger(__name__)

DEFAULT_API_KEY_ENV = "OPEN_AI_SEO_API_KEY"
_VALID_POSITIONS = frozenset({"top", "middle", "bottom", "none"})
_VALID_SENTIMENT = frozenset({"positive", "neutral", "negative"})


def _extraction_model() -> str:
    return getattr(settings, "AEO_EXTRACTION_PARSER_MODEL", None) or _get_model()


def _extraction_temperature() -> float:
    try:
        return float(getattr(settings, "AEO_EXTRACTION_TEMPERATURE", 0.1))
    except (TypeError, ValueError):
        return 0.1


def _extraction_max_tokens() -> int:
    try:
        return max(256, min(2048, int(getattr(settings, "AEO_EXTRACTION_MAX_TOKENS", 800))))
    except (TypeError, ValueError):
        return 800


def _extraction_timeout_seconds() -> float:
    try:
        return max(5.0, float(getattr(settings, "AEO_EXTRACTION_TIMEOUT", 60.0)))
    except (TypeError, ValueError):
        return 60.0


def _extraction_api_key_env() -> str:
    return getattr(settings, "AEO_EXTRACTION_API_KEY_ENV", DEFAULT_API_KEY_ENV)


def _extraction_openai_client():
    base = _get_client(_extraction_api_key_env())
    timeout = _extraction_timeout_seconds()
    try:
        return base.with_options(timeout=timeout, max_retries=0)
    except Exception:
        return base


def _strip_llm_json_fence(text: str) -> str:
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    raw = raw[3:].lstrip()
    if raw.lower().startswith("json"):
        raw = raw[4:].lstrip()
    if raw.startswith("\n"):
        raw = raw[1:]
    if "```" in raw:
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def _format_competitor_hints(hints: str | Sequence[str] | None) -> str:
    if hints is None:
        return "(none)"
    if isinstance(hints, str):
        t = hints.strip()
        return t if t else "(none)"
    parts = [str(x).strip() for x in hints if str(x).strip()]
    return "\n".join(parts) if parts else "(none)"


def root_domain_from_fragment(text: str) -> str | None:
    """
    Normalize a URL or bare host to a lowercase root-style host (strip www, path, query, port).
    """
    s = (text or "").strip().strip("<>").strip()
    if not s:
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        candidate = s.split()[0].split("/")[0]
    else:
        parsed = urlparse(s)
        candidate = parsed.netloc or parsed.path.split("/")[0]
    candidate = candidate.lower()
    if "@" in candidate:
        candidate = candidate.split("@")[-1]
    if candidate.startswith("www."):
        candidate = candidate[4:]
    if ":" in candidate:
        candidate = candidate.split(":")[0]
    candidate = candidate.rstrip(".")
    if not candidate or "." not in candidate:
        return None
    if not re.match(r"^[a-z0-9.-]+$", candidate):
        return None
    return candidate


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(item).strip())
    return out


def _extract_domains_from_raw_answer(raw_text: str) -> list[str]:
    """
    Best-effort domain extraction fallback from raw answer text.
    Merges URL-like and bare-domain patterns, normalized to root domains.
    """
    text = (raw_text or "").strip()
    if not text:
        return []
    patterns = (
        r"https?://[^\s)>\]\"']+",
        r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b",
    )
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(re.findall(pat, text, flags=re.IGNORECASE))
    domains: list[str] = []
    for cand in candidates:
        dom = root_domain_from_fragment(cand)
        if dom:
            domains.append(dom)
    return _dedupe_preserve_order(domains)[:100]


def _sanitize_competitors(items: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(items, list):
        return out
    seen: set[str] = set()
    for item in items:
        name = str(item).strip()
        if len(name) < 2 or len(name) > 200:
            continue
        low = name.lower()
        if low in GENERIC_COMPETITOR_TOKENS:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(name)
    return out[:50]


def _sanitize_ranking_order(items: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        name = str(item).strip()
        if len(name) < 2 or len(name) > 200:
            continue
        low = name.lower()
        if low in GENERIC_COMPETITOR_TOKENS:
            continue
        out.append(name)
    return _dedupe_preserve_order(out)[:50]


def _sanitize_citations(items: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(items, list):
        return out
    seen: set[str] = set()
    for item in items:
        dom = root_domain_from_fragment(str(item))
        if not dom or dom in seen:
            continue
        seen.add(dom)
        out.append(dom)
    return out[:50]


def normalize_extraction_payload(data: Mapping[str, Any], *, raw_response: str = "") -> dict[str, Any]:
    """
    Coerce model JSON into canonical shape for storage and APIs.
    """
    brand = bool(data.get("brand_mentioned"))
    pos = str(data.get("mention_position") or "none").lower().strip()
    if pos not in _VALID_POSITIONS:
        pos = "none"
    if not brand:
        pos = "none"

    try:
        count = int(data.get("mention_count", 0))
    except (TypeError, ValueError):
        count = 0
    count = max(0, min(1000, count))
    if not brand:
        count = 0

    competitors = _sanitize_competitors(data.get("competitors"))
    ranking_order = _sanitize_ranking_order(data.get("ranking_order"))
    # Keep ranking order useful for weighted scoring: include named mentions in order.
    if not ranking_order:
        ranking_order = _dedupe_preserve_order([*competitors])

    model_citations = _sanitize_citations(data.get("citations"))
    regex_citations = _extract_domains_from_raw_answer(raw_response)
    citations = _dedupe_preserve_order([*model_citations, *regex_citations])[:50]

    sent = str(data.get("sentiment") or "neutral").lower().strip()
    if sent not in _VALID_SENTIMENT:
        sent = "neutral"

    conf: float | None
    try:
        c = float(data["confidence_score"])
        if c != c:  # NaN
            conf = None
        else:
            conf = max(0.0, min(1.0, c))
    except (KeyError, TypeError, ValueError):
        conf = None

    return {
        "brand_mentioned": brand,
        "mention_position": pos,
        "mention_count": count,
        "competitors": competitors,
        "ranking_order": ranking_order,
        "citations": citations,
        "sentiment": sent,
        "confidence_score": conf,
    }


def _default_failed_payload() -> dict[str, Any]:
    return normalize_extraction_payload(
        {
            "brand_mentioned": False,
            "mention_position": "none",
            "mention_count": 0,
            "competitors": [],
            "ranking_order": [],
            "citations": [],
            "sentiment": "neutral",
            "confidence_score": None,
        }
    )


def _parse_extraction_json(raw_text: str) -> dict[str, Any] | None:
    cleaned = _strip_llm_json_fence(raw_text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    required = (
        "brand_mentioned",
        "mention_position",
        "mention_count",
        "competitors",
        "citations",
        "sentiment",
    )
    if not all(k in obj for k in required):
        return None
    return obj


def _call_extraction_openai(user_content: str) -> str:
    client = _extraction_openai_client()
    try:
        completion = client.chat.completions.create(
            model=_extraction_model(),
            messages=[
                {"role": "system", "content": AEO_STRUCTURED_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=_extraction_temperature(),
            max_tokens=_extraction_max_tokens(),
        )
        return (
            (completion.choices[0].message.content or "").strip()
            if completion.choices
            else ""
        )
    finally:
        if hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass


def _build_user_content(
    *,
    raw_response: str,
    business_profile: BusinessProfile,
    competitor_hints: str | Sequence[str] | None,
    prompt_text: str = "",
) -> str:
    name = (getattr(business_profile, "business_name", None) or "").strip() or "(unknown)"
    industry = (getattr(business_profile, "industry", None) or "").strip() or "(unknown)"
    return AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE.format(
        business_name=name,
        industry=industry,
        competitor_hints=_format_competitor_hints(competitor_hints),
        prompt_text=(prompt_text or "").strip() or "(unknown)",
        raw_response=raw_response or "",
    )


def extract_aeo_response(
    raw_response: str,
    business_profile: BusinessProfile,
    *,
    competitor_hints: str | Sequence[str] | None = None,
    prompt_text: str = "",
) -> dict[str, Any]:
    """
    Second-pass OpenAI extraction; returns normalized dict plus parse_ok / raw_llm metadata.

    On failure after one retry, returns safe defaults with parse_ok False (does not raise).
    """
    user_content = _build_user_content(
        raw_response=raw_response,
        business_profile=business_profile,
        competitor_hints=competitor_hints,
        prompt_text=prompt_text,
    )
    model = _extraction_model()
    raw_llm = ""
    parsed: dict[str, Any] | None = None

    try:
        raw_llm = _call_extraction_openai(user_content)
        parsed = _parse_extraction_json(raw_llm)
        if parsed is None:
            logger.warning("AEO extraction JSON parse failed; retrying once")
            raw_llm = _call_extraction_openai(user_content + AEO_STRUCTURED_EXTRACTION_RETRY_SUFFIX)
            parsed = _parse_extraction_json(raw_llm)
    except Exception as exc:
        logger.exception("AEO extraction OpenAI call failed: %s", exc)
        parsed = None

    if parsed is None:
        normalized = _default_failed_payload()
        return {
            **normalized,
            "parse_ok": False,
            "raw_llm": raw_llm,
            "extraction_model": model,
        }

    normalized = normalize_extraction_payload(parsed, raw_response=raw_response)
    return {
        **normalized,
        "parse_ok": True,
        "raw_llm": raw_llm,
        "extraction_model": model,
    }


def save_extraction_result(
    *,
    response_snapshot: AEOResponseSnapshot,
    data: Mapping[str, Any],
    extraction_model: str,
    parse_failed: bool = False,
) -> AEOExtractionSnapshot:
    """
    Persist extraction output linked to the Phase 2 response row.
    """
    norm = normalize_extraction_payload(data) if not parse_failed else _default_failed_payload()
    return AEOExtractionSnapshot.objects.create(
        response_snapshot=response_snapshot,
        brand_mentioned=norm["brand_mentioned"],
        mention_position=norm["mention_position"],
        mention_count=norm["mention_count"],
        competitors_json=norm["competitors"],
        citations_json=norm["citations"],
        sentiment=norm["sentiment"],
        confidence_score=norm["confidence_score"],
        extraction_model=(extraction_model or "")[:128],
        extraction_parse_failed=parse_failed,
    )


def run_single_extraction(
    snapshot: AEOResponseSnapshot,
    *,
    save: bool = True,
    competitor_hints: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Extract from one AEOResponseSnapshot; optionally writes AEOExtractionSnapshot.

    Default competitor hints: profile's seo_competitor_domains_override (comma-separated).
    """
    profile = snapshot.profile
    hints = competitor_hints
    if hints is None:
        raw = (getattr(profile, "seo_competitor_domains_override", None) or "").strip()
        hints = [x.strip() for x in raw.split(",") if x.strip()] if raw else None

    result = extract_aeo_response(
        snapshot.raw_response or "",
        profile,
        competitor_hints=hints,
        prompt_text=snapshot.prompt_text or "",
    )
    parse_ok = bool(result.pop("parse_ok", False))
    raw_llm = result.pop("raw_llm", "")
    model = str(result.pop("extraction_model", "") or _extraction_model())

    snapshot_id: int | None = None
    save_error: str | None = None
    if save:
        try:
            row = save_extraction_result(
                response_snapshot=snapshot,
                data=result,
                extraction_model=model,
                parse_failed=not parse_ok,
            )
            snapshot_id = row.id
        except Exception as exc:
            save_error = f"{type(exc).__name__}: {exc}"
            logger.exception("AEO extraction save failed for response_snapshot_id=%s", snapshot.id)

    return {
        "extraction_snapshot_id": snapshot_id,
        "parse_ok": parse_ok,
        "save_error": save_error,
        "response_snapshot_id": snapshot.id,
        "profile_id": profile.id,
        "brand_mentioned": result["brand_mentioned"],
        "mention_position": result["mention_position"],
        "mention_count": result["mention_count"],
        "competitors": result["competitors"],
        "ranking_order": result.get("ranking_order") or [],
        "citations": result["citations"],
        "sentiment": result["sentiment"],
        "confidence_score": result["confidence_score"],
        "extraction_model": model,
        "raw_llm_preview": raw_llm[:500] if raw_llm else "",
    }
