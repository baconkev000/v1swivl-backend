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

import ast
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
_NAME_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "in",
        "at",
        "to",
        "inc",
        "llc",
        "ltd",
        "plc",
        "pc",
        "llp",
        "co",
    }
)


def _normalize_for_brand_match(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def _significant_name_tokens(name: str) -> list[str]:
    n = _normalize_for_brand_match(name)
    toks = [t for t in n.split() if len(t) >= 2 and t not in _NAME_STOPWORDS]
    return toks if toks else ([n] if n else [])


def _domain_grounds_brand(raw_text: str, website_domain: str) -> bool:
    if not website_domain:
        return False
    d = website_domain.strip().lower()
    if not d:
        return False
    raw_compact = re.sub(r"\s+", "", (raw_text or "").lower())
    if d in raw_compact:
        return True
    first = d.split(".", 1)[0]
    if len(first) >= 4 and first in raw_compact:
        return True
    return False


def _tracked_brand_grounded_in_text(
    raw_text: str,
    *,
    business_name: str,
    website_domain: str = "",
) -> bool:
    """
    True if the tracked business is present in the raw answer (case-insensitive),
    via full-name substring, all significant name tokens, or website domain.
    """
    raw_norm = _normalize_for_brand_match(raw_text)
    if not raw_norm:
        return False
    if _domain_grounds_brand(raw_text, website_domain):
        return True
    bn = (business_name or "").strip()
    if not bn:
        return False
    full = _normalize_for_brand_match(bn)
    if len(full) >= 3 and full in raw_norm:
        return True
    tokens = _significant_name_tokens(bn)
    if not tokens:
        return False
    if len(tokens) == 1:
        t = tokens[0]
        return len(t) >= 3 and t in raw_norm
    return all(t in raw_norm for t in tokens)


def _strict_parse_bool(value: Any) -> bool:
    """
    Strict bool parser for model outputs.
    - Accept bool directly
    - Accept "true"/"false" strings (case-insensitive)
    - Default False for all other types/values
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    return False


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


def _coerce_dict_like_string(s: str) -> dict[str, Any] | None:
    """
    Parse a string that is either JSON or a Python repr of a dict (e.g. LLM output
    ``"{'name': 'X', 'url': '...'}"`` stored as a string instead of a real JSON object).
    """
    raw = (s or "").strip()
    if not raw.startswith("{"):
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError, MemoryError):
        pass
    return None


def flatten_competitor_name_field(value: Any) -> str:
    """
    Coerce a competitor ``name`` field to a plain string.

    The model sometimes nests objects (e.g. ``name: { name: \"X\" }``); ``str(dict)``
    would otherwise produce a Python repr shown in the UI. String values may also be
    an entire ``{'name':..., 'url':...}`` blob.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
        coerced = _coerce_dict_like_string(s)
        if coerced is not None:
            return flatten_competitor_name_field(coerced.get("name"))
        return s
    if isinstance(value, dict):
        inner = value.get("name") or value.get("Name") or value.get("title")
        if inner is None:
            return ""
        if isinstance(inner, dict):
            return flatten_competitor_name_field(inner)
        return str(inner).strip()
    return str(value).strip()


def flatten_competitor_url_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
        coerced = _coerce_dict_like_string(s)
        if coerced is not None:
            return flatten_competitor_url_field(coerced.get("url"))
        return s
    if isinstance(value, dict):
        inner = value.get("url") or value.get("URL") or value.get("href")
        if inner is None:
            return ""
        if isinstance(inner, dict):
            return flatten_competitor_url_field(inner)
        return str(inner).strip()
    return str(value).strip()


def parse_competitor_raw_item(item: Any) -> dict[str, str]:
    """
    Normalize one competitor from the model or DB to ``{"name": str, "url": str}``.

    Handles real dicts, JSON strings, Python repr strings, and plain business names.
    """
    if item is None:
        return {"name": "", "url": ""}
    if isinstance(item, dict):
        return {
            "name": flatten_competitor_name_field(item.get("name")),
            "url": flatten_competitor_url_field(item.get("url")),
        }
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return {"name": "", "url": ""}
        coerced = _coerce_dict_like_string(s)
        if coerced is not None:
            return {
                "name": flatten_competitor_name_field(coerced.get("name")),
                "url": flatten_competitor_url_field(coerced.get("url")),
            }
        return {"name": s, "url": ""}
    return {"name": str(item).strip(), "url": ""}


def competitor_entry_display_name(entry: Any) -> str:
    """Normalize a stored competitor entry (``{name, url}`` object or legacy string) to a display name."""
    return parse_competitor_raw_item(entry)["name"]


def _competitor_url_dedupe_key(url: str) -> str | None:
    """
    Root domain used to treat duplicate competitor rows as the same business (same site).
    Returns None when no usable domain (dedupe by name only).
    """
    s = (url or "").strip()
    if not s:
        return None
    dom = root_domain_from_fragment(s)
    return dom


def _sanitize_competitors(items: Any) -> list[dict[str, str]]:
    """
    Coerce model output to ``[{"name": str, "url": str}, ...]``.

    Drops duplicate businesses: same root URL as a prior entry, or same name (case-insensitive)
    when URL is missing or not parseable to a domain.
    """
    out: list[dict[str, str]] = []
    if not isinstance(items, list):
        return out
    seen_domains: set[str] = set()
    seen_name_only: set[str] = set()

    for item in items:
        parsed = parse_competitor_raw_item(item)
        name = parsed["name"]
        url = parsed["url"]
        if len(name) < 2 or len(name) > 200:
            continue
        low = name.lower()
        if low in GENERIC_COMPETITOR_TOKENS:
            continue
        url = url[:2048] if url else ""

        dom_key = _competitor_url_dedupe_key(url)
        if dom_key:
            if dom_key in seen_domains:
                continue
            seen_domains.add(dom_key)
        else:
            nk = name.casefold()
            if nk in seen_name_only:
                continue
            seen_name_only.add(nk)

        out.append({"name": name, "url": url})
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


def normalize_extraction_payload(
    data: Mapping[str, Any],
    *,
    raw_response: str = "",
    tracked_business_name: str = "",
    tracked_website_domain: str = "",
) -> dict[str, Any]:
    """
    Coerce model JSON into canonical shape for storage and APIs.

    When ``tracked_business_name`` and ``raw_response`` are provided, ``brand_mentioned`` is
    grounded in the raw answer text (not only the model boolean). Competitors/citations from
    the model are kept even when brand flags are cleared.
    """
    brand = _strict_parse_bool(data.get("brand_mentioned"))
    pos = str(data.get("mention_position") or "none").lower().strip()
    if pos not in _VALID_POSITIONS:
        pos = "none"

    try:
        count = int(data.get("mention_count", 0))
    except (TypeError, ValueError):
        count = 0
    count = max(0, min(1000, count))

    name = (tracked_business_name or "").strip()
    raw = (raw_response or "").strip()
    domain = (tracked_website_domain or "").strip().lower()

    if name and not raw:
        if brand:
            logger.warning(
                "AEO extraction: brand_mentioned=True but no raw_response for grounding; overriding. business=%s",
                name,
            )
        brand = False
        pos = "none"
        count = 0
    elif name and raw:
        grounded = _tracked_brand_grounded_in_text(
            raw,
            business_name=name,
            website_domain=domain,
        )
        if brand and not grounded:
            logger.warning(
                "AEO extraction: model brand_mentioned=True but text did not ground target; overriding. "
                "business=%s snippet=%s",
                name,
                raw[:200],
            )
        if not grounded:
            brand = False
            pos = "none"
            count = 0
        else:
            brand = True
            if count == 0 and pos == "none":
                count = 1
                pos = "middle"
            elif count == 0:
                count = 1
            elif pos == "none":
                pos = "middle"
            count = max(1, count)

    if count == 0 and pos == "none":
        brand = False

    if not brand:
        pos = "none"
    if not brand:
        count = 0

    competitors = _sanitize_competitors(data.get("competitors"))
    ranking_order = _sanitize_ranking_order(data.get("ranking_order"))
    # Keep ranking order useful for weighted scoring: include named mentions in order.
    if not ranking_order:
        comp_names = [c["name"] for c in competitors if c.get("name")]
        ranking_order = _dedupe_preserve_order(comp_names)

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
    # Strict type guard for brand_mentioned to avoid truthy coercion bugs.
    bm = obj.get("brand_mentioned")
    if not isinstance(bm, bool):
        if not (isinstance(bm, str) and bm.strip().lower() in {"true", "false"}):
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

    normalized = normalize_extraction_payload(
        parsed,
        raw_response=raw_response,
        tracked_business_name=(getattr(business_profile, "business_name", None) or "").strip(),
        tracked_website_domain=root_domain_from_fragment(getattr(business_profile, "website_url", None) or "")
        or "",
    )
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
    profile = response_snapshot.profile
    tracked_name = (getattr(profile, "business_name", None) or "").strip()
    tracked_domain = root_domain_from_fragment(getattr(profile, "website_url", None) or "") or ""
    raw = response_snapshot.raw_response or ""
    norm = (
        normalize_extraction_payload(
            data,
            raw_response=raw,
            tracked_business_name=tracked_name,
            tracked_website_domain=tracked_domain,
        )
        if not parse_failed
        else _default_failed_payload()
    )
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
