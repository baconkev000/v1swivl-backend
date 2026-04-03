"""
Phase 3: structured extraction from raw AEO answers (second-pass OpenAI JSON).

Keeps execution (Phase 2) separate; prompt strings for extraction live in aeo_prompts.py.

Optional settings:
    AEO_EXTRACTION_PARSER_MODEL — env; defaults to OPENAI_MODEL (see settings.base)
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
from ..openai_utils import _get_client, _get_model, chat_completion_create_logged
from .aeo_prompts import (
    AEO_STRUCTURED_EXTRACTION_RETRY_SUFFIX,
    AEO_STRUCTURED_EXTRACTION_SYSTEM_PROMPT,
    AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE,
    GENERIC_COMPETITOR_TOKENS,
)

logger = logging.getLogger(__name__)

DEFAULT_API_KEY_ENV = "OPEN_AI_SEO_API_KEY"
_VALID_SENTIMENT = frozenset({"positive", "neutral", "negative"})


def _normalize_for_brand_match(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def programmatic_tracked_brand_from_urls(
    tracked_website_domain: str,
    raw_response: str,
    competitors: list[dict[str, str]],
) -> tuple[bool, int]:
    """
    True when the tracked site's registrable root matches a host extracted from ``raw_response``
    or from any sanitized competitor ``url`` (``registered_root_domains_match``; not LLM judgment).
    ``mention_count`` is the number of distinct matched host roots (minimum 1 when cited).
    """
    tracked_root = root_domain_from_fragment(tracked_website_domain) or ""
    tracked_root = tracked_root.strip().lower().rstrip(".")
    if not tracked_root:
        return False, 0
    matched_hosts: set[str] = set()
    for c in competitors:
        url = (c.get("url") or "").strip()
        if not url:
            continue
        h = root_domain_from_fragment(url)
        if h and registered_root_domains_match(tracked_root, h):
            matched_hosts.add(h)
    for h in _extract_domains_from_raw_answer(raw_response):
        if registered_root_domains_match(tracked_root, h):
            matched_hosts.add(h)
    if not matched_hosts:
        return False, 0
    return True, max(1, len(matched_hosts))


def _domain_grounds_brand(raw_text: str, website_domain: str) -> bool:
    """
    True only if the answer text contains a hostname whose registrable root matches the tracked
    site (exact match, subdomain of tracked, or tracked is subdomain of host) via
    registered_root_domains_match on roots from _extract_domains_from_raw_answer — never via a
    naive first-label substring of the compact text (avoids saltlakedental inside saltlakedentalcare).
    """
    return programmatic_tracked_brand_from_urls(website_domain, raw_text, [])[0]


def _extraction_model() -> str:
    raw = getattr(settings, "AEO_EXTRACTION_PARSER_MODEL", "") or ""
    s = str(raw).strip()
    return s or _get_model()


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


def registered_root_domains_match(tracked_root: str, host_root: str) -> bool:
    """
    True when ``host_root`` is the same site as ``tracked_root`` (subdomains count as the same brand).
    Both inputs should be lowercase host-style roots (e.g. ``saltlakedentalcare.com``).
    """
    a = (tracked_root or "").strip().lower().rstrip(".")
    b = (host_root or "").strip().lower().rstrip(".")
    if not a or not b:
        return False
    if a == b:
        return True
    if b.endswith("." + a):
        return True
    if a.endswith("." + b):
        return True
    return False


def tracked_domain_listed_in_competitors(tracked_website_domain: str, competitors: Any) -> bool:
    """
    True when any structured competitor row links to the same site as the business website.

    Extraction models often put the target business only inside ``competitors`` (with URL)
    while setting ``brand_mentioned`` false; coverage and scoring should still count that as a cite.
    """
    t = (tracked_website_domain or "").strip().lower().rstrip(".")
    if not t or not competitors:
        return False
    if not isinstance(competitors, (list, tuple)):
        return False
    for raw in competitors:
        parsed = parse_competitor_raw_item(raw)
        url = (parsed.get("url") or "").strip()
        if not url:
            continue
        comp_dom = root_domain_from_fragment(url)
        if comp_dom and registered_root_domains_match(t, comp_dom):
            return True
    return False


def _domain_fallback_display(domain: str) -> str:
    """Human-readable label when no competitor name maps to this citation domain."""
    d = (domain or "").strip().lower().rstrip(".")
    if not d:
        return "Unknown"
    first = d.split(".")[0].replace("-", " ").strip()
    if not first:
        return d
    return first[0].upper() + first[1:].lower() if len(first) > 1 else first.upper()


def competitor_display_name_for_citation_domain(competitors_json: Any, cite_domain: str) -> str | None:
    """
    First non-empty competitor ``name`` whose URL registrable root matches ``cite_domain``.
    """
    cite_root = root_domain_from_fragment(cite_domain) or (cite_domain or "").strip().lower().rstrip(".")
    if not cite_root or not isinstance(competitors_json, list):
        return None
    for raw in competitors_json:
        p = parse_competitor_raw_item(raw)
        url = (p.get("url") or "").strip()
        name = (p.get("name") or "").strip()
        comp_dom = root_domain_from_fragment(url)
        if comp_dom and registered_root_domains_match(cite_root, comp_dom) and len(name) >= 2:
            return name
    return None


def competitor_url_for_citation_domain(competitors_json: Any, cite_domain: str) -> str | None:
    """First competitor ``url`` whose registrable root matches ``cite_domain``."""
    cite_root = root_domain_from_fragment(cite_domain) or (cite_domain or "").strip().lower().rstrip(".")
    if not cite_root or not isinstance(competitors_json, list):
        return None
    for raw in competitors_json:
        p = parse_competitor_raw_item(raw)
        url = (p.get("url") or "").strip()
        if not url:
            continue
        comp_dom = root_domain_from_fragment(url)
        if comp_dom and registered_root_domains_match(cite_root, comp_dom):
            return url.split("?")[0].rstrip("/")
    return None


def _citation_row_display_url(
    raw_citation_item: str,
    cite_domain: str,
    competitors_json: Any,
    *,
    citation_is_tracked_domain: bool,
    tracked_website_url_or_domain: str,
    tracked_root: str,
) -> str:
    """Stable URL for tooltips: raw citation, tracked site when citation is yours, else competitor match or domain root."""
    raw = str(raw_citation_item or "").strip()
    if raw.lower().startswith(("http://", "https://")):
        try:
            parsed = urlparse(raw)
            if parsed.netloc:
                return raw.split("?")[0].rstrip("/")
        except Exception:
            pass
    if citation_is_tracked_domain:
        tw = (tracked_website_url_or_domain or "").strip()
        if tw.lower().startswith(("http://", "https://")):
            return tw.split("?")[0].rstrip("/")
        if tracked_root:
            return f"https://{tracked_root}/"
    matched = competitor_url_for_citation_domain(competitors_json, cite_domain)
    if matched:
        return matched
    return f"https://{cite_domain}/"


def _pick_better_citation_url(a: str, b: str) -> str:
    """Prefer https and longer (more specific) URLs when merging platform rows."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    a_h = a.lower().startswith("https://")
    b_h = b.lower().startswith("https://")
    if a_h and not b_h:
        return a
    if b_h and not a_h:
        return b
    return a if len(a) >= len(b) else b


def citations_ranking_for_prompt_coverage(
    citations_json: Any,
    competitors_json: Any,
    *,
    tracked_website_url_or_domain: str,
    brand_mentioned: bool,
    tracked_business_name: str,
) -> tuple[list[dict[str, Any]], int | None]:
    """
    Build UI rows from stored ``citations_json`` (ordered root domains).

    Display names come from ``competitors_json`` when a row's URL matches that domain; otherwise a
    short label derived from the domain. When ``brand_mentioned`` is true and a citation domain
    matches the profile site, the row uses ``tracked_business_name`` and ``is_target`` is true.

    Returns ``(rows, target_url_position)`` where ``target_url_position`` is the 1-based index of
    the first citation whose domain matches the tracked site, or ``None``.
    """
    tracked_root = root_domain_from_fragment(tracked_website_url_or_domain) or ""
    tracked_root = (tracked_root or (tracked_website_url_or_domain or "").strip().lower()).rstrip(".")
    rows: list[dict[str, Any]] = []
    target_pos: int | None = None
    raw_list = citations_json if isinstance(citations_json, list) else []
    pos_counter = 0
    for item in raw_list:
        cite_dom = root_domain_from_fragment(str(item))
        if not cite_dom:
            continue
        pos_counter += 1
        is_domain_target = bool(tracked_root and registered_root_domains_match(tracked_root, cite_dom))
        if is_domain_target and target_pos is None:
            target_pos = pos_counter

        is_target = bool(is_domain_target and brand_mentioned)
        if brand_mentioned and is_domain_target:
            tb = (tracked_business_name or "").strip()
            display_name = tb if tb else _domain_fallback_display(cite_dom)
        else:
            looked = competitor_display_name_for_citation_domain(competitors_json, cite_dom)
            display_name = looked if looked else _domain_fallback_display(cite_dom)

        row_url = _citation_row_display_url(
            str(item),
            cite_dom,
            competitors_json,
            citation_is_tracked_domain=is_domain_target,
            tracked_website_url_or_domain=tracked_website_url_or_domain,
            tracked_root=tracked_root,
        )
        rows.append(
            {
                "name": display_name,
                "position": pos_counter,
                "is_target": is_target,
                "url": row_url,
            }
        )
    return rows, target_pos


def merge_citations_rankings_across_platform_cells(platform_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge ``citations_ranking`` from multiple platform cells (e.g. OpenAI + Gemini).

    Union by case-insensitive display name: keep the best (lowest) 1-based citation position;
    ``is_target`` is true if true on any platform. Sort by best position then name, then assign
    contiguous positions 1..n for the combined rankings table.
    """
    aggregated: dict[str, dict[str, Any]] = {}
    for cell in platform_cells:
        if not isinstance(cell, dict) or not cell.get("has_data"):
            continue
        ranking = cell.get("citations_ranking") or []
        if not isinstance(ranking, list):
            continue
        for row in ranking:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            key = name.casefold()
            raw_pos = row.get("position")
            try:
                pos_int = int(raw_pos) if raw_pos is not None else 10**9
            except (TypeError, ValueError):
                pos_int = 10**9
            is_target = bool(row.get("is_target"))
            row_url = str(row.get("url") or "").strip()
            if key not in aggregated:
                aggregated[key] = {
                    "name": name,
                    "best_pos": pos_int,
                    "is_target": is_target,
                    "url": row_url,
                }
            else:
                cur = aggregated[key]
                cur["best_pos"] = min(int(cur["best_pos"]), pos_int)
                cur["is_target"] = bool(cur["is_target"]) or is_target
                if len(name) > len(str(cur["name"] or "")):
                    cur["name"] = name
                cur["url"] = _pick_better_citation_url(str(cur.get("url") or ""), row_url)
    items = sorted(
        aggregated.values(),
        key=lambda x: (int(x["best_pos"]), str(x["name"] or "").lower()),
    )
    out: list[dict[str, Any]] = []
    for i, it in enumerate(items, start=1):
        out.append(
            {
                "name": str(it["name"]),
                "position": i,
                "is_target": bool(it["is_target"]),
                "url": str(it.get("url") or ""),
            }
        )
    return out


def merged_target_url_position(merged_ranking: list[dict[str, Any]]) -> int | None:
    """1-based position of the target row in a merged ranking list, if any."""
    for row in merged_ranking:
        if isinstance(row, dict) and row.get("is_target"):
            try:
                return int(row["position"])
            except (TypeError, ValueError, KeyError):
                continue
    return None


def unique_business_count_excluding_target(merged_ranking: list[dict[str, Any]]) -> int:
    """
    Count unique businesses in a merged ``citations_ranking``, excluding the tracked business
    when it appears as ``is_target`` (so this is "other" brands in the extraction).
    """
    if not merged_ranking:
        return 0
    target_in = any(isinstance(r, dict) and bool(r.get("is_target")) for r in merged_ranking)
    n = len(merged_ranking)
    return max(0, n - (1 if target_in else 0))


def brand_effectively_cited(
    brand_mentioned: bool,
    competitors_json: Any,
    *,
    tracked_website_url_or_domain: str = "",
) -> bool:
    """
    Whether APIs and UI should treat the business as cited for this extraction.

    Uses stored ``brand_mentioned`` or a URL match against ``competitors_json``.
    ``tracked_website_url_or_domain`` may be a full URL or bare host; it is normalized.
    """
    if brand_mentioned:
        return True
    rooted = root_domain_from_fragment(tracked_website_url_or_domain) or ""
    rooted = (rooted or (tracked_website_url_or_domain or "").strip().lower()).rstrip(".")
    return tracked_domain_listed_in_competitors(rooted, competitors_json)


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

    ``brand_mentioned``, ``mention_count``, and ``mention_position`` are derived only in code: the
    tracked profile domain must match a host from URLs in the raw answer or from sanitized
    competitor rows (``registered_root_domains_match``). The model must not supply brand fields;
    any legacy keys in ``data`` are ignored for brand. ``tracked_business_name`` is accepted for
    API compatibility only.
    """
    raw = (raw_response or "").strip()
    domain = (tracked_website_domain or "").strip()

    competitors = _sanitize_competitors(data.get("competitors"))
    brand, count = programmatic_tracked_brand_from_urls(domain, raw, competitors)
    count = max(0, min(1000, count))
    pos: str = "middle" if brand else "none"
    if brand:
        count = max(1, count)

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
            "competitors": [],
            "ranking_order": [],
            "citations": [],
            "sentiment": "neutral",
            "confidence_score": None,
        },
        raw_response="",
        tracked_website_domain="",
    )


def _parse_extraction_json(raw_text: str) -> dict[str, Any] | None:
    cleaned = _strip_llm_json_fence(raw_text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    required = ("competitors", "citations", "sentiment")
    if not all(k in obj for k in required):
        return None
    return obj


def _call_extraction_openai(
    user_content: str,
    *,
    business_profile: BusinessProfile | None = None,
) -> str:
    client = _extraction_openai_client()
    try:
        completion = chat_completion_create_logged(
            client,
            operation="openai.chat.aeo_structured_extraction",
            business_profile=business_profile,
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
    site = (getattr(business_profile, "website_url", None) or "").strip()
    domain = (root_domain_from_fragment(site) or "").strip() or "(unknown)"
    return AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE.format(
        business_name=name,
        business_domain=domain,
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
        raw_llm = _call_extraction_openai(
            user_content,
            business_profile=business_profile,
        )
        parsed = _parse_extraction_json(raw_llm)
        if parsed is None:
            logger.warning("AEO extraction JSON parse failed; retrying once")
            raw_llm = _call_extraction_openai(
                user_content + AEO_STRUCTURED_EXTRACTION_RETRY_SUFFIX,
                business_profile=business_profile,
            )
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
