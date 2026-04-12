"""
Onboarding review topics (domain-only input).

Primary: Perplexity Chat Completions when ``PERPLEXITY_API_KEY`` is set.
Optional: Gemini via ``ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK`` (default off).

Templates live in ``accounts.prompts.review_topics`` (shared JSON schema).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from django.conf import settings

from .aeo.perplexity_execution_utils import (
    get_perplexity_aeo_model,
    perplexity_chat_completion_raw,
    perplexity_execution_enabled,
)
from .gemini_utils import generate_gemini_execution_text, get_gemini_execution_model
from .models import BusinessProfile
from .prompts.review_topics import (
    REVIEW_TOPICS_DOMAIN_PLACEHOLDER,
    REVIEW_TOPICS_SYSTEM_INSTRUCTION,
    REVIEW_TOPICS_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

MAX_REVIEW_TOPICS = 20


def _strip_code_fence(text: str) -> str:
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


def normalize_root_domain(domain: str) -> str:
    """Host only: no scheme, no path, no query (matches onboarding crawl ``domain`` field)."""
    d = (domain or "").strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = d.split("/")[0].split("?")[0].strip()
    if d.startswith("www."):
        d = d[4:]
    return d[:255]


def get_gemini_review_topics_model() -> str:
    override = (getattr(settings, "GEMINI_REVIEW_TOPICS_MODEL", None) or "").strip()
    if override:
        return override
    return get_gemini_execution_model()


def get_perplexity_onboarding_review_topics_model() -> str:
    v = (getattr(settings, "PERPLEXITY_ONBOARDING_REVIEW_TOPICS_MODEL", None) or "").strip()
    if v:
        return v
    return get_perplexity_aeo_model()


def parse_review_topics_json(raw_text: str) -> tuple[list[dict[str, Any]], str | None]:
    """
    Parse model output into a list of topic dicts.
    Returns (items, error_message). items is empty on failure.
    """
    cleaned = _strip_code_fence(raw_text)
    if not cleaned:
        return [], "review_topics_empty_model_response"
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return [], f"review_topics_invalid_json: {exc}"

    if not isinstance(data, dict):
        return [], "review_topics_json_not_object"
    topics = data.get("topics")
    if not isinstance(topics, list):
        return [], "review_topics_missing_topics_array"

    out: list[dict[str, Any]] = []
    for item in topics:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "").strip()
        topic = re.sub(r"\s+", " ", topic).rstrip(".!?")
        if len(topic) < 2:
            continue
        row: dict[str, Any] = {"topic": topic}
        cat = item.get("category")
        if isinstance(cat, str) and cat.strip():
            row["category"] = cat.strip()[:64]
        rat = item.get("rationale")
        if isinstance(rat, str) and rat.strip():
            row["rationale"] = rat.strip()[:500]
        out.append(row)

    if not out:
        return [], "review_topics_parsed_zero_valid_items"
    return out, None


def dedupe_and_cap_topics(items: list[dict[str, Any]], cap: int = MAX_REVIEW_TOPICS) -> list[dict[str, Any]]:
    """≤ cap unique topics by normalized label (first occurrence wins)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in items:
        t = str(row.get("topic") or "").strip()
        if not t:
            continue
        key = re.sub(r"\s+", " ", t.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= cap:
            break
    return out


def _fetch_review_topics_raw_perplexity(
    *,
    user_text: str,
    business_profile: BusinessProfile | None,
    host: str,
) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": REVIEW_TOPICS_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_text},
    ]
    raw, err = perplexity_chat_completion_raw(
        messages=messages,
        business_profile=business_profile,
        temperature=0.35,
        max_tokens=4096,
        model=get_perplexity_onboarding_review_topics_model(),
        log_operation="perplexity.chat.completions.onboarding_review_topics",
    )
    if err == "skipped_no_api_key":
        return "", "review_topics_perplexity_not_configured"
    if err:
        out_err = f"review_topics_perplexity_failed: {err}"
        logger.warning("[onboarding review_topics] domain=%s %s", host, out_err)
        return "", out_err
    if not (raw or "").strip():
        return "", "review_topics_empty_model_response"
    return raw.strip(), ""


def _fetch_review_topics_raw_gemini(
    *,
    user_text: str,
    business_profile: BusinessProfile | None,
    host: str,
) -> tuple[str, str]:
    raw, gem_err = generate_gemini_execution_text(
        system_instruction=REVIEW_TOPICS_SYSTEM_INSTRUCTION,
        user_text=user_text,
        temperature=0.35,
        max_output_tokens=4096,
        business_profile=business_profile,
        log_operation="gemini.generate_content.onboarding_review_topics",
        model_name_override=get_gemini_review_topics_model(),
    )
    if gem_err:
        err = f"review_topics_gemini_failed: {gem_err}"
        logger.warning("[onboarding review_topics] domain=%s %s", host, err)
        return "", err
    if not (raw or "").strip():
        return "", "review_topics_empty_model_response"
    return raw.strip(), ""


def generate_review_topics_for_domain(
    *,
    domain: str,
    business_profile: BusinessProfile | None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Produce ``review_topics`` for onboarding (same JSON shape as before).

    Uses Perplexity when ``PERPLEXITY_API_KEY`` is set; otherwise returns an error unless
    ``ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK`` is True (Gemini path).

    On success the second return value is ``""``.
    """
    host = normalize_root_domain(domain)
    if not host or "." not in host:
        return [], "review_topics_invalid_domain"

    user_text = REVIEW_TOPICS_USER_TEMPLATE.replace(REVIEW_TOPICS_DOMAIN_PLACEHOLDER, host)

    raw: str
    fetch_err: str

    if perplexity_execution_enabled():
        raw, fetch_err = _fetch_review_topics_raw_perplexity(
            user_text=user_text,
            business_profile=business_profile,
            host=host,
        )
    elif bool(getattr(settings, "ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK", False)):
        raw, fetch_err = _fetch_review_topics_raw_gemini(
            user_text=user_text,
            business_profile=business_profile,
            host=host,
        )
    else:
        logger.warning(
            "[onboarding review_topics] domain=%s skipped (no PERPLEXITY_API_KEY; Gemini fallback off)",
            host,
        )
        return [], "review_topics_perplexity_not_configured"

    if fetch_err:
        return [], fetch_err

    parsed, parse_err = parse_review_topics_json(raw)
    if parse_err:
        logger.warning("[onboarding review_topics] domain=%s %s", host, parse_err)
        return [], parse_err

    normalized = dedupe_and_cap_topics(parsed, cap=MAX_REVIEW_TOPICS)
    if not normalized:
        return [], "review_topics_empty_after_normalize"
    return normalized, ""
