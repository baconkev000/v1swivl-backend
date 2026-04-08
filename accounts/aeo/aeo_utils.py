"""
AEO prompt generation (Phase 1).

Execution and persistence live in aeo_execution_utils.py.
Uses template/metadata from aeo_prompts.py; batch *generation* uses openai_utils.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from string import Formatter
from typing import Any, Final, Mapping, Sequence
from urllib.parse import urlparse

from django.conf import settings

from ..models import BusinessProfile

# Onboarding / tracking: exactly this many prompts in combined output and saved profile lists.
AEO_ONBOARDING_PROMPT_COUNT: Final[int] = 50
from ..openai_utils import _get_client, _get_model, chat_completion_create_logged
from .aeo_prompts import (
    AEOPromptType,
    AEOPromptTemplateSpec,
    AEO_BATCH_JSON_SCHEMA_INSTRUCTION,
    AEO_BATCH_USER_PROMPT_INTRO,
    AEO_EXTRACTION_PREP_SYSTEM_PROMPT,
    AEO_PROMPT_ENGINE_SYSTEM_PROMPT_AUTHORITY,
    AEO_PROMPT_ENGINE_SYSTEM_PROMPT_COMPARISON,
    AEO_PROMPT_ENGINE_SYSTEM_PROMPT,
    AEO_PROMPT_ENGINE_SYSTEM_PROMPT_TRANSACTIONAL,
    AEO_PROMPT_ENGINE_SYSTEM_PROMPT_TRUST,
    DYNAMIC_BUSINESS_NAME_SPECS,
    DYNAMIC_PROMPT_SPECS,
)

OPENAI_ONBOARDING_TYPE_RATIO: Final[dict[str, float]] = {
    AEOPromptType.TRANSACTIONAL.value: 0.35,
    AEOPromptType.TRUST.value: 0.25,
    AEOPromptType.COMPARISON.value: 0.25,
    AEOPromptType.AUTHORITY.value: 0.15,
}
OPENAI_ONBOARDING_MAX_BATCH_SIZE: Final[int] = 24


@dataclass
class AEOPromptBusinessInput:
    """
    Serializable business context for prompt building (Celery / future DB friendly).

    Populate from BusinessProfile via aeo_business_input_from_profile() or construct directly.
    """

    industry: str = ""
    city: str = ""
    business_name: str = ""
    website_domain: str = ""
    services: list[str] = field(default_factory=list)
    niche_modifiers: list[str] = field(default_factory=list)
    differentiators: list[str] = field(default_factory=list)
    language: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def aeo_business_input_from_onboarding_payload(
    *,
    business_name: str,
    website_url: str,
    location: str,
    language: str = "",
    selected_topics: Sequence[str],
) -> AEOPromptBusinessInput:
    """
    Build prompt-plan context from onboarding step 1+2 POST body (not from profile fields).

    ``industry`` is a short hint derived from selected topic labels so OpenAI has vertical
    context without relying on BusinessProfile.industry mid-onboarding.
    """
    topics = [str(t).strip() for t in selected_topics if str(t).strip()]
    loc = (location or "").strip()
    city_guess = infer_city_from_address(loc) or _normalize_city(loc[:200])
    industry_hint = ", ".join(topics[:6]) if topics else ""
    return AEOPromptBusinessInput(
        industry=industry_hint[:500],
        city=city_guess,
        business_name=(business_name or "").strip(),
        website_domain=_website_domain_from_url(website_url or ""),
        services=list(topics),
        niche_modifiers=[],
        differentiators=[],
        language=(language or "").strip(),
    )


def assign_onboarding_prompts_to_selected_topics(
    combined: Sequence[Mapping[str, Any]],
    topic_order: Sequence[str],
) -> dict[str, list[str]]:
    """
    Map flat combined prompts onto step-2 topic tabs for the onboarding UI.

    Rule: total count matches len(combined); prompts are distributed in order so the first
    topic gets prompt[0], second gets prompt[1], …, then remaining prompts round-robin.
    This keeps AEO_ONBOARDING_PROMPT_COUNT (flat list) aligned with per-topic review in step 3.
    """
    texts = [str(p.get("prompt") or "").strip() for p in combined if str(p.get("prompt") or "").strip()]
    topics = [str(t).strip() for t in topic_order if str(t).strip()]
    out: dict[str, list[str]] = {t: [] for t in topics}
    if not topics or not texts:
        return out
    n_t, n_p = len(topics), len(texts)
    for i in range(min(n_t, n_p)):
        out[topics[i]].append(texts[i])
    for j in range(n_t, n_p):
        out[topics[j % n_t]].append(texts[j])
    return out


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


def _normalize_city(city: str) -> str:
    c = (city or "").strip()
    return re.sub(r"\s+", " ", c)


def _website_domain_from_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "https://" + u
    try:
        host = (urlparse(u).netloc or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


# --- LLM-only topic sanitization (onboarding keywords / services / industry) -----------------

# Registrable-label tokens this short or generic are not treated as brand roots for removal
# (avoids stripping "best", "usa", common TLD-ish labels from unrelated text).
_REGISTRABLE_ROOT_BLOCKLIST: Final[frozenset[str]] = frozenset(
    {
        "best",
        "us",
        "usa",
        "www",
        "com",
        "net",
        "org",
        "gov",
        "edu",
        "mail",
        "blog",
        "shop",
        "app",
        "dev",
        "api",
        "cdn",
        "ftp",
        "inc",
        "llc",
        "ltd",
        "corp",
        "co",
    }
)

_TOPIC_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "vs",
    }
)


def _normalize_topic_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _token_boundary_pattern(core: str) -> str:
    """
    Whole-token match for alnum + hyphen tokens (\\b breaks on hyphenated brands like mh-usa).
    """
    return rf"(?<![\w\-]){re.escape(core)}(?![\w\-])"


def _topic_only_stopwords(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return len(tokens) == 0 or all(t in _TOPIC_STOPWORDS for t in tokens)


def _remove_multiword_brand_phrase(text: str, business_name: str) -> str:
    phrase = _normalize_topic_whitespace(business_name)
    words = phrase.split()
    if len(words) < 2:
        return text
    parts = [_token_boundary_pattern(w) for w in words]
    gap = r"[\s\W]+"
    pattern = gap.join(parts)
    return re.sub(pattern, " ", text, flags=re.IGNORECASE)


def _remove_singleword_brand_token(text: str, token: str) -> str:
    t = _normalize_topic_whitespace(token)
    if len(t) < 4 or t.lower() in _REGISTRABLE_ROOT_BLOCKLIST:
        return text
    return re.sub(_token_boundary_pattern(t), " ", text, flags=re.IGNORECASE)


def _strip_host_literals(text: str, host: str) -> str:
    if not host:
        return text
    h = host.strip().lower()
    for variant in (h, "www." + h):
        text = re.sub(re.escape(variant), " ", text, flags=re.IGNORECASE)
    for prefix in ("https://", "http://"):
        text = re.sub(re.escape(prefix + h), " ", text, flags=re.IGNORECASE)
        text = re.sub(re.escape(prefix + "www." + h), " ", text, flags=re.IGNORECASE)
    return text


def _registrable_label_from_host(host: str) -> str:
    """
    First DNS label of the normalized host (same string shape as _website_domain_from_url).
    For mh-usa.example.co.uk returns mh-usa — not a naive split that drops multi-part public suffixes
    incorrectly; the *label* is still the leftmost segment, which matches typical branded hosts.
    """
    h = (host or "").strip().lower()
    if h.startswith("www."):
        h = h[4:]
    if not h:
        return ""
    return h.split(".", 1)[0]


def _strip_registrable_brand_tokens(text: str, host: str) -> str:
    root = _registrable_label_from_host(host)
    if len(root) < 4 or root.lower() in _REGISTRABLE_ROOT_BLOCKLIST:
        return text
    text = re.sub(_token_boundary_pattern(root), " ", text, flags=re.IGNORECASE)
    if "-" in root:
        text = re.sub(_token_boundary_pattern(root.replace("-", " ")), " ", text, flags=re.IGNORECASE)
    return text


def sanitize_topic(topic: str, business_name: str, website_domain: str) -> str:
    """
    Remove tracked business name and domain tokens from a topic string for LLM payloads only.

    Uses phrase-safe removal for multi-word business names and token boundaries that tolerate
    hyphens (so ``mh-usa`` and ``mh usa`` both match) without naive substring damage to unrelated
    words (e.g. ``super`` inside ``supercharger``).

    Edge cases (still possible): competitor brands embedded in keywords, extremely generic one-word
    business names, model non-compliance despite sanitized inputs.
    """
    original = _normalize_topic_whitespace(topic)
    if not original:
        return "this category"

    text = original
    bn = _normalize_topic_whitespace(business_name)
    host = (website_domain or "").strip().lower()

    if bn:
        bw = bn.split()
        if len(bw) >= 2:
            text = _remove_multiword_brand_phrase(text, bn)
        elif len(bw) == 1:
            text = _remove_singleword_brand_token(text, bw[0])

    text = _strip_host_literals(text, host)
    text = _strip_registrable_brand_tokens(text, host)

    text = _normalize_topic_whitespace(text)
    if not text or _topic_only_stopwords(text):
        return "this category"
    return text


def prompt_contains_tracked_brand_leakage(
    prompt: str,
    business_name: str,
    website_domain: str,
) -> bool:
    """
    Second-line defense: drop generated prompts that still mention the tracked brand or domain.

    Single-word business names only count when length >= 4 and not blocklisted (avoids matching
    generic words like ``best``). Registrable host labels use the same length/blocklist rules.
    """
    t = prompt or ""
    if not t:
        return False
    tl = t.lower()
    bn = _normalize_topic_whitespace(business_name)
    host = (website_domain or "").strip().lower()

    if bn:
        bw = bn.split()
        if len(bw) >= 2:
            parts = [_token_boundary_pattern(w) for w in bw]
            gap = r"[\s\W]+"
            if re.search(gap.join(parts), t, flags=re.IGNORECASE):
                return True
        elif len(bw) == 1:
            w = bw[0]
            if len(w) >= 4 and w.lower() not in _REGISTRABLE_ROOT_BLOCKLIST:
                if re.search(_token_boundary_pattern(w), t, flags=re.IGNORECASE):
                    return True

    if host:
        if host in tl:
            return True
        if f"www.{host}" in tl:
            return True

    root = _registrable_label_from_host(host)
    if len(root) >= 4 and root.lower() not in _REGISTRABLE_ROOT_BLOCKLIST:
        if re.search(_token_boundary_pattern(root), t, flags=re.IGNORECASE):
            return True
        if "-" in root and re.search(
            _token_boundary_pattern(root.replace("-", " ")),
            t,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def infer_city_from_address(address: str) -> str:
    """
    Best-effort city or region for AEO prompt templates (not street-level precision).

    Typical outputs: ``Salt Lake City``, ``Austin, TX``, ``Paris, France``. Omits street lines;
    when the string looks like an unparseable street-only line, returns empty so callers
    avoid leaking full addresses into prompts.
    """
    addr = (address or "").strip()
    if not addr:
        return ""
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    n = len(parts)
    if n >= 3:
        city_part = parts[-2]
        last = parts[-1]
        first_tok = last.split()[0] if last else ""
        if len(first_tok) == 2 and first_tok.isalpha():
            return _normalize_city(f"{city_part}, {first_tok.upper()}")
        return _normalize_city(city_part)
    if n == 2:
        left, right = parts[0], parts[1]
        if len(right) == 2 and right.isalpha():
            return _normalize_city(f"{left}, {right.upper()}")
        left_has_digit = any(ch.isdigit() for ch in left)
        if left_has_digit:
            return _normalize_city(right)
        return _normalize_city(f"{left}, {right}")
    seg = parts[0]
    if any(ch.isdigit() for ch in seg):
        return ""
    return _normalize_city(seg)


def aeo_business_input_from_profile(
    profile: BusinessProfile,
    *,
    city: str | None = None,
    industry: str | None = None,
    services: Sequence[str] | None = None,
    niche_modifiers: Sequence[str] | None = None,
    differentiators: Sequence[str] | None = None,
) -> AEOPromptBusinessInput:
    """
    Map a Django BusinessProfile into AEOPromptBusinessInput.

    Optional kwargs override inferred values (recommended until services live on the model).
    """
    addr = getattr(profile, "business_address", "") or ""
    resolved_city = _normalize_city(city) if city else infer_city_from_address(addr)
    if industry is not None:
        ind = str(industry).strip()
    else:
        ind = (getattr(profile, "industry", None) or "").strip()
    return AEOPromptBusinessInput(
        industry=ind,
        city=resolved_city,
        business_name=(getattr(profile, "business_name", None) or "").strip(),
        website_domain=_website_domain_from_url(getattr(profile, "website_url", None) or ""),
        services=list(services) if services is not None else [],
        niche_modifiers=list(niche_modifiers) if niche_modifiers is not None else [],
        differentiators=list(differentiators) if differentiators is not None else [],
    )


def _format_template(
    spec: AEOPromptTemplateSpec,
    *,
    city: str,
    service: str = "",
    modifier: str = "",
    differentiator: str = "",
    business_name: str = "",
    website_domain: str = "",
) -> str:
    values = {
        "city": city,
        "service": service,
        "modifier": modifier,
        "differentiator": differentiator,
        "business_name": business_name,
        "website_domain": website_domain,
    }
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(spec.template)
        if field_name
    }
    # Prevent runtime KeyError if any unexpected placeholder is introduced.
    safe_values = {k: values.get(k, "") for k in fields}
    return spec.template.format(**safe_values)


def plan_items_from_saved_prompt_strings(texts: Sequence[str]) -> list[dict[str, Any]]:
    """
    Normalize stored onboarding prompt strings into JSON-ready dicts (exactly up to
    ``AEO_ONBOARDING_PROMPT_COUNT`` items).
    """
    out: list[dict[str, Any]] = []
    for raw in texts:
        t = str(raw).strip()
        if not t:
            continue
        out.append(
            prompt_record(
                t,
                prompt_type=AEOPromptType.TRANSACTIONAL,
                weight=1.0,
                dynamic=True,
            )
        )
        if len(out) >= AEO_ONBOARDING_PROMPT_COUNT:
            break
    return out


def _split_combined_into_source_groups(
    combined: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
    dynamic: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition ``combined`` (in order) into fixed / dynamic / openai groups by prompt text."""
    fixed_texts = {str(x.get("prompt", "")).strip() for x in fixed}
    dynamic_texts = {str(x.get("prompt", "")).strip() for x in dynamic}
    out_fixed: list[dict[str, Any]] = []
    out_dynamic: list[dict[str, Any]] = []
    out_openai: list[dict[str, Any]] = []
    for item in combined:
        t = str(item.get("prompt", "")).strip()
        if t in fixed_texts:
            out_fixed.append(item)
        elif t in dynamic_texts:
            out_dynamic.append(item)
        else:
            out_openai.append(item)
    return out_fixed, out_dynamic, out_openai


def prompt_record(
    text: str,
    *,
    prompt_type: str | AEOPromptType,
    weight: float,
    dynamic: bool,
) -> dict[str, Any]:
    ptype = prompt_type.value if isinstance(prompt_type, AEOPromptType) else str(prompt_type)
    return {
        "prompt": re.sub(r"\s+", " ", (text or "").strip()),
        "type": ptype,
        "weight": float(weight),
        "dynamic": bool(dynamic),
    }


def generate_fixed_prompts(industry: str, city: str) -> list[dict[str, Any]]:
    """
    Fixed prompt generation is disabled for current onboarding flow.
    Keep function for API/backward compatibility.
    """
    return []


def _clean_token(s: str) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    return t[:80] if t else ""


def generate_dynamic_prompts(
    business_profile: BusinessProfile | AEOPromptBusinessInput | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Compose prompts from DYNAMIC_PROMPT_SPECS plus generic local-intent extensions
    (DYNAMIC_BUSINESS_NAME_SPECS — no brand or domain in consumer-facing text).

    Accepts BusinessProfile, AEOPromptBusinessInput, or a mapping with the same keys.
    """
    if isinstance(business_profile, BusinessProfile):
        ctx = aeo_business_input_from_profile(business_profile)
    elif isinstance(business_profile, AEOPromptBusinessInput):
        ctx = business_profile
    else:
        m = business_profile
        ctx = AEOPromptBusinessInput(
            industry=str(m.get("industry", "") or ""),
            city=str(m.get("city", "") or ""),
            business_name=str(m.get("business_name", "") or ""),
            website_domain=str(m.get("website_domain", "") or ""),
            services=list(m.get("services") or []),
            niche_modifiers=list(m.get("niche_modifiers") or []),
            differentiators=list(m.get("differentiators") or []),
        )

    c = _normalize_city(ctx.city)
    if not c:
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    services = [_clean_token(x) for x in ctx.services if _clean_token(x)]
    modifiers = [_clean_token(x) for x in ctx.niche_modifiers if _clean_token(x)]
    diffs = [_clean_token(x) for x in ctx.differentiators if _clean_token(x)]

    def _template_fields(template: str) -> set[str]:
        return {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name
        }

    def _value_sets_for_spec(spec: AEOPromptTemplateSpec) -> list[dict[str, str]]:
        fields = _template_fields(spec.template)
        candidates: list[dict[str, str]] = [{}]
        if "service" in fields:
            if not services:
                return []
            candidates = [{**c, "service": svc} for c in candidates for svc in services]
        if "modifier" in fields:
            if not modifiers:
                return []
            candidates = [{**c, "modifier": mod} for c in candidates for mod in modifiers]
        if "differentiator" in fields:
            if not diffs:
                return []
            candidates = [{**c, "differentiator": d} for c in candidates for d in diffs]
        return candidates

    for spec in DYNAMIC_PROMPT_SPECS:
        for params in _value_sets_for_spec(spec):
            text = _format_template(spec, city=c, **params)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                prompt_record(
                    text,
                    prompt_type=spec.prompt_type,
                    weight=spec.weight,
                    dynamic=True,
                )
            )

    for spec in DYNAMIC_BUSINESS_NAME_SPECS:
        text = _format_template(
            spec,
            city=c,
            business_name="",
            website_domain="",
        )
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(
            prompt_record(
                text,
                prompt_type=spec.prompt_type,
                weight=spec.weight,
                dynamic=True,
            )
        )

    return out


def combine_prompt_set(
    *prompt_lists: Sequence[Mapping[str, Any]],
    dedupe: bool = True,
) -> list[dict[str, Any]]:
    """
    Merge multiple JSON-ready prompt lists. When dedupe=True, drops same prompt text (casefold).
    """
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lst in prompt_lists:
        for item in lst:
            rec = normalize_aeo_prompt_dict(item)
            if not rec.get("prompt"):
                continue
            k = str(rec["prompt"]).casefold()
            if dedupe and k in seen:
                continue
            if dedupe:
                seen.add(k)
            merged.append(rec)
    return merged


def normalize_aeo_prompt_dict(item: Mapping[str, Any]) -> dict[str, Any]:
    """
    Coerce arbitrary mapping to the standard JSON-ready shape.
    """
    text = str(item.get("prompt", "") or "").strip()
    ptype = str(item.get("type", AEOPromptType.TRANSACTIONAL.value) or "").strip()
    if ptype not in {e.value for e in AEOPromptType}:
        ptype = AEOPromptType.TRANSACTIONAL.value
    try:
        w = float(item.get("weight", 1.0))
    except (TypeError, ValueError):
        w = 1.0
    dyn = item.get("dynamic")
    if dyn is None:
        dyn = False
    return prompt_record(text, prompt_type=ptype, weight=w, dynamic=bool(dyn))


def build_openai_batch_user_content(
    business: AEOPromptBusinessInput,
    seed_prompts: Sequence[Mapping[str, Any]] | None,
    max_additional: int,
    *,
    onboarding_topic_details: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    biz_for_llm = copy.deepcopy(business.as_dict())
    bn = business.business_name
    wd = business.website_domain
    svc = business.services or []
    biz_for_llm["services"] = [sanitize_topic(str(s), bn, wd) for s in svc]
    ind_raw = str(biz_for_llm.get("industry") or "")
    if ind_raw.strip():
        ind_parts = [p.strip() for p in ind_raw.split(",") if p.strip()]
        biz_for_llm["industry"] = ", ".join(sanitize_topic(p, bn, wd) for p in ind_parts)[:500]
    lines = [
        AEO_BATCH_USER_PROMPT_INTRO,
        "Business context (JSON):\n",
        json.dumps(biz_for_llm, ensure_ascii=False),
        f"\n\nGenerate at most {max_additional} new prompts not overlapping the seed list.",
        "\n\nSeed prompts (JSON array, may be empty):\n",
        json.dumps(list(seed_prompts or []), ensure_ascii=False),
    ]
    if onboarding_topic_details:
        sanitized_rows: list[dict[str, Any]] = []
        for row in onboarding_topic_details:
            r = copy.deepcopy(dict(row))
            kw = str(r.get("keyword") or "")
            r["keyword"] = sanitize_topic(kw, bn, wd)
            sanitized_rows.append(r)
        lines.extend(
            [
                "\n\nUser-selected monitoring topics from onboarding (prioritize natural consumer "
                "or buyer questions that relate to these themes; vary wording). "
                "JSON may include Labs/AEO metadata (search volume, rank, category) — use as weak hints, not hard constraints:\n",
                json.dumps(sanitized_rows, ensure_ascii=False),
            ],
        )
    lines.extend(["\n\n", AEO_BATCH_JSON_SCHEMA_INSTRUCTION])
    return "".join(lines)


def run_prompt_batch_via_openai(
    business: AEOPromptBusinessInput,
    seed_prompts: Sequence[Mapping[str, Any]] | None = None,
    *,
    max_additional: int = 12,
    system_prompt: str | None = None,
    api_key_env: str = "OPEN_AI_SEO_API_KEY",
    model: str | None = None,
    onboarding_topic_details: Sequence[Mapping[str, Any]] | None = None,
    business_profile: BusinessProfile | None = None,
) -> list[dict[str, Any]]:
    """
    Ask OpenAI for additional prompts; returns normalized JSON-ready dicts.

    Uses the same API key resolution as SEO/AEO helpers in openai_utils.
    """
    user_content = build_openai_batch_user_content(
        business,
        seed_prompts,
        max_additional=max_additional,
        onboarding_topic_details=onboarding_topic_details,
    )
    sys_p = system_prompt or AEO_PROMPT_ENGINE_SYSTEM_PROMPT
    use_model = model or _get_model()

    try:
        client = _get_client(api_key_env)
        completion = chat_completion_create_logged(
            client,
            operation="openai.chat.aeo_prompt_batch",
            business_profile=business_profile,
            model=use_model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            top_p=0.9,
            max_tokens=500,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        return []
    parsed = parse_aeo_prompt_json_array(raw)
    return [
        p
        for p in parsed
        if not prompt_contains_tracked_brand_leakage(
            str(p.get("prompt") or ""),
            business.business_name,
            business.website_domain,
        )
    ]


def parse_aeo_prompt_json_array(raw_text: str) -> list[dict[str, Any]]:
    """
    Parse model output into normalized prompt dicts; returns [] on failure.
    """
    try:
        payload = json.loads(_strip_code_fence(raw_text))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for el in payload:
        if isinstance(el, Mapping):
            out.append(normalize_aeo_prompt_dict(el))
    return [p for p in out if p.get("prompt")]


def prepare_structured_extraction_input(
    raw_completion: str,
    *,
    business: AEOPromptBusinessInput | None = None,
) -> dict[str, Any]:
    """
    Shape raw text for a future extraction step (no second model call yet).

    Returns a dict you can persist or pass to a later Celery task.
    """
    parsed = parse_aeo_prompt_json_array(raw_completion)
    return {
        "extraction_system_prompt": AEO_EXTRACTION_PREP_SYSTEM_PROMPT,
        "raw_completion": raw_completion,
        "parsed_prompts": parsed,
        "business": business.as_dict() if business else None,
        "openai_model": _get_model(),
    }


def build_full_aeo_prompt_plan(
    profile: BusinessProfile,
    *,
    services: Sequence[str] | None = None,
    niche_modifiers: Sequence[str] | None = None,
    differentiators: Sequence[str] | None = None,
    city: str | None = None,
    industry: str | None = None,
    include_openai: bool = False,
    max_openai_prompts: int | None = None,
    target_combined_count: int | None = None,
    business_input: AEOPromptBusinessInput | None = None,
    onboarding_topic_details: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build onboarding prompt plan using OpenAI generation only.

    Total combined prompt count = ``target_combined_count`` or ``AEO_ONBOARDING_PROMPT_COUNT``
    (typically 50). Prompts are one flat list for ``selected_aeo_prompts`` persistence.

    When ``onboarding_topic_details`` is set (step-2 onboarding flow), OpenAI batches receive
    those topics + Labs/AEO metadata; ``prompts_by_topic`` maps the same flat list onto
    each selected topic for the review UI (see ``assign_onboarding_prompts_to_selected_topics``).

    Output schema remains backward compatible:
    - fixed: []
    - dynamic: []
    - openai_generated: generated prompts
    - combined: same generated prompts (capped to target)
    - prompts_by_topic: optional dict topic -> prompt strings (onboarding only)
    """
    target = int(target_combined_count or AEO_ONBOARDING_PROMPT_COUNT)

    if business_input is not None:
        ctx = business_input
    else:
        ctx = aeo_business_input_from_profile(
            profile,
            city=city,
            industry=industry,
            services=services,
            niche_modifiers=niche_modifiers,
            differentiators=differentiators,
        )
    _ = include_openai  # kept for API compatibility; onboarding plan is OpenAI-only.
    type_prompts: dict[str, str] = {
        AEOPromptType.TRANSACTIONAL.value: AEO_PROMPT_ENGINE_SYSTEM_PROMPT_TRANSACTIONAL,
        AEOPromptType.TRUST.value: AEO_PROMPT_ENGINE_SYSTEM_PROMPT_TRUST,
        AEOPromptType.COMPARISON.value: AEO_PROMPT_ENGINE_SYSTEM_PROMPT_COMPARISON,
        AEOPromptType.AUTHORITY.value: AEO_PROMPT_ENGINE_SYSTEM_PROMPT_AUTHORITY,
    }
    type_order = [
        AEOPromptType.TRANSACTIONAL.value,
        AEOPromptType.TRUST.value,
        AEOPromptType.COMPARISON.value,
        AEOPromptType.AUTHORITY.value,
    ]

    def _allocate_type_quotas(total: int) -> dict[str, int]:
        quotas = {t: int(total * OPENAI_ONBOARDING_TYPE_RATIO.get(t, 0.0)) for t in type_order}
        assigned = sum(quotas.values())
        i = 0
        while assigned < total:
            t = type_order[i % len(type_order)]
            quotas[t] += 1
            assigned += 1
            i += 1
        return quotas

    def _typed_batch(batch: list[dict[str, Any]], forced_type: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in batch:
            text = str(item.get("prompt", "") or "").strip()
            if not text:
                continue
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            out.append(prompt_record(text, prompt_type=forced_type, weight=weight, dynamic=True))
        return out

    combined: list[dict[str, Any]] = []
    openai_status = "attempted"
    openai_message = ""
    max_batch_size = int(
        getattr(
            settings,
            "AEO_ONBOARDING_OPENAI_MAX_BATCH_SIZE",
            OPENAI_ONBOARDING_MAX_BATCH_SIZE,
        )
    )
    quotas = _allocate_type_quotas(target)

    # Fast single-pass generation per type (no retry/top-up loops).
    for prompt_type in type_order:
        if max_openai_prompts is not None and len(combined) >= max_openai_prompts:
            break
        req = min(max_batch_size, max(0, quotas[prompt_type]))
        if req <= 0:
            continue
        batch = run_prompt_batch_via_openai(
            ctx,
            seed_prompts=combined,
            max_additional=req,
            system_prompt=type_prompts[prompt_type],
            onboarding_topic_details=onboarding_topic_details,
            business_profile=profile,
        )
        typed = _typed_batch(batch, prompt_type)
        if not typed:
            continue
        combined = combine_prompt_set(combined, typed)

    if len(combined) > target:
        combined = combined[:target]

    prompts_by_topic: dict[str, list[str]] = {}
    if onboarding_topic_details:
        topic_order = [
            str(d.get("keyword") or "").strip()
            for d in onboarding_topic_details
            if str(d.get("keyword") or "").strip()
        ]
        if topic_order:
            prompts_by_topic = assign_onboarding_prompts_to_selected_topics(combined, topic_order)

    shortfall = max(0, target - len(combined))
    if len(combined) >= target:
        openai_status = "ok"
    elif len(combined) == 0:
        openai_status = "failed_empty"
        openai_message = "AI prompt generation returned no usable prompts."
    else:
        openai_status = "partial"
        openai_message = f"Only {len(combined)} distinct prompts available; target is {target}."

    out_fixed: list[dict[str, Any]] = []
    out_dynamic: list[dict[str, Any]] = []
    out_openai = combined[:]

    meta = {
        "openai_status": openai_status,
        "openai_message": openai_message.strip(),
        "openai_prompt_count": len(out_openai),
        "combined_count": len(combined),
        "combined_target": target,
        "combined_shortfall": shortfall,
    }
    return {
        "business": ctx.as_dict(),
        "fixed": out_fixed,
        "dynamic": out_dynamic,
        "openai_generated": out_openai,
        "combined": combined,
        "meta": meta,
        "prompts_by_topic": prompts_by_topic,
    }
