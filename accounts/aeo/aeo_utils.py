"""
AEO prompt generation (Phase 1).

Execution and persistence live in aeo_execution_utils.py.
Uses template/metadata from aeo_prompts.py; batch *generation* uses openai_utils.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from string import Formatter
from typing import Any, Final, Mapping, Sequence
from urllib.parse import urlparse

from django.conf import settings

from ..models import BusinessProfile
from .aeo_plan_targets import AEO_PLAN_CAP_PRO, AEO_PLAN_CAP_STARTER

logger = logging.getLogger(__name__)

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


def aeo_openai_max_output_tokens_for_target(target_combined_count: int) -> int:
    """
    Per-completion max_tokens for AEO prompt-batch OpenAI calls, scaled to the combined target size.

    High max_tokens reduces truncated JSON when the model returns a large array of prompts
    (failed_empty / partial expansion). TPM and RPM are account-level; this value only caps
    each individual completion output.
    """
    t = max(1, int(target_combined_count))
    starter = int(getattr(settings, "AEO_OPENAI_MAX_TOKENS_STARTER", 1024))
    pro = int(getattr(settings, "AEO_OPENAI_MAX_TOKENS_PRO", 4096))
    advanced = int(getattr(settings, "AEO_OPENAI_MAX_TOKENS_ADVANCED", 8192))
    if t <= AEO_PLAN_CAP_STARTER:
        return starter
    if t <= AEO_PLAN_CAP_PRO:
        return pro
    return advanced


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
    customer_reach: str = ""
    customer_reach_state: str = ""
    customer_reach_city: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def aeo_business_input_from_onboarding_payload(
    *,
    business_name: str,
    website_url: str,
    location: str,
    language: str = "",
    selected_topics: Sequence[str],
    customer_reach: str = "online",
    customer_reach_state: str = "",
    customer_reach_city: str = "",
) -> AEOPromptBusinessInput:
    """
    Build prompt-plan context from onboarding step 1+2 POST body (not from profile fields).

    ``industry`` is a short hint derived from selected topic labels so OpenAI has vertical
    context without relying on BusinessProfile.industry mid-onboarding.

    When ``customer_reach`` is ``local``, the ``city`` field pairs city + state (from explicit
    fields and/or inferred address) so generated prompts stay geographically consistent.
    """
    topics = [str(t).strip() for t in selected_topics if str(t).strip()]
    loc = (location or "").strip()
    reach = str(customer_reach or "online").strip().lower()
    state = _normalize_city(str(customer_reach_state or ""))
    city = _normalize_city(str(customer_reach_city or ""))
    inferred = infer_city_from_address(loc) or _normalize_city(loc[:200])
    if reach == "local":
        city_guess = _compose_locality_for_local_business(
            inferred_city=inferred,
            explicit_city=city,
            explicit_state=state,
        )
    else:
        city_guess = inferred
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
        customer_reach=reach,
        customer_reach_state=state,
        customer_reach_city=city,
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

    If ``len(topic_order) > len(texts)``, only the first ``len(texts)`` topics receive at least one
    prompt; later topics stay empty. The onboarding frontend therefore enforces
    ``selected_topics <= aeo_onboarding_prompt_target_count`` so the UI never depends on empty pools.
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


# Lowercase US state / DC full name → postal abbrev (for deduping "City, TX" vs state "Texas").
_US_STATE_FULL_LOWER_TO_ABB: Final[dict[str, str]] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}


def _state_in_locality(locality: str, state: str) -> bool:
    """
    True if ``state`` (full name or 2-letter US code) is already represented in ``locality``.

    Avoids appending a second state fragment when the city string already ends with ``City, ST``
    or contains the full state name.
    """
    loc_n = _normalize_city(locality)
    st = _normalize_city(state)
    if not loc_n or not st:
        return False
    loc_l = loc_n.lower()
    st_l = st.lower()
    if len(st) > 2 and st_l in loc_l:
        return True
    parts = [p.strip() for p in loc_n.split(",") if p.strip()]
    if not parts:
        return False
    tail = parts[-1]
    tail_l = tail.lower()
    tail_is_abbrev = len(tail) == 2 and tail.isalpha()
    tail_u = tail.upper() if tail_is_abbrev else ""

    if len(st) == 2 and st.isalpha():
        st_u = st.upper()
        if tail_is_abbrev and tail_u == st_u:
            return True
        if _US_STATE_FULL_LOWER_TO_ABB.get(tail_l) == st_u:
            return True
        if re.search(rf"(^|,\s*|\s){re.escape(st_u)}(\s*,|\s*$)", loc_n, re.IGNORECASE):
            return True
    if len(st) > 2:
        abb = _US_STATE_FULL_LOWER_TO_ABB.get(st_l)
        if abb and tail_is_abbrev and tail_u == abb:
            return True
        if tail_l == st_l:
            return True
    return False


def _compose_locality_for_local_business(
    *,
    inferred_city: str,
    explicit_city: str,
    explicit_state: str,
) -> str:
    """
    Build the single ``city`` string for templates + OpenAI batch context when reach is local.

    Prefer explicit onboarding/profile city, then inferred address fragment. When a state is
    known and not already reflected next to the city, append ``", {state}"`` so prompts stay
    consistent (city-only localities are ambiguous across regions).
    """
    inferred = _normalize_city(inferred_city)
    city_ex = _normalize_city(explicit_city)
    state = _normalize_city(explicit_state)
    city_base = city_ex or inferred
    if not city_base and state:
        return state
    if not state:
        return city_base
    if _state_in_locality(city_base, state):
        return city_base
    return f"{city_base}, {state}" if city_base else state


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
    inferred = infer_city_from_address(addr)
    reach = str(getattr(profile, "customer_reach", None) or "online").strip().lower()
    state_f = _normalize_city(str(getattr(profile, "customer_reach_state", "") or ""))
    profile_city = _normalize_city(str(getattr(profile, "customer_reach_city", "") or ""))
    explicit_override = _normalize_city(city) if city else ""

    if reach == "local":
        explicit_city = explicit_override or profile_city
        resolved_city = _compose_locality_for_local_business(
            inferred_city=inferred,
            explicit_city=explicit_city,
            explicit_state=state_f,
        )
    else:
        resolved_city = explicit_override or inferred

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
        customer_reach=reach,
        customer_reach_state=state_f,
        customer_reach_city=explicit_override or profile_city,
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


def phase2_prompt_plan_items_for_execution_run(run: Any) -> list[dict[str, Any]]:
    """
    Build Phase 2 ``prompt_set`` from aggregates tied to ``run`` (same hashes as Phase 1).

    Falls back to an empty list when there are no rows — callers may pass ``None`` to
    ``run_aeo_phase2_confidence_task`` to use profile-wide prompts, but expansion/backfill
    runs should always have aggregates here.
    """
    from ..models import AEOPromptExecutionAggregate
    from .aeo_plan_targets import aeo_effective_monitored_target_for_profile

    profile = run.profile
    aggs = AEOPromptExecutionAggregate.objects.filter(execution_run=run).order_by("id")
    texts: list[str] = []
    seen_hash: set[str] = set()
    for a in aggs:
        t = (a.prompt_text or "").strip()
        if not t:
            continue
        h = str(a.prompt_hash or "").strip()
        if h and h in seen_hash:
            continue
        if h:
            seen_hash.add(h)
        texts.append(t)
    if not texts:
        return []
    cap = max(len(texts), aeo_effective_monitored_target_for_profile(profile))
    return plan_items_from_saved_prompt_strings(texts, max_items=cap)


def plan_items_from_saved_prompt_strings(
    texts: Sequence[Any] | None,
    *,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """
    Normalize stored prompt rows (strings and/or dicts from ``selected_aeo_prompts``).

    When ``max_items`` is None, caps at ``AEO_ONBOARDING_PROMPT_COUNT`` (onboarding / legacy).
    Pass a larger ``max_items`` for Pro/Advanced execution payloads (e.g. expansion backfill).
    """
    from .prompt_storage import plan_items_from_profile_selected

    return plan_items_from_profile_selected(list(texts or []), max_items=max_items)


def prompt_record(
    text: str,
    *,
    prompt_type: str | AEOPromptType,
    weight: float,
    dynamic: bool,
    is_custom: bool = False,
) -> dict[str, Any]:
    ptype = prompt_type.value if isinstance(prompt_type, AEOPromptType) else str(prompt_type)
    rec: dict[str, Any] = {
        "prompt": re.sub(r"\s+", " ", (text or "").strip()),
        "type": ptype,
        "weight": float(weight),
        "dynamic": bool(dynamic),
    }
    if is_custom:
        rec["is_custom"] = True
    return rec


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
    return prompt_record(
        text,
        prompt_type=ptype,
        weight=w,
        dynamic=bool(dyn),
        is_custom=bool(item.get("is_custom")),
    )


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
    if str(biz_for_llm.get("customer_reach") or "").strip().lower() == "local":
        lines.append(
            "\n\nLocal geography rule: customer_reach is \"local\". Whenever a generated prompt "
            "names a place (city, metro, neighborhood, or service area), include BOTH city and state "
            "in the wording exactly as reflected by the combined `city` string in business context "
            "(that field already pairs city with state when both are known). Do not use city-only "
            "place names when the state is present in context — keep geography consistent for the model.\n",
        )
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
    max_output_tokens: int | None = None,
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
    eff_max_tokens = (
        int(max_output_tokens)
        if max_output_tokens is not None
        else aeo_openai_max_output_tokens_for_target(AEO_PLAN_CAP_STARTER)
    )

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
            max_tokens=eff_max_tokens,
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

    After the four-type pass, optional top-up rounds (``AEO_PROMPT_TOPUP_MAX_ROUNDS``) request
    more prompts until ``target`` is reached or progress stops. Top-up uses the same
    ``type_order`` system prompts in rotation (transactional → trust → comparison → authority).

    Output schema remains backward compatible:
    - fixed: []
    - dynamic: []
    - openai_generated: generated prompts
    - combined: same generated prompts (capped to target)
    - prompts_by_topic: optional dict topic -> prompt strings (onboarding only)
    """
    target = int(target_combined_count or AEO_ONBOARDING_PROMPT_COUNT)
    batch_max_output_tokens = aeo_openai_max_output_tokens_for_target(target)

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

    # Initial single pass: one batch per prompt type (quotas sum to target).
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
            max_output_tokens=batch_max_output_tokens,
        )
        typed = _typed_batch(batch, prompt_type)
        if not typed:
            continue
        combined = combine_prompt_set(combined, typed)

    if len(combined) > target:
        combined = combined[:target]

    topup_max_rounds = max(0, int(getattr(settings, "AEO_PROMPT_TOPUP_MAX_ROUNDS", 3)))
    topup_buffer = max(0, int(getattr(settings, "AEO_PROMPT_TOPUP_BUFFER", 8)))
    for topup_round in range(1, topup_max_rounds + 1):
        if len(combined) >= target:
            break
        if max_openai_prompts is not None and len(combined) >= max_openai_prompts:
            break
        need = target - len(combined)
        max_additional = min(max_batch_size, need + topup_buffer)
        if max_additional <= 0:
            break
        # Rotate system prompt across types so top-up variety matches the main pass.
        type_idx = (topup_round - 1) % len(type_order)
        topup_type = type_order[type_idx]
        len_before = len(combined)
        batch = run_prompt_batch_via_openai(
            ctx,
            seed_prompts=combined,
            max_additional=max_additional,
            system_prompt=type_prompts[topup_type],
            onboarding_topic_details=onboarding_topic_details,
            business_profile=profile,
            max_output_tokens=batch_max_output_tokens,
        )
        typed = _typed_batch(batch, topup_type)
        if not typed:
            logger.info(
                "[AEO prompt top-up] empty_batch target=%s have=%s need=%s round=%s/%s type=%s",
                target,
                len_before,
                need,
                topup_round,
                topup_max_rounds,
                topup_type,
            )
            break
        combined = combine_prompt_set(combined, typed)
        if len(combined) > target:
            combined = combined[:target]
        added = len(combined) - len_before
        logger.info(
            "[AEO prompt top-up] target=%s have=%s need=%s round=%s/%s added=%s type=%s",
            target,
            len(combined),
            need,
            topup_round,
            topup_max_rounds,
            added,
            topup_type,
        )
        if added == 0:
            break

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
