"""
AEO prompt generation (Phase 1).

Execution and persistence live in aeo_execution_utils.py.
Uses template/metadata from aeo_prompts.py; batch *generation* uses openai_utils.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Mapping, Sequence
from urllib.parse import urlparse

from django.conf import settings

from ..models import BusinessProfile

# Onboarding / tracking: exactly this many prompts in combined output and saved profile lists.
AEO_ONBOARDING_PROMPT_COUNT: Final[int] = 50
from ..openai_utils import _get_client, _get_model
from .aeo_prompts import (
    AEOPromptType,
    AEOPromptTemplateSpec,
    AEO_BATCH_JSON_SCHEMA_INSTRUCTION,
    AEO_BATCH_USER_PROMPT_INTRO,
    AEO_EXTRACTION_PREP_SYSTEM_PROMPT,
    AEO_PROMPT_ENGINE_SYSTEM_PROMPT,
    DYNAMIC_BUSINESS_NAME_SPECS,
    DYNAMIC_PROMPT_SPECS,
    FIXED_INDUSTRY_PROMPT_SPECS,
    provider_label_for_bucket,
    resolve_industry_bucket,
)


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    provider_label: str,
    service: str = "",
    modifier: str = "",
    differentiator: str = "",
    business_name: str = "",
    website_domain: str = "",
) -> str:
    return spec.template.format(
        city=city,
        provider_label=provider_label,
        service=service,
        modifier=modifier,
        differentiator=differentiator,
        business_name=business_name,
        website_domain=website_domain,
    )


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
    """Partition ``combined`` (in order) into fixed / dynamic / openai buckets by prompt text."""
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
    Deterministic industry benchmark prompts from FIXED_INDUSTRY_PROMPT_SPECS.
    """
    bucket = resolve_industry_bucket(industry)
    specs = FIXED_INDUSTRY_PROMPT_SPECS.get(bucket) or FIXED_INDUSTRY_PROMPT_SPECS["default"]
    provider = provider_label_for_bucket(bucket)
    c = _normalize_city(city)
    out: list[dict[str, Any]] = []
    for spec in specs:
        text = _format_template(
            spec,
            city=c,
            provider_label=provider,
            business_name="",
            website_domain="",
        )
        if not text:
            continue
        out.append(
            prompt_record(
                text,
                prompt_type=spec.prompt_type,
                weight=spec.weight,
                dynamic=False,
            )
        )
    return out


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

    bucket = resolve_industry_bucket(ctx.industry)
    provider = provider_label_for_bucket(bucket)
    c = _normalize_city(ctx.city)
    if not c:
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    services = [_clean_token(x) for x in ctx.services if _clean_token(x)]
    modifiers = [_clean_token(x) for x in ctx.niche_modifiers if _clean_token(x)]
    diffs = [_clean_token(x) for x in ctx.differentiators if _clean_token(x)]

    for spec in DYNAMIC_PROMPT_SPECS:
        if spec.key.startswith("dyn_modifier") and modifiers:
            for mod in modifiers:
                text = _format_template(
                    spec, city=c, provider_label=provider, modifier=mod
                )
                key = text.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(
                        prompt_record(
                            text,
                            prompt_type=spec.prompt_type,
                            weight=spec.weight,
                            dynamic=True,
                        )
                    )
        elif spec.key.startswith("dyn_service") and services:
            for svc in services:
                text = _format_template(
                    spec, city=c, provider_label=provider, service=svc
                )
                key = text.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(
                        prompt_record(
                            text,
                            prompt_type=spec.prompt_type,
                            weight=spec.weight,
                            dynamic=True,
                        )
                    )
        elif spec.key.startswith("dyn_diff") and diffs:
            for d in diffs:
                text = _format_template(
                    spec, city=c, provider_label=provider, differentiator=d
                )
                key = text.lower()
                if key not in seen:
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
            provider_label=provider,
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
) -> str:
    lines = [
        AEO_BATCH_USER_PROMPT_INTRO,
        "Business context (JSON):\n",
        json.dumps(business.as_dict(), ensure_ascii=False),
        f"\n\nGenerate at most {max_additional} new prompts not overlapping the seed list.",
        "\n\nSeed prompts (JSON array, may be empty):\n",
        json.dumps(list(seed_prompts or []), ensure_ascii=False),
        "\n\n",
        AEO_BATCH_JSON_SCHEMA_INSTRUCTION,
    ]
    return "".join(lines)


def run_prompt_batch_via_openai(
    business: AEOPromptBusinessInput,
    seed_prompts: Sequence[Mapping[str, Any]] | None = None,
    *,
    max_additional: int = 12,
    system_prompt: str | None = None,
    api_key_env: str = "OPEN_AI_SEO_API_KEY",
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Ask OpenAI for additional prompts; returns normalized JSON-ready dicts.

    Uses the same API key resolution as SEO/AEO helpers in openai_utils.
    """
    user_content = build_openai_batch_user_content(
        business, seed_prompts, max_additional=max_additional
    )
    sys_p = system_prompt or AEO_PROMPT_ENGINE_SYSTEM_PROMPT
    use_model = model or _get_model()

    try:
        client = _get_client(api_key_env)
        completion = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        return []
    return parse_aeo_prompt_json_array(raw)


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
        "openai_model": getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
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
) -> dict[str, Any]:
    """
    Convenience bundler for Django services: fixed + dynamic [+ optional OpenAI expansion].

    Onboarding returns exactly ``AEO_ONBOARDING_PROMPT_COUNT`` prompts when enough distinct
    templates + LLM rows exist; otherwise ``meta.combined_shortfall`` reports the gap.
    OpenAI batches repeat (up to a few rounds) until the target is reached or the API stops
    returning new rows (48 prompts max per call).
    """
    target = AEO_ONBOARDING_PROMPT_COUNT

    ctx = aeo_business_input_from_profile(
        profile,
        city=city,
        industry=industry,
        services=services,
        niche_modifiers=niche_modifiers,
        differentiators=differentiators,
    )
    fixed = generate_fixed_prompts(ctx.industry, ctx.city)
    dynamic = generate_dynamic_prompts(ctx)
    combined = combine_prompt_set(fixed, dynamic)
    llm_extra: list[dict[str, Any]] = []
    openai_status = "disabled"
    openai_message = ""

    max_rounds = 5
    round_i = 0
    if include_openai and ctx.city:
        openai_status = "attempted"
        while len(combined) < target and round_i < max_rounds:
            if max_openai_prompts is not None and len(llm_extra) >= max_openai_prompts:
                break
            need = min(48, target - len(combined))
            if need <= 0:
                break
            batch = run_prompt_batch_via_openai(
                ctx,
                seed_prompts=combined,
                max_additional=need,
            )
            round_i += 1
            if not batch:
                break
            llm_extra.extend(batch)
            combined = combine_prompt_set(combined, batch)
        if not llm_extra and len(combined) < target:
            openai_status = "failed_empty"
            openai_message = (
                "Additional AI-generated prompts could not be loaded (API error or empty response). "
                "Showing template-based prompts only."
            )
        elif len(combined) >= target:
            openai_status = "ok" if llm_extra else "disabled"
        elif llm_extra:
            openai_status = "ok"
    elif include_openai and not ctx.city:
        openai_status = "skipped_no_city"
        openai_message = (
            "A city or region could not be inferred from your business address; AI expansion was skipped. "
            "Add an address that includes at least city and state or region (street is optional) for fuller coverage."
        )

    if len(combined) > target:
        combined = combined[:target]

    shortfall = max(0, target - len(combined))
    if shortfall and openai_message:
        openai_message = f"{openai_message} ({shortfall} short of {target} distinct prompts.)"
    elif shortfall:
        openai_message = f"Only {len(combined)} distinct prompts available; target is {target}."

    out_fixed, out_dynamic, out_openai = _split_combined_into_source_groups(combined, fixed, dynamic)

    meta = {
        "openai_status": openai_status,
        "openai_message": openai_message.strip(),
        "openai_prompt_count": len(llm_extra),
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
    }
