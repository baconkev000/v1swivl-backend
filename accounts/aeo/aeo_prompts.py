"""
AEO (Answer Engine Optimization) prompt definitions only.

Edit templates and metadata here; business logic lives in aeo_utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class AEOPromptType(str, Enum):
    """High-level intent bucket for a consumer-style answer-engine query."""

    TRANSACTIONAL = "transactional"
    TRUST = "trust"
    COMPARISON = "comparison"
    AUTHORITY = "authority"


# --- System prompts (OpenAI layers) ------------------------------------------

AEO_PROMPT_ENGINE_SYSTEM_PROMPT: Final[str] = (
    "You are evaluating AI answer-engine visibility prompts for local businesses. "
    "Generate stable high-intent prompts that mimic realistic consumer search behavior "
    "in ChatGPT, Perplexity, Google AI Overviews, and similar systems. "
    "Rules: avoid duplicates and near-duplicates; avoid unnatural or keyword-stuffed phrasing; "
    "preserve local commercial intent; prefer phrasing a real person would type; "
    "keep each prompt under 120 characters; output must be valid JSON only when asked. "
    "Never put the tracked business name, brand, or website domain from context into any prompt—"
    "prompts must read like organic discovery questions, not branded lookups."
)

# Reserved for lightweight prep payloads (Phase 1 helper); not the Phase 3 extractor.
AEO_EXTRACTION_PREP_SYSTEM_PROMPT: Final[str] = (
    "You normalize and validate AEO visibility prompts. "
    "You receive raw model text and return strict JSON suitable for downstream scoring. "
    "Do not invent business facts; preserve prompt wording unless fixing clear typos."
)

# --- Phase 3: structured extraction (second-pass JSON from raw answers) --------

AEO_STRUCTURED_EXTRACTION_SYSTEM_PROMPT: Final[str] = (
    "You are an extraction engine. Read the AI answer carefully and return only structured JSON. "
    "Do not explain. Do not use markdown code fences. Do not add keys beyond those requested. "
    "Do not infer beyond what the text reasonably supports: if the target brand is not clearly "
    "referenced, set brand_mentioned to false. "
    "Competitors must be clearly named businesses (proper names), not generic categories. "
    "Citations must be root domains only when the text shows URLs or domains (e.g. example.com), "
    "never full paths with query strings when you can reduce to the registrable-style host."
)

AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE: Final[str] = (
    "Target business we are tracking:\n{business_name}\n\n"
    "Industry context (may be empty):\n{industry}\n\n"
    "Optional competitor hints (names or domains; may be empty):\n{competitor_hints}\n\n"
    "Original consumer-style question that was answered (may be empty):\n{prompt_text}\n\n"
    "Raw assistant answer to extract from:\n---\n{raw_response}\n---\n\n"
    "Return ONLY one JSON object with exactly these keys:\n"
    '- "brand_mentioned": boolean\n'
    '- "mention_position": one of "top", "middle", "bottom", "none" '
    "(where \"top\" means in the first ~25% of substantive content, \"bottom\" last ~25%, "
    '"none" if brand_mentioned is false)\n'
    '- "mention_count": integer (0 or more) — count of distinct mentions of the target brand\n'
    '- "competitors": array of strings — other specifically named businesses only\n'
    '- "citations": array of strings — root domains only (e.g. healthgrades.com), no schemes or paths\n'
    '- "sentiment": one of "positive", "neutral", "negative" toward the target brand if inferable, else "neutral"\n'
    '- "confidence_score": number between 0 and 1 — your confidence in the extraction\n'
    "No other text before or after the JSON."
)

AEO_STRUCTURED_EXTRACTION_RETRY_SUFFIX: Final[str] = (
    "\n\nYour previous reply was not valid JSON or was missing required keys. "
    "Reply again with ONLY a single valid JSON object and no markdown."
)

# Lowercase tokens; used post-extraction to drop non-business competitor strings.
GENERIC_COMPETITOR_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "local business",
        "local businesses",
        "small business",
        "business",
        "businesses",
        "company",
        "companies",
        "provider",
        "providers",
        "service",
        "services",
        "dentist",
        "dentists",
        "dental",
        "doctor",
        "doctors",
        "physician",
        "physicians",
        "contractor",
        "contractors",
        "plumber",
        "plumbers",
        "best",
        "top",
        "near me",
    }
)

# --- Prompt execution (Phase 2: one consumer query → one model answer) --------

AEO_EXECUTION_SYSTEM_PROMPT: Final[str] = (
    "You are a neutral general-purpose assistant answering realistic consumer questions. "
    "Answer directly. When the question is about local services or providers, respond as a "
    "helpful answer engine would: clear, factual where possible, and may name specific "
    "businesses or options when that matches typical assistant behavior. "
    "Do not prepend disclaimers about being unable to browse the web unless necessary."
)

# --- Phase 5: recommendation explanations (optional OpenAI) -----------------

AEO_RECOMMENDATION_NL_SYSTEM_PROMPT: Final[str] = (
    "You write Answer Engine Optimization (AEO) guidance for any local or regional business. "
    "You receive a JSON gap object that may include gap_kind (visibility_miss, citation_share, "
    "citation_share_generic, no_specific_gap), business_name, city, industry, services, "
    "competitors_in_answer, competitors (comma-separated landscape from the score snapshot), "
    "top_competitor_hint, prompt_text, source_domain, citation_share_at_check, "
    "visibility, citation_share, and snapshot ids for traceability. "
    "Use only facts present in the JSON; never invent brands, URLs, or reviews. "
    "When competitors are listed, refer to them generically as 'named alternatives' or by the names "
    "given—do not add others. "
    "Cover transactional (book/hire), trust (reviews/credibility), comparison (vs alternatives), "
    "and authority (expertise, citations) only when relevant to the gap—do not label these words. "
    "Anchor advice to the city or region when city is provided; otherwise say 'your market'. "
    "Prioritize high-impact fixes: missing brand mentions in answers, or low citation-style share. "
    "Output exactly one or two short sentences, plain language, no markdown, bullets, or JSON."
)

# --- User prompt builder (LLM batch generation) -------------------------------

AEO_BATCH_USER_PROMPT_INTRO: Final[str] = (
    "Using the business context below, produce additional distinct AEO visibility prompts "
    "that real consumers would ask answer engines when choosing a provider in the area. "
    "Each prompt must be generic discovery-style wording: do not include the business name, "
    "brand, or website domain from the context (those fields are for scoring only).\n\n"
)

AEO_BATCH_JSON_SCHEMA_INSTRUCTION: Final[str] = (
    "Return ONLY a JSON array. Each element must be an object with exactly these keys:\n"
    '- "prompt": string (the full natural-language query)\n'
    '- "type": one of "transactional", "trust", "comparison", "authority"\n'
    '- "weight": number (0.5–1.5 importance weight; default 1.0 when unsure)\n'
    '- "dynamic": boolean (true if tightly tied to this business’s specific services or claims)\n'
    "Every \"prompt\" must be organic and unbranded: never name the business, brand, or domain from context.\n"
    "Do not wrap in markdown fences. Do not include commentary."
)


# --- Industry wording (maps free-text industry → provider noun for templates) -

INDUSTRY_PROVIDER_LABELS: Final[dict[str, str]] = {
    "dental": "dentist",
    "default": "local business",
    "home_services": "contractor",
}


def resolve_industry_bucket(industry: str) -> str:
    """
    Map free-text industry to a key in FIXED_INDUSTRY_PROMPT_SPECS / INDUSTRY_PROVIDER_LABELS.
    """
    s = (industry or "").lower()
    if any(k in s for k in ("dental", "dentist", "orthodont", "teeth", "oral")):
        return "dental"
    if any(
        k in s
        for k in (
            "plumb",
            "hvac",
            "roof",
            "electric",
            "landscap",
            "remodel",
            "contractor",
            "handyman",
        )
    ):
        return "home_services"
    return "default"


def provider_label_for_bucket(bucket: str) -> str:
    return INDUSTRY_PROVIDER_LABELS.get(bucket, INDUSTRY_PROVIDER_LABELS["default"])


# --- Template specs (metadata + template string) -----------------------------


@dataclass(frozen=True)
class AEOPromptTemplateSpec:
    """
    A single prompt pattern before variable substitution.

    * template: supports placeholders {city}, {provider_label}, {service}, {modifier},
      {differentiator}, {business_name}, {website_domain} (consumer prompts must stay generic—
      avoid embedding brand or domain in final wording)
    * prompt_type: transactional | trust | comparison | authority
    * weight: used when turning into JSON-ready dicts (LLM may override in batch step)
    * is_fixed: True for industry benchmark patterns; False for business-composed patterns
    * key: stable id for dedupe / future persistence
    """

    key: str
    template: str
    prompt_type: AEOPromptType
    weight: float
    is_fixed: bool


# Fixed (industry benchmark) specs — add rows per vertical over time.
FIXED_INDUSTRY_PROMPT_SPECS: Final[dict[str, tuple[AEOPromptTemplateSpec, ...]]] = {
    "dental": (
        AEOPromptTemplateSpec(
            key="fixed_dental_cosmetic",
            template="Who's a good cosmetic {provider_label} in {city} if I want veneers or whitening?",
            prompt_type=AEOPromptType.TRANSACTIONAL,
            weight=1.0,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_dental_emergency",
            template="I need an emergency {provider_label} in {city} today—who can see me fast?",
            prompt_type=AEOPromptType.TRANSACTIONAL,
            weight=1.0,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_dental_trusted",
            template="Which {provider_label}s near {city} do patients say they actually trust?",
            prompt_type=AEOPromptType.TRUST,
            weight=0.95,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_dental_compare",
            template="How do the top-rated {provider_label}s in {city} compare on reviews and pricing?",
            prompt_type=AEOPromptType.COMPARISON,
            weight=0.9,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_dental_authority",
            template="Who's the best {provider_label} in {city} for dental implants, realistically?",
            prompt_type=AEOPromptType.AUTHORITY,
            weight=0.85,
            is_fixed=True,
        ),
    ),
    "default": (
        AEOPromptTemplateSpec(
            key="fixed_default_best",
            template="Who's the best {provider_label} in {city} for someone who's never used one before?",
            prompt_type=AEOPromptType.TRANSACTIONAL,
            weight=1.0,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_default_top_rated",
            template="What {provider_label} near {city} has the strongest reviews lately?",
            prompt_type=AEOPromptType.TRUST,
            weight=0.95,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_default_recommended",
            template="Can you recommend a {provider_label} in {city} that's easy to book and reliable?",
            prompt_type=AEOPromptType.TRUST,
            weight=0.9,
            is_fixed=True,
        ),
    ),
    "home_services": (
        AEOPromptTemplateSpec(
            key="fixed_hs_emergency",
            template="Who's the best emergency {provider_label} in {city} if something breaks after hours?",
            prompt_type=AEOPromptType.TRANSACTIONAL,
            weight=1.0,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_hs_trusted",
            template="Which {provider_label}s around {city} are known for not upselling you?",
            prompt_type=AEOPromptType.TRUST,
            weight=0.95,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_hs_compare",
            template="How do the main {provider_label}s in {city} compare on price and warranty?",
            prompt_type=AEOPromptType.COMPARISON,
            weight=0.85,
            is_fixed=True,
        ),
    ),
}


# Dynamic pattern specs (combined with business services / modifiers / differentiators).
DYNAMIC_PROMPT_SPECS: Final[tuple[AEOPromptTemplateSpec, ...]] = (
    AEOPromptTemplateSpec(
        key="dyn_modifier_best",
        template="I'm looking for a {modifier} {provider_label} in {city}—who should I call first?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.05,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_service_offers_near",
        template="Where can I get {service} near {city} without waiting weeks for an appointment?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.0,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_service_best",
        template="Who's the strongest {provider_label} in {city} specifically for {service}?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.0,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_diff_trust",
        template="Which {provider_label} in {city} is actually known for {differentiator}?",
        prompt_type=AEOPromptType.TRUST,
        weight=0.95,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_diff_who",
        template="Among {provider_label}s in {city}, who really stands out for {differentiator}?",
        prompt_type=AEOPromptType.AUTHORITY,
        weight=0.9,
        is_fixed=False,
    ),
)

# Extra local-intent patterns when city is known — generic wording only (no brand/domain in prompts).
DYNAMIC_BUSINESS_NAME_SPECS: Final[tuple[AEOPromptTemplateSpec, ...]] = (
    AEOPromptTemplateSpec(
        key="dyn_brand_best",
        template="Which {provider_label} in {city} is usually the safest first call if I'm still comparing options?",
        prompt_type=AEOPromptType.TRUST,
        weight=1.05,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_reviews",
        template="Where should I read honest reviews of {provider_label}s in {city} before I commit?",
        prompt_type=AEOPromptType.TRUST,
        weight=1.0,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_vs",
        template="How do people usually shortlist {provider_label}s in {city} when several look fine online?",
        prompt_type=AEOPromptType.COMPARISON,
        weight=1.0,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_alternatives",
        template="Who are strong backup {provider_label} options in {city} if my top choice is booked?",
        prompt_type=AEOPromptType.COMPARISON,
        weight=0.95,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_worth",
        template="How can I tell if a {provider_label} in {city} is fairly priced for the work?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.0,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_recommend",
        template="What signals actually matter when a friend recommends a {provider_label} in {city}?",
        prompt_type=AEOPromptType.TRUST,
        weight=0.95,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_cost",
        template="What does a typical {provider_label} visit or quote usually run in {city}?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=0.9,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_book",
        template="What's the least painful way to book a {provider_label} in {city} without phone tag?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=0.9,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_site",
        template="What should I verify on a {provider_label}'s site before I schedule in {city}?",
        prompt_type=AEOPromptType.AUTHORITY,
        weight=0.85,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_domain_trust",
        template="How can I tell if a {provider_label}'s website in {city} is legit before I book online?",
        prompt_type=AEOPromptType.TRUST,
        weight=0.9,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_domain_compare",
        template="What red flags show up on local {provider_label} websites around {city}?",
        prompt_type=AEOPromptType.COMPARISON,
        weight=0.85,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_brand_near_me",
        template="What should I ask a {provider_label} near {city} before I hire them?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=0.95,
        is_fixed=False,
    ),
)

# Optional notes for product/tuning (not used in code paths yet).
PROMPT_WEIGHTING_NOTES: Final[str] = (
    "Weights are relative priorities for planning batches, not statistical guarantees. "
    "Transactional and high commercial-intent patterns default to ~1.0; "
    "comparison and authority prompts are slightly down-weighted to balance coverage."
)
