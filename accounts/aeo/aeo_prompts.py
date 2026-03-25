"""
AEO (Answer Engine Optimization) prompt definitions only.

Edit templates and metadata here; business logic lives in aeo_utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


PROMPT_VERSION: Final[str] = "v1.3"


class AEOPromptType(str, Enum):
    """High-level intent bucket for a consumer-style answer-engine query."""

    TRANSACTIONAL = "transactional"
    TRUST = "trust"
    COMPARISON = "comparison"
    AUTHORITY = "authority"


# --- System prompts (OpenAI layers) ------------------------------------------

AEO_PROMPT_ENGINE_SYSTEM_PROMPT: Final[str] = (
    "You are evaluating AI answer-engine visibility prompts for businesses across multiple models. "
    "Generate stable high-intent prompts that mimic realistic consumer search behavior "
    "in ChatGPT, Perplexity, Google AI Overviews, and similar systems. "
    "Rules: avoid duplicates and near-duplicates; avoid unnatural or keyword-stuffed phrasing; "
    "preserve commercial intent; keep location context when relevant; "
    "avoid assumptions about booking, calling, visits, or appointments unless context requires it; "
    "prefer phrasing a real person would type; "
    "keep each prompt under 120 characters; output must be valid JSON only when asked. "
    "Never put the tracked business name, brand, or website domain from context into any prompt—"
    "prompts must read like organic discovery questions, not branded lookups."
)

AEO_EXTRACTION_PREP_SYSTEM_PROMPT: Final[str] = (
    "You normalize and validate AEO visibility prompts. "
    "You receive raw model text and return strict JSON suitable for downstream scoring. "
    "Do not invent business facts; preserve prompt wording unless fixing clear typos."
)

# --- Phase 3: structured extraction ------------------------------------------

AEO_STRUCTURED_EXTRACTION_SYSTEM_PROMPT: Final[str] = (
    "You are an extraction engine. Read the AI answer carefully and return only structured JSON. "
    "Do not explain. Do not use markdown code fences. Do not add keys beyond those requested. "
    "Do not infer beyond what the text reasonably supports. "
    "If the target brand is not clearly referenced, set brand_mentioned to false. "
    "Competitors must be clearly named businesses only. "
    "ranking_order must contain named businesses in first-mention order. "
    "Include the tracked business if it appears. "
    "Citations must be root domains only when URLs or domains appear."
)

AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE: Final[str] = (
    "Target business we are tracking:\n{business_name}\n\n"
    "Industry context:\n{industry}\n\n"
    "Optional competitor hints:\n{competitor_hints}\n\n"
    "Original prompt:\n{prompt_text}\n\n"
    "Raw assistant answer:\n---\n{raw_response}\n---\n\n"
    "Return ONLY one JSON object with exactly these keys:\n"
    '- "brand_mentioned": boolean\n'
    '- "mention_position": one of "top", "middle", "bottom", "none"\n'
    '- "mention_count": integer\n'
    '- "competitors": array of strings\n'
    '- "ranking_order": array of strings\n'
    '- "citations": array of strings\n'
    '- "sentiment": one of "positive", "neutral", "negative"\n'
    '- "confidence_score": number between 0 and 1\n'
    "No other text."
)

AEO_STRUCTURED_EXTRACTION_RETRY_SUFFIX: Final[str] = (
    "\n\nYour previous reply was invalid JSON. Reply again with one valid JSON object only."
)

# --- Generic competitor cleanup ----------------------------------------------

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
        "doctor",
        "doctors",
        "contractor",
        "contractors",
        "best",
        "top",
        "near me",
    }
)

# --- Execution prompt --------------------------------------------------------

AEO_EXECUTION_SYSTEM_PROMPT: Final[str] = (
    "You are a neutral answer engine simulating realistic consumer-facing results. "
    "Answer directly in 60-110 words. "
    "Prefer one concise paragraph or two short paragraphs. "
    "Use natural language, not JSON, markdown tables, or heavy bullets. "
    "Mention only the most relevant options for the query. "
    "Naming competitors is allowed when naturally relevant. "
    "Keep ordering natural by relevance. "
    "Include citations or domains only when naturally appropriate. "
    "Avoid unnecessary disclaimers, filler, hedging, or long caveats."
)

# --- Recommendations ---------------------------------------------------------

AEO_RECOMMENDATION_NL_SYSTEM_PROMPT: Final[str] = (
    "You write Answer Engine Optimization guidance for businesses. "
    "Use only facts present in input JSON. "
    "Never invent brands, URLs, reviews, or citations. "
    "Output exactly one or two short sentences, plain language only."
)

# --- Batch prompt generation -------------------------------------------------

AEO_BATCH_USER_PROMPT_INTRO: Final[str] = (
    "Using the business context below, produce additional distinct AEO visibility prompts "
    "that real consumers would ask answer engines when choosing a provider. "
    "Each prompt must stay generic and never include the business name, brand, or domain.\n\n"
)

AEO_BATCH_JSON_SCHEMA_INSTRUCTION: Final[str] = (
    "Return ONLY a JSON array. Each element must contain:\n"
    '- "prompt": string\n'
    '- "type": one of "transactional", "trust", "comparison", "authority"\n'
    '- "weight": number\n'
    '- "dynamic": boolean\n'
)

# --- Industry provider labels ------------------------------------------------

INDUSTRY_PROVIDER_LABELS: Final[dict[str, str]] = {
    "local_service": "local business",
    "professional_service": "provider",
    "ecommerce": "store",
    "saas": "software",
    "healthcare": "provider",
    "home_services": "contractor",
    "default": "business",
}


def resolve_industry_bucket(industry: str) -> str:
    s = (industry or "").lower()

    if any(k in s for k in ("saas", "software", "platform", "tool")):
        return "saas"

    if any(k in s for k in ("ecommerce", "retail", "shop", "store")):
        return "ecommerce"

    if any(k in s for k in ("health", "dental", "medical", "clinic")):
        return "healthcare"

    if any(k in s for k in ("plumb", "hvac", "roof", "electric", "contractor")):
        return "home_services"

    if any(k in s for k in ("law", "legal", "account", "consult", "agency")):
        return "professional_service"

    return "default"


def provider_label_for_bucket(bucket: str) -> str:
    return INDUSTRY_PROVIDER_LABELS.get(bucket, INDUSTRY_PROVIDER_LABELS["default"])


# --- Template specs ----------------------------------------------------------

@dataclass(frozen=True)
class AEOPromptTemplateSpec:
    key: str
    template: str
    prompt_type: AEOPromptType
    weight: float
    is_fixed: bool


# --- Fixed prompts -----------------------------------------------------------

FIXED_INDUSTRY_PROMPT_SPECS: Final[dict[str, tuple[AEOPromptTemplateSpec, ...]]] = {
    "default": (
        AEOPromptTemplateSpec(
            key="fixed_default_best",
            template="What {provider_label} in {city} is strongest for first-time buyers?",
            prompt_type=AEOPromptType.TRANSACTIONAL,
            weight=1.0,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_default_trust",
            template="Which {provider_label} near {city} has the most consistent reviews?",
            prompt_type=AEOPromptType.TRUST,
            weight=0.95,
            is_fixed=True,
        ),
        AEOPromptTemplateSpec(
            key="fixed_default_compare",
            template="How do people compare {provider_label} options in {city} before deciding?",
            prompt_type=AEOPromptType.COMPARISON,
            weight=0.9,
            is_fixed=True,
        ),
    ),
}

# --- Dynamic prompts ---------------------------------------------------------

DYNAMIC_PROMPT_SPECS: Final[tuple[AEOPromptTemplateSpec, ...]] = (
    AEOPromptTemplateSpec(
        key="dyn_modifier_best",
        template="What {modifier} {provider_label} in {city} is most reliable right now?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.05,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_service_offers_near",
        template="Where can I get {service} from a trusted provider in {city}?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.0,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_service_best",
        template="Which {provider_label} in {city} is strongest specifically for {service}?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.0,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_diff_trust",
        template="Which {provider_label} in {city} is known for {differentiator}?",
        prompt_type=AEOPromptType.TRUST,
        weight=0.95,
        is_fixed=False,
    ),
)

PROMPT_WEIGHTING_NOTES: Final[str] = (
    "Transactional prompts default highest priority. "
    "Comparison and authority prompts slightly lower for balanced scoring."
)

