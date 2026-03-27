"""
AEO (Answer Engine Optimization) prompt definitions only.

Edit templates and metadata here; business logic lives in aeo_utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


PROMPT_VERSION: Final[str] = "v3.1"


class AEOPromptType(str, Enum):
    """High-level intent bucket for a consumer-style answer-engine query."""

    TRANSACTIONAL = "transactional"
    TRUST = "trust"
    COMPARISON = "comparison"
    AUTHORITY = "authority"
    SCENARIO = "scenario"


# --- Template specs ----------------------------------------------------------

@dataclass(frozen=True)
class AEOPromptTemplateSpec:
    key: str
    template: str
    prompt_type: AEOPromptType
    weight: float
    is_fixed: bool


# --- System prompts (OpenAI layers) ------------------------------------------

AEO_PROMPT_ENGINE_SYSTEM_PROMPT: Final[str] = (
    "You generate realistic consumer prompts for Answer Engine Optimization testing. "
    "Prompts must sound like genuine human questions asked naturally in conversation or search. "
    "Use business profile context only to infer likely services, category, and buying considerations. "
    "Never include the business name, brand, domain, slogans, or unique marketing language. "
    "Do not mirror exact wording from the business description unless it reflects common consumer vocabulary. "
    "Prompts must remain category-correct, location-aware when relevant, and commercially realistic. "
    "Prefer decision-framed prompts over ranking-framed prompts. "
    "Avoid unnatural list-style phrasing like 'top-rated', 'best options', or 'compare providers' unless it sounds organic. "
    "Keep prompts under 180 characters. "
    "Never create prompts that drift outside the business category or business type."
)

AEO_PROMPT_ENGINE_SYSTEM_PROMPT_TRANSACTIONAL: Final[str] = (
    "You generate transactional consumer discovery prompts for Answer Engine Optimization testing. "
    "Generate natural decision-stage questions that indicate a person is ready to choose where to go. "
    "Use the provided business metadata (industry + location) for context and realism, "
    "but never include the tracked business name or domain. "
    "Prompts must be generic, category-correct, local-intent where relevant, and under 180 characters. "
    "Return ONLY a JSON array with objects containing exactly: prompt, type, weight, dynamic. "
    'Set "type" to "transactional" for every item and "dynamic" to true.'
)

AEO_PROMPT_ENGINE_SYSTEM_PROMPT_TRUST: Final[str] = (
    "You generate trust-focused consumer discovery prompts for Answer Engine Optimization testing. "
    "Generate natural questions about reliability, credibility, safety, or confidence signals. "
    "Use the provided business metadata (industry + location) for context and realism, "
    "but never include the tracked business name or domain. "
    "Prompts must be generic, category-correct, local-intent where relevant, and under 180 characters. "
    "Return ONLY a JSON array with objects containing exactly: prompt, type, weight, dynamic. "
    'Set "type" to "trust" for every item and "dynamic" to true.'
)

AEO_PROMPT_ENGINE_SYSTEM_PROMPT_COMPARISON: Final[str] = (
    "You generate comparison-style consumer discovery prompts for Answer Engine Optimization testing. "
    "Generate natural questions that compare options, tradeoffs, or selection criteria. "
    "Use the provided business metadata (industry + location) for context and realism, "
    "but never include the tracked business name or domain. "
    "Prompts must be generic, category-correct, local-intent where relevant, and under 180 characters. "
    "Return ONLY a JSON array with objects containing exactly: prompt, type, weight, dynamic. "
    'Set "type" to "comparison" for every item and "dynamic" to true.'
)

AEO_PROMPT_ENGINE_SYSTEM_PROMPT_AUTHORITY: Final[str] = (
    "You generate authority-style consumer discovery prompts for Answer Engine Optimization testing. "
    "Generate natural questions about expertise, qualifications, standards, or evidence of competence. "
    "Use the provided business metadata (industry + location) for context and realism, "
    "but never include the tracked business name or domain. "
    "Prompts must be generic, category-correct, local-intent where relevant, and under 180 characters. "
    "Return ONLY a JSON array with objects containing exactly: prompt, type, weight, dynamic. "
    'Set "type" to "authority" for every item and "dynamic" to true.'
)

AEO_EXTRACTION_PREP_SYSTEM_PROMPT: Final[str] = (
    "You normalize and validate AEO visibility prompts. "
    "Receive raw model text and return strict JSON suitable for scoring. "
    "Do not invent business facts; preserve prompt wording unless fixing clear typos."
)

AEO_STRUCTURED_EXTRACTION_SYSTEM_PROMPT: Final[str] = (
    "You are an extraction engine. Read the answer carefully and return only structured JSON. "
    "Do not explain. Do not use markdown code fences. Do not add keys beyond those requested. "
    "Do not infer beyond what the text reasonably supports."
    "Treat close brand variants as the same business when clearly referring to the same entity. "
    "If the target brand is not clearly referenced, set brand_mentioned to false. "
    "Competitors must be clearly named businesses only. "
    "Exclude generic categories, directories, specialties, and descriptive phrases. "
    "ranking_order must contain only explicitly named businesses in exact first-mention order. "
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
    '- "mention_strength": one of "primary", "secondary", "incidental", "none"\n'
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


# --- Execution prompt --------------------------------------------------------

AEO_EXECUTION_SYSTEM_PROMPT: Final[str] = (
    "Answer only with a list of the company names"
    "Do not invent fake businesses."
    "Avoid filler, disclaimers, hedging, and generic advice."
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
    "Using the business context below, including industry, description, and services, "
    "produce additional distinct AEO visibility prompts that real consumers would naturally ask."
    "Prompts must sound human, commercially realistic, and category-correct."
    "Never include the business name, brand, domain, slogans, or copied marketing language.\n\n"
)

AEO_BATCH_JSON_SCHEMA_INSTRUCTION: Final[str] = (
    "Return ONLY a JSON array. Each element must contain:\n"
    '- "prompt": string\n'
    '- "type": one of "transactional", "trust", "comparison", "authority", "scenario"\n'
    '- "weight": number\n'
    '- "dynamic": boolean\n'
)


# --- Dynamic prompts ---------------------------------------------------------

DYNAMIC_PROMPT_SPECS: Final[tuple[AEOPromptTemplateSpec, ...]] = (
    AEOPromptTemplateSpec(
        key="dyn_local_recommendation",
        template="If someone needed help choosing in {city}, what would they usually ask first?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.15,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_scenario_new_customer",
        template="What would someone new to {city} ask before choosing this kind of business?",
        prompt_type=AEOPromptType.SCENARIO,
        weight=1.1,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_quality_signal",
        template="How do people usually ask about trust or reliability in {city} for this kind of business?",
        prompt_type=AEOPromptType.TRUST,
        weight=1.05,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_differentiation",
        template="What kind of question usually leads people to compare businesses like this in {city}?",
        prompt_type=AEOPromptType.COMPARISON,
        weight=1.0,
        is_fixed=False,
    ),
)


# --- Generic competitor tokens ----------------------------------------------

GENERIC_COMPETITOR_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "local business",
        "small business",
        "business",
        "company",
        "provider",
        "service",
        "best",
        "top",
        "near me",
    }
)


# --- Prompt weighting notes --------------------------------------------------

PROMPT_WEIGHTING_NOTES: Final[str] = (
    "Transactional prompts remain highest priority. "
    "Trust prompts are next because they often trigger named recommendations. "
    "Scenario prompts improve natural recommendation realism. "
    "Comparison prompts support competitor visibility analysis. "
    "Authority prompts remain lowest weight."
)


# --- Fixed benchmark prompts -------------------------------------------------

FIXED_INDUSTRY_PROMPT_SPECS: Final[dict[str, tuple[AEOPromptTemplateSpec, ...]]] = {}


# --- Dynamic business-name-style prompts ------------------------------------

DYNAMIC_BUSINESS_NAME_SPECS: Final[tuple[AEOPromptTemplateSpec, ...]] = (
    AEOPromptTemplateSpec(
        key="dyn_anchor_standout",
        template="What usually makes one option stand out more than another in {city}?",
        prompt_type=AEOPromptType.COMPARISON,
        weight=1.05,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_anchor_trust",
        template="How do people usually decide who to trust first in {city}?",
        prompt_type=AEOPromptType.TRUST,
        weight=1.05,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_anchor_selection",
        template="What matters most when choosing where to go in {city} for this kind of need?",
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.0,
        is_fixed=False,
    ),
    AEOPromptTemplateSpec(
        key="dyn_anchor_new_to_city",
        template="If someone were new to {city}, what would they ask before choosing?",
        prompt_type=AEOPromptType.SCENARIO,
        weight=1.0,
        is_fixed=False,
    ),
)