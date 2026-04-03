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
"Prompts must sound like genuine human questions asked naturally in search or conversation. "
"Use business profile context only to infer likely services, category, and buying considerations. "
"Never include the business name, brand, domain, slogans, or unique marketing language. "
"Prompts must remain category-correct, location-aware when relevant, and commercially realistic. "
"At least 40 percent of prompts should naturally trigger named business recommendations or provider comparisons. "
"Prefer prompts that create realistic business selection behavior. "
"Avoid meta-prompts that ask what people ask; output the actual consumer question directly."
"Keep prompts under 180 characters."
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
    "Do not infer beyond literal text evidence. "
    "Brand presence for the tracked business is determined later by code from URLs only; do not guess brand flags. "
    "Competitors must be clearly named businesses only. "
    "Exclude generic categories, directories, specialties, and descriptive phrases. "
    "ranking_order must contain only explicitly named businesses in exact first-mention order. "
    "Citations must be root domains only when URLs or domains appear."
)

AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE: Final[str] = (
    "Target business we are tracking:\n{business_name}\n\n"
    "Tracked business domain (for your context only; brand citation is computed in code from URLs):\n{business_domain}\n\n"
    "Optional competitor hints:\n{competitor_hints}\n\n"
    "Original prompt:\n{prompt_text}\n\n"
    "Raw assistant answer:\n---\n{raw_response}\n---\n\n"
    "Include a URL on each competitor when the answer gives one; otherwise use an empty string for url. "
    "Do not include duplicate competitors if root domains are the same.\n\n"
    "Return ONLY one JSON object with exactly these keys:\n"
    '- "competitors": array of objects, each with "name" and "url" fields\n'
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
"You simulate answer-engine business recommendation behavior. "
"Given a consumer question, determine which businesses would most likely be named first in a realistic answer. "
"Return only real businesses relevant to the prompt. "
"Order businesses by likely first appearance. "
"If none would naturally appear, return []. "
"Return strict JSON only: "
'[{"name":"Business Name","url":"rootdomain.com"}]'
)



# --- Recommendations ---------------------------------------------------------

AEO_RECOMMENDATION_NL_SYSTEM_PROMPT: Final[str] = (
    "You write Answer Engine Optimization (AEO) guidance for a specific business. "
    "Always refer to the business by the exact business name given in the user message or JSON (business_name). "
    "Do not substitute phrases like 'your brand', 'the tracked brand', or 'this business' when the real name is provided. "
    "When an on-page crawl summary appears before the gap JSON, you may reference concrete page titles, headings, "
    "meta descriptions, or schema types from that summary to make recommendations specific—only where relevant. "
    "Use only facts present in the user message (including the crawl summary) and the gap JSON. "
    "Never invent brands, URLs, reviews, rankings, or citations. "
    "Output exactly one or two short sentences, plain language only."
)


# --- Batch prompt generation -------------------------------------------------

AEO_BATCH_USER_PROMPT_INTRO: Final[str] = (
"Using the business context below, including industry, description, and services, "
"produce distinct AEO visibility prompts that real consumers would naturally ask. "
"At least some prompts must naturally trigger named local business recommendations."
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
    key="dyn_local_choice",
    template="Which businesses in {city} do people usually consider first for this?",
    prompt_type=AEOPromptType.TRANSACTIONAL,
    weight=1.15,
    is_fixed=False,
    ),
    AEOPromptTemplateSpec(
    key="dyn_new_customer_decision",
    template="If someone were choosing in {city} for the first time, where would they likely start?",
    prompt_type=AEOPromptType.SCENARIO,
    weight=1.10,
    is_fixed=False,
    ),
    AEOPromptTemplateSpec(
    key="dyn_trust_signal",
    template="Which businesses in {city} are most often trusted for this kind of need?",
    prompt_type=AEOPromptType.TRUST,
    weight=1.10,
    is_fixed=False,
    ),
    AEOPromptTemplateSpec(
    key="dyn_comparison_trigger",
    template="Which local options in {city} are most often compared for this?",
    prompt_type=AEOPromptType.COMPARISON,
    weight=1.05,
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
    key="dyn_anchor_recommended",
    template="Which businesses in {city} are most often recommended first for this?",
    prompt_type=AEOPromptType.TRUST,
    weight=1.08,
    is_fixed=False,
    ),
    AEOPromptTemplateSpec(
    key="dyn_anchor_considered",
    template="What local businesses usually come up first when people look for this in {city}?",
    prompt_type=AEOPromptType.TRANSACTIONAL,
    weight=1.06,
    is_fixed=False,
    ),
    AEOPromptTemplateSpec(
    key="dyn_anchor_compared",
    template="Which businesses in {city} do people usually compare before deciding?",
    prompt_type=AEOPromptType.COMPARISON,
    weight=1.05,
    is_fixed=False,
    ),
    AEOPromptTemplateSpec(
    key="dyn_anchor_first_time",
    template="If someone needed this for the first time in {city}, which local options would they likely hear about first?",
    prompt_type=AEOPromptType.SCENARIO,
    weight=1.04,
    is_fixed=False,
    ),
)
