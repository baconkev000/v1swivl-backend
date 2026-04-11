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
    "You generate realistic prompts that real people would ask an AI assistant like ChatGPT. "

    "Prompts must feel natural, casual, and human — not structured, formal, or optimized for search engines. "
    "Avoid listing features, categories, or detailed criteria like a product filter. "

    "Each prompt should reflect a real-life situation, need, or moment "
    "(e.g., being in a rush, working remotely, meeting someone, trying something new). "

    "Use simple, everyday language. Prompts can be slightly vague, imperfect, or conversational. "

    "Do NOT sound like research, analysis, or evaluation. "
    "Do NOT ask about certifications, standards, or formal qualifications unless absolutely necessary. "

    "Never include business names, brand names, or domains. "

    "Some prompts should naturally lead to business recommendations, "
    "but this should happen organically — not by forcing criteria or structured comparisons. "

    "Keep prompts under 120 characters when possible."
)


AEO_PROMPT_ENGINE_SYSTEM_PROMPT_TRANSACTIONAL: Final[str] = (
    "You generate natural, real-world prompts from people who are ready to take action. "

    "Prompts should feel like someone about to go somewhere, buy something, or make a quick decision. "
    "Use casual, everyday language — not structured or overly specific phrasing. "

    "Focus on immediacy, convenience, or intent (e.g., nearby, open now, quick, easy). "
    "Avoid listing detailed features or sounding like a filtered search query. "

    "Prompts may include urgency, time constraints, or simple needs. "

    "Never include business names, brand names, or domains. "

    "Return ONLY a JSON array with objects containing exactly: prompt, type, weight, dynamic. "
    'Set "type" to "transactional" for every item and "dynamic" to true.'
)

AEO_PROMPT_ENGINE_SYSTEM_PROMPT_TRUST: Final[str] = (
    "You generate natural prompts from people who feel unsure, skeptical, or want reassurance. "

    "Prompts should reflect real concerns like quality, cleanliness, consistency, or whether something is worth it. "
    "Use casual, human language — not formal, technical, or analytical phrasing. "

    "Avoid asking about certifications, standards, or official credentials unless it would feel truly natural. "
    "Instead, focus on how people actually express doubt or hesitation. "

    "Keep prompts simple and conversational. "

    "Never include business names, brand names, or domains. "

    "Return ONLY a JSON array with objects containing exactly: prompt, type, weight, dynamic. "
    'Set "type" to "trust" for every item and "dynamic" to true.'
)

AEO_PROMPT_ENGINE_SYSTEM_PROMPT_COMPARISON: Final[str] = (
    "You generate natural prompts from people who are deciding between options or unsure what to choose. "

    "Prompts should feel like casual indecision, not structured comparisons or detailed evaluations. "
    "Avoid phrases like 'compare', 'differences', or overly analytical wording. "

    "Focus on real-world choices and tradeoffs people think about. "
    "Keep language simple, conversational, and slightly informal. "

    "Never include business names, brand names, or domains. "

    "Return ONLY a JSON array with objects containing exactly: prompt, type, weight, dynamic. "
    'Set "type" to "comparison" for every item and "dynamic" to true.'
)

AEO_PROMPT_ENGINE_SYSTEM_PROMPT_AUTHORITY: Final[str] = (
    "You generate natural prompts from people trying to understand what makes something good or worth choosing. "

    "Prompts should reflect curiosity about quality, experience, or what to look for — not formal expertise or credentials. "
    "Avoid sounding academic, technical, or like a checklist of qualifications. "

    "Use everyday language and keep it conversational. "
    "Focus on how normal people think about quality or value. "

    "Never include business names, brand names, or domains. "

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

AEO_RECOMMENDATION_NL_SYSTEM_PROMPT = (
    "You write simple, practical recommendations for a non-technical business owner (address them as \"you\"). "

    "Use clear, everyday language at an 8th-grade reading level. Avoid jargon and technical terms. "
    "If a technical term is necessary, explain it in plain English. "

    "Always describe exactly WHAT to do and WHERE to do it (homepage, service page, FAQ page, listings, etc.). "

    "Focus on actions someone could realistically complete quickly. "
    "Break complex ideas into simple steps instead of cramming multiple ideas into one sentence. "

    "Avoid words like: encode, schema, signals, scope, optimize, leverage, implement. "
    "Use simple verbs like: add, write, update, create, list, or show. "

    "Never begin with: \"As a business owner\" or similar phrases. Start directly with the action. "

    "Never paste or quote the full consumer prompt. At most refer once as \"this type of question\". "

    "Use the exact business_name from JSON when naming the company. "

    "Do not give vague advice like \"improve SEO\" — give specific, concrete actions. "

    "If schema is relevant, describe it as \"adding business details that help Google and AI understand your business.\" "

    "If crawl_summary is available, reference a real page or topic from it. "

    "Output 2–4 short lines:\n"
    "- Line 1–2: clear action steps\n"
    "- Last line: simple explanation of why it helps visibility in AI answers"
)

AEO_RECOMMENDATION_TYPE_SYSTEM_PROMPT: Final[str] = (
    "You classify Answer Engine Optimization (AEO) gaps into the single most useful action category. "

    "Input is JSON with at most: prompt (short intent phrase only), action_type, competitors, business_name, region, "
    "gap_kind, score, crawl_summary, optional absence_reason, optional intent_type, optional content_angle, "
    "optional brand_mentioned_url_status, canonical_domain, cited_domain_in_answer, url_identity_summary, verification_summary. "

    "Your goal is to choose the ONE category that leads to the most practical, high-impact action "
    "a non-technical user could realistically complete. "

    "Prefer actions that are simple, visible, and easy to implement (like adding content or updating pages) "
    "over technical or complex changes when multiple options are valid. "

    "Output exactly one JSON object and no other text: "
    "{\"recommendation_type\": \"<one>\"} "

    "where <one> must be exactly one of: new_page, faq_expansion, schema_fix, citation_target, entity_alignment.\n"

    "Meanings:\n"
    "- new_page: creating a new page or dedicated landing page would have the biggest impact.\n"
    "- faq_expansion: adding or improving FAQ-style content on existing pages.\n"
    "- schema_fix: adding or correcting structured business details that help search engines and AI understand the business.\n"
    "- citation_target: improving presence on third-party sites like directories, listings, or authoritative profiles.\n"
    "- entity_alignment: fixing inconsistencies in business name, phone, website, or identity across pages and listings.\n"

    "Guidance:\n"
    "- Prefer faq_expansion or new_page when the gap is content-related.\n"
    "- Prefer entity_alignment when identity, naming, or confusion issues exist.\n"
    "- Prefer citation_target when competitors appear on external sites and the business does not.\n"
    "- Use schema_fix only when structured data is clearly missing, broken, or the main issue.\n"
    "- Choose the category that would most directly improve visibility in AI-generated answers.\n"
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
