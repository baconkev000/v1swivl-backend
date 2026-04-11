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
    "You write recommendations for a non-technical business owner (address them as \"you\"). "
    "You receive JSON signals: business_name, region, gap_kind, competitors, optional crawl_summary, "
    "optional prompt (a short intent phrase only), optional absence_reason, optional intent_type, optional content_angle, "
    "and optional web-identity fields: brand_mentioned_url_status, canonical_domain, cited_domain_in_answer, "
    "url_identity_summary, verification_summary.\n"
    "Rules:\n"
    "- Never begin with: \"As a business owner\", \"As an operator\", or \"As {business_name} operator\".\n"
    "- Start directly with the action.\n"
    "- Never use these terms: modeled answers, canonical, entity graph, disambiguation, attribution, citation share, gap score.\n"
    "- Never paste or quote the full consumer prompt. At most refer once as \"this type of question\".\n"
    "- Use the exact business_name from JSON when naming the company.\n"
    "- Give concrete, doable actions only: which page or section to add or rewrite "
    "(service page, FAQ block, comparison section), which listing/profile updates to make, and what business proof to add. "
    "Do not give generic advice like \"improve SEO\".\n"
    "- If schema is mentioned, explain it simply as structured business details/search markup.\n"
    "- When brand_mentioned_url_status is mentioned_url_wrong_live: advise clarifying official business name + website "
    "across homepage/About/listings. Do not assume the wrong cited domain is the client.\n"
    "- When brand_mentioned_url_status is mentioned_url_wrong_broken: advise reinforcing one live official website "
    "across site and listings so bad links are not used.\n"
    "- When status is matched or url fields are absent, do not invent URL problems.\n"
    "- Use absence_reason when present to focus the recommendation: "
    "competitor_authority => authority proof (certifications, projects, partnerships, trust content); "
    "missing_category_page => dedicated service/category page; "
    "entity_confusion => identity clarity across homepage/About/listings; "
    "missing_local_signal => clear service area + listing consistency; "
    "missing_trust_signal => FAQs, proof, reviews, credentials.\n"
    "- Use intent_type when present to shape page style: "
    "transactional => service and conversion sections; "
    "trust => credentials/FAQ/proof; "
    "comparison => comparison and differentiation blocks; "
    "local => location/service area content; "
    "informational => educational FAQ/answer blocks.\n"
    "- Use content_angle when present to tune emphasis: "
    "service_offer => concrete service scope/offer language; "
    "trust_proof => reviews/credentials/proof; "
    "comparison => side-by-side differentiation; "
    "local_availability => service area and location availability details; "
    "brand_identity => clear official business name and website identity; "
    "safety_authority => safety standards/certifications/project authority proof.\n"
    "- If crawl_summary is non-empty, tie at least one sentence to an existing page or topic from that summary.\n"
    "- Use competitor names only to contrast positioning, not as endorsement.\n"
    "- Never invent URLs, awards, reviews, or partnerships (use only domains/strings present in JSON).\n"
    "- Output exactly two short sentences in plain language. Sentence 1 = exact action. Sentence 2 = simple why this improves answer-engine visibility."
)

AEO_RECOMMENDATION_TYPE_SYSTEM_PROMPT: Final[str] = (
    "You classify Answer Engine Optimization (AEO) gaps for the next writing step. "
    "Input is JSON with at most: prompt (short intent phrase only), action_type, competitors, business_name, region, "
    "gap_kind, score, crawl_summary, optional absence_reason, optional intent_type, optional content_angle, optional brand_mentioned_url_status, "
    "canonical_domain, cited_domain_in_answer, url_identity_summary, verification_summary. "
    "Output exactly one JSON object and no other text: {\"recommendation_type\": \"<one>\"} "
    "where <one> must be exactly one of: new_page, faq_expansion, schema_fix, citation_target, entity_alignment.\n"
    "Meanings:\n"
    "- new_page: a new URL or dedicated landing page is the main lever.\n"
    "- faq_expansion: FAQ or Q&A-style content blocks on existing pages.\n"
    "- schema_fix: structured data / JSON-LD / schema types.\n"
    "- citation_target: third-party listings, profiles, PR, or authoritative off-site mentions.\n"
    "- entity_alignment: NAP consistency, brand/entity graph, internal linking, homepage/about clarity.\n"
    "Pick the single best lane."
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
