"""
OpenAI-backed helpers for agent chat and summarization.

Keeps all OpenAI client usage and message formatting in one place.
"""

import json
import os
import re
from typing import Any, Mapping

from django.conf import settings
from django.http import HttpRequest

from openai import OpenAI
from rest_framework.response import Response

from .models import (
    AgentConversation,
    AgentMessage,
    BusinessProfile,
)
from .dataforseo_utils import (
    get_or_refresh_seo_score_for_user,
)


def build_seo_system_prompt(user, profile: BusinessProfile | None) -> str:
    """
    Build the system prompt for the SEO agent, optionally using business profile context.
    """
    base = (
        "You are an expert SEO agent that helps a small business understand and act on their search data. "
        "You speak plainly, avoid jargon, and focus on high-intent, revenue-generating opportunities. "
        "Always be specific and actionable."
    )
    if profile:
        details: list[str] = []
        if profile.business_name:
            details.append(f"Business name: {profile.business_name}.")
        if profile.industry:
            details.append(f"Industry: {profile.industry}.")
        if profile.description:
            details.append(f"Business description: {profile.description}.")
        if details:
            base += " Here is context about the business: " + " ".join(details)
    return base


def build_reviews_system_prompt(user, profile: BusinessProfile | None) -> str:
    """
    Build the system prompt for the Reviews Agent (trust, reputation, review response).
    Different role from SEO; same structure (optionally use business profile context).
    """
    base = (
        "You are an expert Reviews and Reputation agent that helps a small business "
        "build trust, respond to reviews, and turn feedback into marketing leverage. "
        "You focus on: responding to reviews in a brand-aligned way, identifying praise themes "
        "for ad copy, flagging recurring complaints, and improving close rate through trust. "
        "You speak plainly and are specific and actionable. "
        "Never argue with reviewers; never sound robotic."
    )
    if profile:
        details: list[str] = []
        if profile.business_name:
            details.append(f"Business name: {profile.business_name}.")
        if profile.industry:
            details.append(f"Industry: {profile.industry}.")
        if profile.description:
            details.append(f"Business description: {profile.description}.")
        if details:
            base += " Here is context about the business: " + " ".join(details)
    return base


def _get_client(api_key_env: str | None = None) -> OpenAI:
    """
    Return an OpenAI client using the given env var for the API key.

    - If api_key_env is set (e.g. OPEN_AI_SEO_API_KEY, OPEN_AI_REVIEWS_API_KEY),
      use that env var, then fall back to OPENAI_API_KEY.
    - Otherwise use OPENAI_API_KEY.
    """
    if api_key_env:
        api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def _get_model() -> str:
    """Resolve chat model from ``settings.OPENAI_MODEL`` (env ``OPENAI_MODEL``)."""
    raw = getattr(settings, "OPENAI_MODEL", "") or ""
    s = str(raw).strip()
    return s or "gpt-4o-mini"


def chat_completion_create_logged(
    client: OpenAI,
    *,
    operation: str,
    business_profile: BusinessProfile | None = None,
    emit_api_error_log: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Call ``client.chat.completions.create`` and append a usage log row on success (best-effort).

    On API failure: when ``emit_api_error_log`` is True (default), records one error row and re-raises.
    Callers with internal retries should pass ``emit_api_error_log=False`` and log once after retries exhaust.
    """
    from accounts.models import ThirdPartyApiProvider
    from accounts.third_party_usage import (
        classify_openai_sdk_exception,
        record_openai_chat_completion,
        record_third_party_api_error,
    )

    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as exc:
        if emit_api_error_log:
            kind, http = classify_openai_sdk_exception(exc)
            record_third_party_api_error(
                provider=ThirdPartyApiProvider.OPENAI,
                operation=operation,
                error_kind=kind,
                message=str(exc)[:1024],
                detail=None,
                http_status=http,
                business_profile=business_profile,
            )
        raise
    record_openai_chat_completion(
        operation=operation,
        response=completion,
        business_profile=business_profile,
    )
    return completion


def _get_chat_reply(
    system_prompt: str,
    recent_messages: list,
    conversation_summary: str | None = None,
    api_key_env: str | None = None,
    business_profile: BusinessProfile | None = None,
) -> str:
    """
    Call OpenAI chat completion. recent_messages must have .role and .content.
    api_key_env: env var for API key (e.g. OPEN_AI_SEO_API_KEY, OPEN_AI_REVIEWS_API_KEY).
    Returns the assistant reply text.
    """
    openai_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    if conversation_summary:
        openai_messages.append(
            {
                "role": "system",
                "content": f"Conversation summary so far (memory): {conversation_summary}",
            },
        )
    for msg in recent_messages:
        openai_messages.append(
            {"role": msg.role, "content": msg.content},
        )

    client = _get_client(api_key_env)
    model = _get_model()
    completion = chat_completion_create_logged(
        client,
        operation="openai.chat.seo_agent_reply",
        business_profile=business_profile,
        model=model,
        messages=openai_messages,
    )
    return (completion.choices[0].message.content or "").strip()


def get_seo_chat_reply(
    system_prompt: str,
    recent_messages: list[AgentMessage],
    conversation_summary: str | None = None,
    *,
    business_profile: BusinessProfile | None = None,
) -> str:
    """Call OpenAI for SEO agent using OPEN_AI_SEO_API_KEY."""
    return _get_chat_reply(
        system_prompt,
        recent_messages,
        conversation_summary,
        api_key_env="OPEN_AI_SEO_API_KEY",
        business_profile=business_profile,
    )


def summarize_seo_conversation(
    messages: list[AgentMessage],
    *,
    business_profile: BusinessProfile | None = None,
) -> str:
    """
    Ask OpenAI to summarize a list of messages into concise memory notes.
    """
    payload: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Summarize the following SEO conversation into concise memory notes. "
                "Capture key goals, constraints, and decisions. 5-10 bullet points max."
            ),
        },
    ]
    for m in messages:
        payload.append({"role": m.role, "content": m.content})

    client = _get_client("OPEN_AI_SEO_API_KEY")
    model = _get_model()
    completion = chat_completion_create_logged(
        client,
        operation="openai.chat.summarize_seo_conversation",
        business_profile=business_profile,
        model=model,
        messages=payload,
    )
    return (completion.choices[0].message.content or "").strip()


def generate_seo_keyword_candidates(
    profile: BusinessProfile | None,
    homepage_meta: str | None = None,
) -> list[str]:
    """
    Ask OpenAI to generate 10–15 candidate SEO keyword phrases for the business.
    Used by the keyword enrichment pipeline; output is validated via DataForSEO search volume.
    Returns a list of phrases (2–4 words, search-intent, no explanations).
    """
    if not profile:
        return []

    parts: list[str] = []
    if profile.business_name:
        parts.append(f"Business name: {profile.business_name}.")
    if profile.industry:
        parts.append(f"Industry: {profile.industry}.")
    if profile.description:
        parts.append(f"Description: {profile.description}.")
    if homepage_meta:
        parts.append(f"Homepage meta/title: {homepage_meta}.")

    if not parts:
        return []

    system = (
        "You are an SEO expert. Generate 10–15 candidate keyword phrases that real users would type into Google "
        "when looking for this business or its services. Rules: return ONLY search-intent phrases; 2–4 words per phrase; "
        "no sentence fragments; no explanations; no generic phrases like 'best in area' or 'near me' unless clearly relevant; "
        "no brand repetition unless the brand keyword is valuable; prefer transactional/commercial intent. "
        "Output exactly one phrase per line, nothing else (no numbering, no bullets)."
    )
    user_content = "Generate 10–15 SEO keyword phrases for this business:\n\n" + "\n".join(parts)

    try:
        client = _get_client("OPEN_AI_SEO_API_KEY")
        model = _get_model()
        completion = chat_completion_create_logged(
            client,
            operation="openai.chat.seo_keyword_candidates",
            business_profile=profile,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        return []

    candidates: list[str] = []
    for line in raw.splitlines():
        phrase = line.strip()
        # Strip leading numbers/bullets (e.g. "1. phrase" or "- phrase")
        while phrase and phrase[0] in "0123456789.-) ":
            phrase = phrase.lstrip("0123456789.-) ")
        if not phrase or len(phrase) > 80:
            continue
        word_count = len(phrase.split())
        if 2 <= word_count <= 4:
            candidates.append(phrase)
        elif word_count == 1 and len(phrase) >= 4:
            candidates.append(phrase)
    return candidates[:15]


def _merged_seo_issue_inputs(seo_data: dict, snapshot: Any) -> tuple[dict[str, Any], list[Any]]:
    base_on: dict[str, Any] = {}
    base_serp: list[Any] = []
    if snapshot is not None:
        from .dataforseo_utils import seo_issue_aux_context_for_snapshot

        aux = seo_issue_aux_context_for_snapshot(snapshot)
        base_on = dict(aux.get("on_page") or {})
        base_serp = list(aux.get("serp") or [])

    if "on_page" in seo_data:
        on_page = {**base_on, **dict(seo_data.get("on_page") or {})}
    else:
        on_page = dict(base_on)

    if "serp" in seo_data:
        serp = list(seo_data.get("serp") or [])
    else:
        serp = list(base_serp)
        for row in list(seo_data.get("top_keywords") or [])[:80]:
            if not isinstance(row, dict):
                continue
            sr = row.get("serp_rows")
            if isinstance(sr, list):
                serp.extend(sr)
        serp = serp[:500]
    return on_page, serp


def _persist_snapshot_structured_seo_issues(snapshot: Any, issues: list[dict[str, Any]]) -> None:
    from django.utils import timezone

    snapshot.seo_structured_issues = list(issues)
    snapshot.seo_structured_issues_refreshed_at = timezone.now()
    snapshot.save(
        update_fields=["seo_structured_issues", "seo_structured_issues_refreshed_at"],
    )


def _recommendation_priority_weight(priority: str) -> int:
    p = str(priority or "").strip().lower()
    if p == "high":
        return 3
    if p == "medium":
        return 2
    return 1


def _recommendation_rank_key(rec: dict) -> tuple[float, float]:
    try:
        impact = float(rec.get("impact_score") or 0.0)
    except (TypeError, ValueError):
        impact = 0.0
    try:
        confidence = float(rec.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        effort = float(rec.get("effort_score") or 0.0)
    except (TypeError, ValueError):
        effort = 0.0
    priority = _recommendation_priority_weight(str(rec.get("priority") or "low"))
    return (priority * 10.0 + impact * confidence * (1.0 - max(0.0, min(1.0, effort))), impact)


def _enforce_structured_recommendation_constraints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validation/quality constraints for deterministic recommendations:
    - max 1 recommendation per keyword cluster
    - max 2 page_creation recommendations
    - max 8 total
    """
    if not rows:
        return []
    sorted_rows = sorted(list(rows), key=_recommendation_rank_key, reverse=True)
    out: list[dict[str, Any]] = []
    seen_clusters: set[str] = set()
    create_page_n = 0
    for rec in sorted_rows:
        execution = rec.get("execution")
        if not isinstance(execution, dict) or not execution:
            continue
        evidence = rec.get("evidence") or {}
        cluster_id = str(evidence.get("cluster_id") or "").strip()
        if cluster_id and cluster_id in seen_clusters:
            continue
        rec_type = str(rec.get("type") or "")
        action_type = str(rec.get("recommended_action_type") or "")
        is_create = rec_type == "page_creation" or action_type == "create_cluster_page"
        if is_create:
            if create_page_n >= 2:
                continue
            create_page_n += 1
        out.append(rec)
        if cluster_id:
            seen_clusters.add(cluster_id)
        if len(out) >= 8:
            break
    return out


def generate_seo_next_steps(seo_data: dict, *, snapshot: Any | None = None) -> list[dict]:
    """
    Produce up to eight structured next steps grounded in ``build_structured_issues`` evidence.

    Each step is rewritten for non-technical business owners: clear outcome title, plain-language
    ``why_it_matters``, optional ``expected_benefit``, and owner-friendly steps. Technical SEO work
    is directed to ``internal_notes`` only (not mixed into customer-facing copy).

    ``seo_data`` should include scores, ``top_keywords``, and optionally ``on_page`` / ``serp`` for the
    deterministic issue engine. When ``snapshot`` is set, merges crawl-derived FAQ signals and
    persists structured issues on the snapshot.

    Returns dicts with ``title``, ``why_it_matters``, ``exact_fix``, ``example``, optional
    ``expected_benefit`` / ``internal_notes`` / ``action_card``, plus ``label`` / ``tag`` for compatibility.
    """
    if not seo_data:
        return []

    from .seo.seo_issue_engine import (
        build_structured_issues,
        build_structured_recommendations,
    )

    keywords = list(seo_data.get("top_keywords") or [])
    on_page, serp = _merged_seo_issue_inputs(seo_data, snapshot)
    # ``top_keywords`` rows can include competitor fields after gap enrichment; pass those rows
    # to both ranked and domain_intersection channels so deterministic rules can trigger.
    issues = build_structured_issues(
        ranked_keywords=keywords,
        domain_intersection=keywords,
        on_page=on_page,
        serp=serp,
    )
    if snapshot is not None:
        _persist_snapshot_structured_seo_issues(snapshot, list(issues))
    if not issues:
        return []

    deterministic = _enforce_structured_recommendation_constraints(
        build_structured_recommendations(issues)
    )
    snap_ctx = seo_data.get("snapshot_context_for_rewrite")
    out: list[dict] = []
    for rec in deterministic:
        rewritten = _rewrite_structured_seo_recommendation(
            rec,
            snapshot_context=snap_ctx if isinstance(snap_ctx, dict) else None,
        )
        merged = dict(rec)
        merged.update(rewritten)
        # Backward compatibility for old consumers.
        merged["label"] = str(merged.get("title") or merged.get("issue") or "").strip()
        merged["tag"] = (
            "High priority"
            if merged.get("priority") == "high"
            else "Medium priority"
            if merged.get("priority") == "medium"
            else "Low priority"
        )
        if merged["label"]:
            out.append(merged)
    return out[:8]


def generate_keyword_action_suggestions(
    keywords: list[dict],
    *,
    snapshot: Any | None = None,
    snapshot_seo_data: dict | None = None,
) -> list[dict]:
    """
    Given keyword dicts from top_keywords / search demand, produce one action per row using the same
    structured-issue pipeline as ``generate_seo_next_steps``, reframed for AI citation / answerability
    in the LLM rewrite layer.

    When ``snapshot`` is set, uses the same on-page / SERP merge and persists structured issues.

    Optional ``snapshot_seo_data`` merges snapshot-wide context while keeping the passed ``keywords``
    list as ``top_keywords`` for ranking rules.

    Returns a list of {\"keyword\": str, \"suggestion\": str, ...} (extended fields unchanged).
    """
    if not keywords:
        return []
    from .seo.seo_issue_engine import (
        build_structured_issues,
        build_structured_recommendations,
    )

    rows = list(keywords)
    if snapshot_seo_data is not None:
        fake_seo = {**dict(snapshot_seo_data), "top_keywords": rows}
    else:
        fake_seo = {"top_keywords": rows}
    on_page, serp = _merged_seo_issue_inputs(fake_seo, snapshot)
    issues = build_structured_issues(
        ranked_keywords=rows, domain_intersection=rows, on_page=on_page, serp=serp
    )
    if snapshot is not None:
        _persist_snapshot_structured_seo_issues(snapshot, list(issues))
    if not issues:
        return []
    deterministic = _enforce_structured_recommendation_constraints(
        build_structured_recommendations(issues)
    )
    snap_ctx = fake_seo.get("snapshot_context_for_rewrite")
    out: list[dict] = []
    for rec in deterministic:
        rewritten = _rewrite_structured_seo_recommendation(
            rec,
            snapshot_context=snap_ctx if isinstance(snap_ctx, dict) else None,
        )
        merged = dict(rec)
        merged.update(rewritten)
        ev = merged.get("evidence") or {}
        kw = str(ev.get("primary_keyword") or ev.get("keyword") or "").strip()
        suggestion = str(merged.get("exact_fix") or "").strip()
        if kw and suggestion:
            out.append(
                {
                    "keyword": kw,
                    "suggestion": suggestion,
                    "title": str(merged.get("title") or merged.get("issue") or "").strip(),
                    "priority": str(merged.get("priority") or "medium"),
                    "why_it_matters": str(merged.get("why_it_matters") or "").strip(),
                    "exact_fix": suggestion,
                    "example": str(merged.get("example") or "").strip(),
                    "evidence": merged.get("evidence") or {},
                    "issue_id": str(merged.get("issue_id") or ""),
                }
            )
    return out


# System prompt for LLM rewrite of deterministic SEO issues → display-ready action cards.
SEO_STRUCTURED_REWRITE_SYSTEM_PROMPT = (
    "You are a business coach and marketing consultant generating ACTION-CARD DATA for a non-technical business owner.\n"
    "Write like a friendly expert explaining growth steps — NOT like SEO software, a developer tool, or an audit report.\n\n"
    "INPUT JSON contains prioritized deterministic actions plus keyword context. Use it ONLY to infer what to recommend; "
    "never echo internal issue_ids, snake_case slugs, or raw diagnostic labels in user-facing strings.\n\n"
    "TITLE (most important):\n"
    "- One clear, human headline that describes a BUSINESS OUTCOME (visibility, trust, or conversion).\n"
    "- Must be understandable in one second with zero SEO vocabulary.\n"
    "- Convert abstract or technical labels into outcomes. BAD: local_trust_gap, LocalBusiness schema, open places near me optimization. "
    "GOOD: Make your business easier to find in local searches; Help customers choose you with clearer contact details.\n\n"
    "WHY IT MATTERS:\n"
    "- Plain language only: how this affects customers finding them, trusting them, or taking action.\n"
    "- No algorithm talk, ranking-math, or search-engine mechanics.\n\n"
    "EXPECTED BENEFIT (recommended):\n"
    "- One short outcome sentence, e.g. More local customers can discover your business when they search nearby.\n\n"
    "STEPS (user-facing only):\n"
    "- Short, imperative, real-world actions the owner can take or delegate (website copy, contact page, service area wording).\n"
    "- No schema/JSON-LD/markup instructions here.\n\n"
    "IMPLEMENTATION SEPARATION (critical):\n"
    "- Put ALL technical SEO implementation (schema types, JSON-LD, NAP consistency, canonicals, hreflang, crawl/index notes, "
    "markup names like LocalBusiness/FAQPage, developer-only fixes) in internal_notes ONLY.\n"
    "- internal_notes must NEVER be copied into title, subtitle, goal, why_it_matters, steps, example, or copy_paste_content.\n"
    "- Leave schema_requirements as an empty array [] (technical items belong in internal_notes only).\n\n"
    "TERMINOLOGY — never use these in user-facing fields; if a concept is needed, plain-English only:\n"
    "- Schema / JSON-LD / structured data → say search engine information or describe the visitor outcome without naming formats.\n"
    "- NAP → business contact details (name, address, phone).\n"
    "- Indexing → showing up in search results.\n"
    "- Ranking signals → visibility in search results.\n"
    "- LocalBusiness markup / FAQPage schema → omit entirely from user-facing text.\n\n"
    "PRIORITY LENS: each action should map loosely to Visibility (being found), Trust (being chosen), or Conversion (getting customers).\n\n"
    "Return ONLY valid JSON. No markdown. No extra prose.\n"
    "Preferred output shape:\n"
    "{"
    "\"actions\":[{"
    "\"id\":\"...\","
    "\"source\":\"seo|keywords|aeo\","
    "\"category_label\":\"SEO|Keyword|AEO\","
    "\"pillar\":\"Content|Technical|Trust|Presence\","
    "\"priority\":\"high|medium|low\","
    "\"title\":\"...\","
    "\"subtitle\":\"...\","
    "\"why_it_matters\":\"...\","
    "\"expected_benefit\":\"...\","
    "\"goal\":\"...\","
    "\"whats_missing\":[\"...\"],"
    "\"internal_notes\":[\"...\"],"
    "\"steps\":[{\"step_number\":1,\"title\":\"...\",\"instruction\":\"...\"}],"
    "\"copy_paste_content\":{\"local_trust_block\":\"...\",\"faq\":[{\"q\":\"...\",\"a\":\"...\"}],\"quick_facts\":[\"...\"]},"
    "\"schema_requirements\":[],"
    "\"internal_linking\":[{\"from_or_section\":\"...\",\"to_url\":\"...\",\"anchor_hint\":\"...\"}],"
    "\"target_url\":\"...\","
    "\"evidence\":{\"keyword\":\"...\",\"search_volume\":0,\"rank\":0,\"competitor_rank\":0,\"competitor_domains\":[\"...\"],\"location\":\"...\",\"source_issue_ids\":[\"...\"]},"
    "\"display_hints\":{\"expanded_by_default\":false,\"show_copy_paste_section\":true,\"show_schema_section\":false,\"show_internal_linking_section\":true}"
    "}]"
    "}\n"
    "If you cannot build the full card, return legacy compact JSON with keys: "
    "title, why_it_matters, exact_fix, example, optional expected_benefit, optional internal_notes (string array).\n\n"
    "STRICT RULES:\n"
    "- Do NOT invent keywords, ranks, URLs, search volumes, competitor domains, location, or issue IDs.\n"
    "- Deduplicate overlapping recommendations and keep the richer merged result.\n"
    "- No vague directives like 'optimize SEO' or 'improve content' without concrete owner-friendly steps.\n"
    "- whats_missing should be short customer-facing gaps (plain English), not developer tickets.\n"
    "- Use empty strings/arrays instead of guessing unknown values.\n"
)

_SEO_VAGUE_PATTERN = re.compile(
    r"\b(optimi[sz]e\s+seo|improve\s+seo\b|enhance\s+seo\b|increase\s+visibility|boost\s+rankings?|rank\s+higher|"
    r"get\s+more\s+traffic|drive\s+more\s+clicks|climb\s+the\s+serp)\b",
    re.IGNORECASE,
)
# Owner-friendly steps should sound like real marketing tasks, not developer tickets.
_SUBSTANTIVE_OWNER_STEP_PATTERN = re.compile(
    r"\b(add|show|write|update|include|put|create|list|describe|mention|phone|address|hours?|"
    r"services?|service\s+area|neighborhood|city|cities|towns?|contact|website|web\s+site|homepage|"
    r"contact\s+page|location|map|directions|reviews?|photos?|customers|business\s+name|"
    r"clearly|where\s+you\s+serve|serve|serving|book|call|email|form|page|section|paragraph|"
    r"headline|intro|about|team|coverage|areas?)\b",
    re.IGNORECASE,
)
# Vague “AI visibility” claims without concrete owner-facing steps.
_AEO_VAGUE_PATTERN = re.compile(
    r"\b(ai\s+visibility|be\s+seen\s+in\s+ai|win\s+at\s+aeo|dominate\s+ai)\b",
    re.IGNORECASE,
)
_BANNED_GENERIC_WORDS = re.compile(r"\b(optimi[sz]e|improve|enhance)\b", re.IGNORECASE)
_SEO_FORBIDDEN_USER_JARGON = re.compile(
    r"\b(?:schema|json-ld|json\s*ld|structured\s+data|localbusiness|faqpage|naps?|ranking\s+signals?|"
    r"indexation|indexing|crawl\s+budget|canonicals?|301\s+redirect|hreflang|rich\s+results|entity\s+signals?|"
    r"eeat|e-e-a-t|serps?|sge|search\s+console|core\s+web\s+vitals|lcp|cls)\b",
    re.IGNORECASE,
)


def _looks_like_internal_issue_slug(title: str) -> bool:
    t = (title or "").strip()
    if not t or " " in t:
        return False
    if "_" not in t:
        return False
    return bool(re.match(r"^[a-z0-9_]+$", t.lower()))


def _seo_rewrite_output_is_valid(payload: dict, recommendation: dict) -> bool:
    title = str(payload.get("title") or "").strip()
    why = str(payload.get("why_it_matters") or "").strip()
    fix = str(payload.get("exact_fix") or "").strip()
    if not title or not why or not fix:
        return False
    if _looks_like_internal_issue_slug(title):
        return False
    blob = f"{title} {why} {fix} {payload.get('example') or ''}"
    if _SEO_FORBIDDEN_USER_JARGON.search(blob):
        return False
    if len(fix) < 35:
        return False
    if _SEO_VAGUE_PATTERN.search(fix) and not _SUBSTANTIVE_OWNER_STEP_PATTERN.search(fix):
        return False
    if _AEO_VAGUE_PATTERN.search(fix) and not _SUBSTANTIVE_OWNER_STEP_PATTERN.search(fix):
        return False
    if _BANNED_GENERIC_WORDS.search(fix) and not _SUBSTANTIVE_OWNER_STEP_PATTERN.search(fix):
        return False
    evidence = recommendation.get("evidence") or {}
    keyword = str(evidence.get("primary_keyword") or evidence.get("keyword") or "").strip().lower()
    url = str(evidence.get("url") or evidence.get("your_url") or "").strip().lower()
    combined = f"{title} {why} {fix} {payload.get('example') or ''}".lower()
    if keyword and keyword not in combined and url and url not in combined:
        return False
    return True


def _coerce_rewrite_payload_to_legacy(data: dict) -> dict:
    """
    Accept execution_tasks[] JSON and coerce to legacy keys used by SEO overview responses.
    Keeps compatibility with existing API consumers while allowing richer prompt outputs.
    """
    if not isinstance(data, dict):
        return {}
    if all(k in data for k in ("title", "why_it_matters", "exact_fix")):
        internal_notes = data.get("internal_notes")
        if not isinstance(internal_notes, list):
            internal_notes = []
        internal_notes = [str(x).strip() for x in internal_notes if str(x).strip()]
        return {
            "title": str(data.get("title") or "").strip(),
            "why_it_matters": str(data.get("why_it_matters") or "").strip(),
            "exact_fix": str(data.get("exact_fix") or "").strip(),
            "example": str(data.get("example") or "").strip(),
            "expected_benefit": str(data.get("expected_benefit") or "").strip(),
            "internal_notes": internal_notes,
        }

    cards = data.get("actions")
    if isinstance(cards, list) and cards and isinstance(cards[0], dict):
        first_card = cards[0]
        title = str(first_card.get("title") or "").strip()
        goal = str(first_card.get("goal") or "").strip()
        why_direct = str(first_card.get("why_it_matters") or "").strip()
        missing = [str(x).strip() for x in list(first_card.get("whats_missing") or []) if str(x).strip()]
        # Prefer explicit why_it_matters / goal; do not merge technical whats_missing into the main explanation.
        why = why_direct or goal
        if not why and missing:
            why = ". ".join(missing[:3])

        steps = list(first_card.get("steps") or [])
        step_lines: list[str] = []
        for row in steps[:6]:
            if not isinstance(row, dict):
                continue
            st = str(row.get("title") or "").strip()
            ins = str(row.get("instruction") or "").strip()
            line = " — ".join([x for x in [st, ins] if x]).strip()
            if line:
                step_lines.append(line)
        exact_fix = " ".join([f"{i + 1}) {line}" for i, line in enumerate(step_lines)]).strip()
        if not exact_fix:
            exact_fix = goal

        cpc = first_card.get("copy_paste_content") or {}
        example = ""
        if isinstance(cpc, dict):
            trust = str(cpc.get("local_trust_block") or "").strip()
            faq = cpc.get("faq")
            quick = cpc.get("quick_facts")
            if trust:
                example = trust
            elif isinstance(faq, list) and faq and isinstance(faq[0], dict):
                q = str(faq[0].get("q") or "").strip()
                a = str(faq[0].get("a") or "").strip()
                if q and a:
                    example = f"{q} — {a}"
            elif isinstance(quick, list) and quick:
                example = str(quick[0] or "").strip()

        exp_ben = str(first_card.get("expected_benefit") or "").strip()
        if not exp_ben:
            ai_notes = first_card.get("ai_optimization_notes") or {}
            if isinstance(ai_notes, dict):
                exp_lines = ai_notes.get("expected_impact")
                if isinstance(exp_lines, list) and exp_lines:
                    exp_ben = str(exp_lines[0] or "").strip()

        internal_raw = first_card.get("internal_notes")
        internal_notes: list[str] = []
        if isinstance(internal_raw, list):
            internal_notes = [str(x).strip() for x in internal_raw if str(x).strip()]

        return {
            "title": title,
            "why_it_matters": why,
            "exact_fix": exact_fix,
            "example": example,
            "expected_benefit": exp_ben,
            "internal_notes": internal_notes,
            "action_card": first_card,
        }

    tasks = data.get("execution_tasks")
    if not isinstance(tasks, list) or not tasks:
        return {}
    first = tasks[0] if isinstance(tasks[0], dict) else {}
    if not first:
        return {}

    title = str(first.get("title") or "").strip()
    goal = str(first.get("goal") or "").strip()
    ai_notes = str(first.get("ai_optimization_notes") or "").strip()
    why = " ".join([x for x in [goal, ai_notes] if x]).strip()

    impl_raw = first.get("implementation")
    impl_list = [str(x).strip() for x in (impl_raw or []) if str(x).strip()] if isinstance(impl_raw, list) else []
    exact_fix = " ".join([f"{idx + 1}) {step}" for idx, step in enumerate(impl_list)]).strip()
    if not exact_fix:
        exact_fix = str(first.get("goal") or "").strip()

    content = first.get("content_requirements") or {}
    example = ""
    if isinstance(content, dict):
        trust = str(content.get("trust_block_example") or "").strip()
        faq = content.get("faq")
        quick = content.get("quick_facts")
        if trust:
            example = trust
        elif isinstance(faq, list) and faq and isinstance(faq[0], dict):
            q = str(faq[0].get("q") or faq[0].get("question") or "").strip()
            a = str(faq[0].get("a") or faq[0].get("answer") or "").strip()
            if q and a:
                example = f"{q} — {a}"
        elif isinstance(quick, list) and quick:
            example = str(quick[0] or "").strip()

    internal_raw = first.get("internal_notes")
    internal_notes: list[str] = []
    if isinstance(internal_raw, list):
        internal_notes = [str(x).strip() for x in internal_raw if str(x).strip()]

    return {
        "title": title,
        "why_it_matters": why,
        "exact_fix": exact_fix,
        "example": example,
        "expected_benefit": str(first.get("expected_benefit") or "").strip(),
        "internal_notes": internal_notes,
    }


def _rewrite_structured_seo_recommendation(
    recommendation: dict,
    *,
    snapshot_context: dict | None = None,
) -> dict:
    """
    AI rewrite layer for deterministic SEO issue recommendations.

    Produces owner-friendly ``title``, ``why_it_matters``, and ``exact_fix`` plus optional
    ``expected_benefit`` and ``internal_notes`` (technical items must stay in internal_notes only).
    """
    system = SEO_STRUCTURED_REWRITE_SYSTEM_PROMPT
    ev = recommendation.get("evidence") or {}
    keyword_rows = []
    if isinstance(ev, dict):
        primary_keyword = str(ev.get("primary_keyword") or ev.get("keyword") or "").strip()
        if primary_keyword:
            keyword_rows.append(
                {
                    "keyword": primary_keyword,
                    "search_volume": ev.get("search_volume"),
                    "rank": ev.get("rank"),
                }
            )
        for kw in list(ev.get("target_keywords") or []):
            kws = str(kw or "").strip()
            if kws and kws.lower() != primary_keyword.lower():
                keyword_rows.append({"keyword": kws})
    user_payload = {
        "actions": [recommendation],
        "keywords": keyword_rows[:12],
    }
    user_content = "Input:\n" + json.dumps(user_payload, ensure_ascii=False)
    if snapshot_context:
        user_content += "\n\nSnapshot context (read-only; never invent facts):\n"
        user_content += json.dumps(snapshot_context, ensure_ascii=False)
    fallback = {
        "title": str(recommendation.get("issue") or "").strip(),
        "why_it_matters": str(recommendation.get("why_it_matters") or "").strip(),
        "exact_fix": str(recommendation.get("exact_fix") or "").strip(),
        "example": str(recommendation.get("example") or "").strip(),
        "expected_benefit": str(recommendation.get("expected_benefit") or "").strip(),
        "internal_notes": list(recommendation.get("internal_notes") or [])
        if isinstance(recommendation.get("internal_notes"), list)
        else [],
    }

    for _ in range(2):
        try:
            client = _get_client("OPEN_AI_SEO_API_KEY")
            model = _get_model()
            completion = chat_completion_create_logged(
                client,
                operation="openai.chat.seo_structured_rewrite",
                business_profile=None,
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            )
            raw = (completion.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rstrip("`").strip()
            data = json.loads(raw)
            if not isinstance(data, dict):
                continue
            candidate = _coerce_rewrite_payload_to_legacy(data)
            if _seo_rewrite_output_is_valid(candidate, recommendation):
                return candidate
        except Exception:
            continue
    return fallback


def generate_aeo_recommendations(aeo_data: dict, seo_data: dict | None = None) -> list[str]:
    """
    Generate exactly 5 actionable AEO next-step suggestions.
    Uses AEO metrics plus relevant SEO context to prioritize impact.
    Returns a list of suggestion strings.
    """
    if not aeo_data:
        return []

    seo_data = seo_data or {}
    question_coverage = int(aeo_data.get("question_coverage_score") or 0)
    faq_readiness = int(aeo_data.get("faq_readiness_score") or 0)
    snippet_readiness = int(aeo_data.get("snippet_readiness_score") or 0)
    questions_missing = list(aeo_data.get("questions_missing") or [])[:12]
    top_keywords = list((seo_data or {}).get("top_keywords") or [])[:12]
    visibility = seo_data.get("search_visibility_percent")
    organic = seo_data.get("organic_visitors")
    missed = seo_data.get("missed_searches_monthly")

    keyword_preview = []
    for row in top_keywords:
        kw = str((row or {}).get("keyword") or "").strip()
        sv = int((row or {}).get("search_volume") or 0)
        rank = (row or {}).get("rank")
        if kw:
            keyword_preview.append(f"{kw} (sv={sv}, rank={rank})")

    system = (
        "You are an Answer Engine Optimization strategist. "
        "Given AEO and SEO performance data, produce exactly 5 highly actionable next steps "
        "to improve AEO score in the next 30 days. Focus on question coverage, FAQ quality, "
        "snippet-ready answer blocks, and entity clarity. "
        "Output only a JSON array of 5 plain strings. No markdown, no numbering, no extra text."
    )
    user_content = (
        "AEO data:\n"
        f"- question_coverage_score: {question_coverage}\n"
        f"- faq_readiness_score: {faq_readiness}\n"
        f"- snippet_readiness_score: {snippet_readiness}\n"
        f"- sample_questions_missing: {questions_missing}\n\n"
        "Relevant SEO context:\n"
        f"- search_visibility_percent: {visibility}\n"
        f"- organic_visitors: {organic}\n"
        f"- missed_searches_monthly: {missed}\n"
        f"- top_keywords: {keyword_preview}\n\n"
        "Return a JSON array of exactly 5 short actionable recommendations."
    )

    try:
        client = _get_client("OPEN_AI_SEO_API_KEY")
        model = _get_model()
        completion = chat_completion_create_logged(
            client,
            operation="openai.chat.aeo_recommendations",
            business_profile=None,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rstrip("`").strip()
        payload = json.loads(raw)
        if not isinstance(payload, list):
            return []
        out: list[str] = []
        for item in payload:
            rec = str(item or "").strip()
            if rec:
                out.append(rec)
        return out[:5]
    except Exception:
        return []


def _strip_json_fence(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        if "```" in raw:
            raw = raw.rsplit("```", 1)[0].strip()
    return raw


def _gp_trunc(s: str, max_len: int) -> str:
    t = " ".join(str(s or "").split()).strip()
    if len(t) <= max_len:
        return t
    return t[: max(0, max_len - 1)].rstrip() + "…"


def _gp_as_str_list(raw: Any, cap: int, item_max: int = 480) -> list[str]:
    out: list[str] = []
    if isinstance(raw, list):
        for x in raw:
            s = _gp_trunc(str(x or ""), item_max)
            if s:
                out.append(s)
            if len(out) >= cap:
                break
    elif isinstance(raw, str) and raw.strip():
        out.append(_gp_trunc(raw.strip(), item_max))
    return out[:cap]


def _gp_parse_faqs(raw: Any, cap: int = 8, qa_max: int = 480) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw[: cap * 2]:
        if not isinstance(item, dict):
            continue
        q = _gp_trunc(str(item.get("q") or item.get("question") or ""), qa_max)
        a = _gp_trunc(str(item.get("a") or item.get("answer") or ""), qa_max)
        if q and a:
            out.append({"q": q, "a": a})
        if len(out) >= cap:
            break
    return out


def _gp_parse_plan_steps(raw: Any, cap: int = 12, title_max: int = 200, instr_max: int = 900) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            sn = int(item.get("step_number", i + 1))
        except (TypeError, ValueError):
            sn = i + 1
        title = _gp_trunc(str(item.get("title") or ""), title_max)
        instruction = _gp_trunc(str(item.get("instruction") or ""), instr_max)
        if title or instruction:
            out.append({"step_number": sn, "title": title, "instruction": instruction})
        if len(out) >= cap:
            break
    return out


def _gp_parse_internal_links(raw: Any, cap: int = 12) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "from_or_section": _gp_trunc(str(item.get("from_or_section") or ""), 200),
                "to_url": _gp_trunc(str(item.get("to_url") or ""), 400),
                "anchor_hint": _gp_trunc(str(item.get("anchor_hint") or ""), 200),
            }
        )
        if len(out) >= cap:
            break
    return out


def _build_actions_landing_page_context_block(
    *,
    seo_issues: list[str],
    action_faqs: list[dict[str, str]],
    trust_block: str,
    quick_facts: list[str],
    plan_steps: list[dict[str, Any]],
    internal_links: list[dict[str, str]],
    affected_keywords: list[str],
    target_keywords: list[str],
    copy_example: str,
) -> str:
    """Human-readable block for the user message (bounded)."""
    lines: list[str] = []

    issues_lines = "\n".join(f"- {_gp_trunc(str(x), 480)}" for x in (seo_issues or [])[:32] if str(x).strip())
    if not issues_lines.strip():
        issues_lines = "- None specified"
    lines.append(
        "INTERNAL — Topic and service gaps (structure the page around these in natural language only; "
        "never present them as an audit, checklist, or SEO/ranking explanation on the page)"
    )
    lines.append(issues_lines)

    if action_faqs:
        lines.append("\nASSETS — FAQ (use faithfully where relevant; do not invent Q/As beyond these)")
        for pair in action_faqs:
            lines.append(f"Q: {pair.get('q', '')}\nA: {pair.get('a', '')}")

    tb = (trust_block or "").strip()
    if tb:
        lines.append("\nASSETS — Business facts / local context (use verbatim where appropriate; do not invent credentials)")
        lines.append(tb)

    if quick_facts:
        lines.append("\nASSETS — Quick facts / citation bullets")
        for qf in quick_facts:
            lines.append(f"- {_gp_trunc(qf, 480)}")

    if plan_steps:
        lines.append("\nPLAN — Full steps (titles + instructions; follow this execution detail)")
        for st in plan_steps:
            sn = st.get("step_number", "")
            title = str(st.get("title") or "").strip()
            instr = str(st.get("instruction") or "").strip()
            lines.append(f"{sn}. {title}\n   {instr}" if title else f"{sn}. {instr}")

    if internal_links:
        lines.append("\nASSETS — Internal link hints")
        for link in internal_links:
            lines.append(
                f"- From/section: {link.get('from_or_section', '')} → URL: {link.get('to_url', '')} "
                f"(anchor hint: {link.get('anchor_hint', '')})"
            )

    if affected_keywords:
        lines.append("\nDATA — Affected / cluster keywords")
        lines.append(", ".join(_gp_trunc(k, 80) for k in affected_keywords))

    if target_keywords:
        lines.append("\nDATA — Target keywords")
        lines.append(", ".join(_gp_trunc(k, 80) for k in target_keywords))

    ce = (copy_example or "").strip()
    if ce:
        lines.append("\nASSETS — Example / long-form copy snippet")
        lines.append(ce)

    return "\n".join(lines)


def parse_actions_landing_page_request_extras(data: Any) -> dict[str, Any]:
    """Normalize optional structured fields from the generate-page-preview POST body."""
    raw: dict[str, Any] = dict(data) if isinstance(data, Mapping) else {}
    return {
        "action_faqs": _gp_parse_faqs(raw.get("action_faqs")),
        "trust_block": _gp_trunc(str(raw.get("trust_block") or ""), 900),
        "quick_facts": _gp_as_str_list(raw.get("quick_facts"), 10),
        "plan_steps": _gp_parse_plan_steps(raw.get("plan_steps")),
        "internal_links": _gp_parse_internal_links(raw.get("internal_links")),
        "affected_keywords": _gp_as_str_list(raw.get("affected_keywords"), 12, 120),
        "target_keywords": _gp_as_str_list(raw.get("target_keywords"), 12, 120),
        "copy_example": _gp_trunc(str(raw.get("copy_example") or ""), 900),
    }


def generate_structured_landing_page_preview(
    *,
    keyword: str,
    business_name: str,
    location: str,
    service_area: str,
    seo_issues: list[str],
    page_type: str | None = None,
    business_profile: BusinessProfile | None = None,
    action_faqs: list[dict[str, str]] | None = None,
    trust_block: str = "",
    quick_facts: list[str] | None = None,
    plan_steps: list[dict[str, Any]] | None = None,
    internal_links: list[dict[str, str]] | None = None,
    affected_keywords: list[str] | None = None,
    target_keywords: list[str] | None = None,
    copy_example: str = "",
) -> dict[str, Any]:
    """
    Ask OpenAI for a strict JSON document describing a landing page as a tree of UI components.

    Prompts favor natural business copy and forbid audit/schema/ranking meta language; internal
    SEO issue lines are framing only. Returns a dict with top-level key ``page`` (title, slug, components).
    """
    system = (
        "You are an expert UX copywriter and landing page architect. Generate a high-converting, "
        "natural-sounding local service landing page as STRICT JSON ONLY (no HTML, no markdown, no prose outside JSON).\n\n"
        "Write like a real business website—not an SEO tool, audit report, or AI system. Hide all search/optimization mechanics; "
        "never explain why content exists.\n\n"
        "FORBIDDEN anywhere in visitor-facing strings: mentions of schema, JSON-LD, structured data, SEO implementation, "
        "ranking signals, meta commentary about search engines or AI, or artificial section titles such as "
        "\"Trust & Verification\", \"Business Credentials\", \"SEO Schema Implementation\", \"Ranking Signals\", "
        "or \"Certified Local Provider\" unless explicitly provided in inputs. Do not fabricate ratings, review counts, "
        "or certifications. Do not use \"verified\" or \"licensed team\" unless explicitly in the input assets.\n\n"
        "Trust: weave naturally into sentences (e.g. serving a neighborhood), not as a labeled trust stack. "
        "Avoid repetitive filler (\"top rated\", \"established\", \"trusted\") unless grounded in provided data.\n\n"
        "PAGE FLOW (use sections to separate meaning, not SEO categories):\n"
        "1) Hero: clear human headline, short supporting paragraph, primary CTA button.\n"
        "2) Intro: what the visitor is looking for, location woven in naturally.\n"
        "3) Main value: what the business offers; benefits in plain language.\n"
        "4) Location / service area: cities or neighborhoods in a paragraph or list.\n"
        "5) Social proof: real reviews only if provided; otherwise generic placeholder like "
        "\"Customers often highlight quality and speed of service.\"—no fake numbers.\n"
        "6) Optional details: pricing/hours/services only if provided in inputs.\n"
        "7) FAQ (accordion only): real customer questions—availability, pricing, location, how it works—not audit questions.\n"
        "8) Final CTA: simple, action-oriented, not a repeat of the hero.\n\n"
        "INTERNAL inputs may include topic gaps or FAQs—use them to structure copy only; never expose them as diagnostics.\n\n"
        "Allowed component ``type`` values: h1, h2, h3, h4, h5, h6, paragraph, div, section, button, "
        "list, table, accordion, columns.\n\n"
        "For ``list``, use ``items`` as an array of strings; optional ``ordered`` boolean for numbered list.\n"
        "For ``columns``, set ``columns`` to 2 or 3, and ``children`` as an array of columns where each "
        "column is an array of component objects.\n"
        "For ``accordion``, use ``items`` as an array of objects with ``title`` and ``content`` strings.\n"
        "For ``table``, use ``headers`` (string array) and ``rows`` (array of string arrays).\n"
        "For ``button``, use ``content`` and optional ``url``.\n"
        "For headings and ``paragraph``, use ``content``.\n"
        "For ``section`` and ``div``, use ``children`` for nested components.\n\n"
        "Output shape (exact top-level keys): "
        '{"page":{"title":"string","slug":"string","components":[...]}}\n'
        "Do NOT include null values. Do not include trailing commas. Output must parse as JSON."
    )
    faqs = action_faqs or []
    qf = quick_facts or []
    steps = plan_steps or []
    ilinks = internal_links or []
    aff = affected_keywords or []
    tgt = target_keywords or []
    context_block = _build_actions_landing_page_context_block(
        seo_issues=seo_issues or [],
        action_faqs=faqs,
        trust_block=trust_block,
        quick_facts=qf,
        plan_steps=steps,
        internal_links=ilinks,
        affected_keywords=aff,
        target_keywords=tgt,
        copy_example=copy_example,
    )
    pt = (page_type or "").strip()
    page_type_line = f"Page type: {pt}\n" if pt else ""
    user_msg = (
        "INPUTS (use for copy only; the published page must read like a normal business site):\n\n"
        f"Primary keyword / topic: {keyword}\n"
        f"Business name: {business_name}\n"
        f"Location: {location}\n"
        f"Service area: {service_area}\n"
        f"{page_type_line}"
        f"{context_block}\n\n"
        "Produce one landing page that converts visitors into actions. Use the keyword and location naturally—no stuffing. "
        "If internal topic lines conflict with provided FAQs or business facts, prefer the explicit FAQs and facts.\n\n"
        "Return ONLY valid JSON with this exact shape (no other keys at the top level):\n"
        '{"page":{"title":"string","slug":"string","components":[]}}'
    )

    client = _get_client("OPEN_AI_SEO_API_KEY")
    model = _get_model()
    completion = chat_completion_create_logged(
        client,
        operation="openai.chat.actions_landing_page_preview",
        business_profile=business_profile,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    raw = _strip_json_fence(completion.choices[0].message.content or "")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Model returned non-object JSON")
    page = data.get("page")
    if not isinstance(page, dict):
        raise ValueError("Model JSON must include a page object")
    comps = page.get("components")
    if not isinstance(comps, list):
        page["components"] = []
    return data


def seo_chat(request: HttpRequest) -> Response:
    """
    Core implementation of the SEO agent chat endpoint backed by OpenAI.

    This is called by the Django view wrapper in accounts.views.
    """
    data = request.data
    message = (data.get("message") or "").strip()
    if not message:
        return Response({"detail": "Message is required."}, status=400)

    conversation_id = data.get("conversation_id")

    # Get or create conversation for this user & agent "seo"
    conversation: AgentConversation | None = None
    if conversation_id:
        try:
            conversation = AgentConversation.objects.get(
                id=conversation_id,
                user=request.user,
                agent="seo",
            )
        except AgentConversation.DoesNotExist:
            conversation = None

    if not conversation:
        conversation = AgentConversation.objects.create(
            user=request.user,
            agent="seo",
            title="SEO Agent Chat",
        )

    # Store the user message
    AgentMessage.objects.create(
        conversation=conversation,
        role="user",
        content=message,
    )

    # Build message history (last N messages) for context
    recent_messages = list(
        conversation.messages.order_by("-created_at")[:20],
    )
    recent_messages.reverse()  # oldest → newest

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )

    # Attach live, cached SEO score + core metrics so the agent can reason with them.
    seo_score_block = ""
    try:
        seo_score_data = get_or_refresh_seo_score_for_user(
            request.user,
            site_url=profile.website_url if profile and profile.website_url else None,
            business_profile=profile,
        )
        if seo_score_data is not None:
            seo_score_block = (
                "\n\nCurrent SEO metrics (from DataForSEO, cached hourly): "
                f"Overall SEO score: {seo_score_data['seo_score']}/100. "
                f"Visibility index: {seo_score_data['organic_visitors']} (mapped from search volume). "
                f"Ranking keywords: {seo_score_data['keywords_ranking']}. "
                f"Top 3 positions: {seo_score_data['top3_positions']}. "
                "Use these numbers when assessing SEO health, prioritizing work, and explaining tradeoffs."
            )
    except Exception:
        # Never break chat if the scoring helper fails; just omit the block.
        seo_score_block = ""

    system_prompt = build_seo_system_prompt(request.user, profile) + seo_score_block
    assistant_reply = get_seo_chat_reply(
        system_prompt,
        recent_messages,
        conversation_summary=conversation.summary or None,
        business_profile=profile,
    )

    # Store assistant reply
    AgentMessage.objects.create(
        conversation=conversation,
        role="assistant",
        content=assistant_reply,
    )

    # Periodic summarization to keep history manageable
    total_messages = conversation.messages.count()
    if total_messages > 40:
        summary_messages = list(
            conversation.messages.order_by("created_at")[:80],
        )
        conversation.summary = summarize_seo_conversation(
            summary_messages,
            business_profile=profile,
        )
        conversation.save(update_fields=["summary", "updated_at"])

    return Response(
        {
            "conversation_id": conversation.id,
            "reply": assistant_reply,
        },
    )


