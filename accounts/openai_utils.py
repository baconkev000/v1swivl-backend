"""
OpenAI-backed helpers for agent chat and summarization.

Keeps all OpenAI client usage and message formatting in one place.
"""

import json
import os
import re
from typing import Any

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
    Produce up to eight structured next steps grounded in ``build_structured_issues`` evidence, reframed
    for answer engines (AI assistants, Perplexity-style answers, AI Overviews): citeable blocks,
    entity clarity, and page-level actions—not generic “rank higher” advice.

    ``seo_data`` should include scores, ``top_keywords``, and optionally ``on_page`` / ``serp`` for the
    deterministic issue engine. When ``snapshot`` is set, merges crawl-derived FAQ signals and
    persists structured issues on the snapshot.

    Returns the same dict shape as before (``title``, ``why_it_matters``, ``exact_fix``, ``example``,
    plus ``label`` / ``tag`` for backward compatibility).
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
    "You are generating ACTION-CARD DATA for a frontend Actions page.\n"
    "Focus on clarity, scanability, and implementation detail for SEO/AEO improvements that improve AI-answer inclusion "
    "and citation likelihood in assistants (Google SGE/AI Overviews, ChatGPT, Perplexity).\n\n"
    "Use wording that makes cited/cite outcomes explicit when supported by evidence.\n"
    "INPUT JSON contains:\n"
    "- actions[] (prioritized deterministic fixes)\n"
    "- keywords[] (keyword-level context)\n"
    "- optional snapshot context\n\n"
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
    "\"goal\":\"...\","
    "\"whats_missing\":[\"...\"],"
    "\"steps\":[{\"step_number\":1,\"title\":\"...\",\"instruction\":\"...\"}],"
    "\"copy_paste_content\":{\"local_trust_block\":\"...\",\"faq\":[{\"q\":\"...\",\"a\":\"...\"}],\"quick_facts\":[\"...\"]},"
    "\"schema_requirements\":[\"...\"],"
    "\"internal_linking\":[{\"from_or_section\":\"...\",\"to_url\":\"...\",\"anchor_hint\":\"...\"}],"
    "\"target_url\":\"...\","
    "\"ai_optimization_notes\":{\"why_it_helps\":[\"...\"],\"expected_impact\":[\"...\"]},"
    "\"evidence\":{\"keyword\":\"...\",\"search_volume\":0,\"rank\":0,\"competitor_rank\":0,\"competitor_domains\":[\"...\"],\"location\":\"...\",\"source_issue_ids\":[\"...\"]},"
    "\"display_hints\":{\"expanded_by_default\":false,\"show_copy_paste_section\":true,\"show_schema_section\":true,\"show_internal_linking_section\":true}"
    "}]"
    "}\n"
    "If you cannot build the full card, return legacy compact JSON with keys title, why_it_matters, exact_fix, example.\n\n"
    "STRICT RULES:\n"
    "- Do NOT invent keywords, ranks, URLs, search volumes, competitor domains, location, or issue IDs.\n"
    "- Deduplicate overlapping recommendations and keep the richer merged result.\n"
    "- No vague directives like 'optimize SEO' or 'improve content' without exact sections/actions.\n"
    "- Steps must be imperative and implementation-ready.\n"
    "- Use empty strings/arrays instead of guessing unknown values.\n"
)

_SEO_VAGUE_PATTERN = re.compile(
    r"\b(optimi[sz]e\s+seo|improve\s+seo\b|enhance\s+seo\b|increase\s+visibility|boost\s+rankings?|rank\s+higher|"
    r"get\s+more\s+traffic|drive\s+more\s+clicks|climb\s+the\s+serp)\b",
    re.IGNORECASE,
)
_SEO_ACTION_PATTERN = re.compile(
    r"\b(create|add|publish|rewrite|expand|implement|build|update|link|schema|section|page|heading|h1|h2|faq|"
    r"table|comparison|entity|snippet|paragraph|list|bullet|quote|citat|mentionable|answer block|internal\s+link|"
    r"subhead|structured\s+data|json-ld|proof|testimonial|q&a|question)\b",
    re.IGNORECASE,
)
# Vague “AI visibility” claims without concrete edits (must still match _SEO_ACTION_PATTERN somewhere in exact_fix).
_AEO_VAGUE_PATTERN = re.compile(
    r"\b(ai\s+visibility|be\s+seen\s+in\s+ai|win\s+at\s+aeo|dominate\s+ai)\b",
    re.IGNORECASE,
)
_BANNED_GENERIC_WORDS = re.compile(r"\b(optimi[sz]e|improve|enhance)\b", re.IGNORECASE)


def _seo_rewrite_output_is_valid(payload: dict, recommendation: dict) -> bool:
    title = str(payload.get("title") or "").strip()
    why = str(payload.get("why_it_matters") or "").strip()
    fix = str(payload.get("exact_fix") or "").strip()
    if not title or not why or not fix:
        return False
    if _SEO_VAGUE_PATTERN.search(fix) and not _SEO_ACTION_PATTERN.search(fix):
        return False
    if _AEO_VAGUE_PATTERN.search(fix) and not _SEO_ACTION_PATTERN.search(fix):
        return False
    if _BANNED_GENERIC_WORDS.search(fix) and not _SEO_ACTION_PATTERN.search(fix):
        return False
    if len(fix) < 20:
        return False
    evidence = recommendation.get("evidence") or {}
    keyword = str(evidence.get("primary_keyword") or evidence.get("keyword") or "").strip().lower()
    url = str(evidence.get("url") or evidence.get("your_url") or "").strip().lower()
    # When we have concrete context, ensure rewritten output keeps at least one anchor.
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
        return {
            "title": str(data.get("title") or "").strip(),
            "why_it_matters": str(data.get("why_it_matters") or "").strip(),
            "exact_fix": str(data.get("exact_fix") or "").strip(),
            "example": str(data.get("example") or "").strip(),
        }

    cards = data.get("actions")
    if isinstance(cards, list) and cards and isinstance(cards[0], dict):
        first_card = cards[0]
        title = str(first_card.get("title") or "").strip()
        goal = str(first_card.get("goal") or "").strip()
        missing = [str(x).strip() for x in list(first_card.get("whats_missing") or []) if str(x).strip()]
        why = " ".join([x for x in [goal, ". ".join(missing[:2])] if x]).strip()

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

        return {
            "title": title,
            "why_it_matters": why,
            "exact_fix": exact_fix,
            "example": example,
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

    return {
        "title": title,
        "why_it_matters": why,
        "exact_fix": exact_fix,
        "example": example,
    }


def _rewrite_structured_seo_recommendation(
    recommendation: dict,
    *,
    snapshot_context: dict | None = None,
) -> dict:
    """
    AI rewrite layer for deterministic SEO issue recommendations, reframed for AI answer / citation use
    without changing structured keys (title, why_it_matters, exact_fix, example).
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


