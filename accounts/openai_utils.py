"""
OpenAI-backed helpers for agent chat and summarization.

Keeps all OpenAI client usage and message formatting in one place.
"""

import json
import os

from django.conf import settings
from django.http import HttpRequest

from openai import OpenAI
from rest_framework.response import Response

from .models import (
    AgentConversation,
    AgentMessage,
    BusinessProfile,
    ReviewsConversation,
    ReviewsMessage,
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
        if profile.tone_of_voice:
            details.append(f"Tone of voice: {profile.tone_of_voice}.")
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
    return getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")


def _get_chat_reply(
    system_prompt: str,
    recent_messages: list,
    conversation_summary: str | None = None,
    api_key_env: str | None = None,
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
    completion = client.chat.completions.create(
        model=model,
        messages=openai_messages,
    )
    return (completion.choices[0].message.content or "").strip()


def get_seo_chat_reply(
    system_prompt: str,
    recent_messages: list[AgentMessage],
    conversation_summary: str | None = None,
) -> str:
    """Call OpenAI for SEO agent using OPEN_AI_SEO_API_KEY."""
    return _get_chat_reply(
        system_prompt,
        recent_messages,
        conversation_summary,
        api_key_env="OPEN_AI_SEO_API_KEY",
    )


def summarize_seo_conversation(messages: list[AgentMessage]) -> str:
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
    completion = client.chat.completions.create(
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
        completion = client.chat.completions.create(
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


def generate_seo_next_steps(seo_data: dict) -> list[dict]:
    """
    Ask OpenAI for 6 next steps to improve SEO, given current SEO metrics.
    seo_data should include: seo_score, missed_searches_monthly, organic_visitors,
    total_search_volume, search_visibility_percent, top_keywords (list of {keyword, search_volume, rank}),
    and optionally onpage_issue_summaries.
    Returns list of {"label": str, "tag": str} (length 6). Tags are short, e.g. "Quick win", "High priority".
    """
    if not seo_data:
        return []

    score = seo_data.get("seo_score")
    missed = seo_data.get("missed_searches_monthly")
    organic = seo_data.get("organic_visitors")
    total_vol = seo_data.get("total_search_volume")
    visibility_pct = seo_data.get("search_visibility_percent")
    keywords = seo_data.get("top_keywords") or []
    onpage_summaries = seo_data.get("onpage_issue_summaries") or {}

    keyword_preview = ""
    if keywords:
        parts = [f"{k.get('keyword', '')} ({k.get('search_volume', 0)}/mo)" for k in keywords[:12]]
        keyword_preview = "\n".join(parts)

    system = (
        "You are an SEO expert. Given the site's current SEO data, output exactly 6 actionable next steps "
        "to improve their SEO score. Each step must have: \"label\" (one short sentence, actionable, e.g. "
        "'Rewrite your primary conversion page headline and CTA') and \"tag\" (2–3 words, e.g. 'Quick win', "
        "'High priority', 'This week', 'Cross-channel'). Be specific to the data provided (e.g. mention "
        "missed searches, visibility, or top keywords where relevant). Output only a JSON array of 6 objects "
        "with keys \"label\" and \"tag\". No other text."
    )
    user_parts = [
        f"SEO score (0–100): {score}",
        f"Monthly missed searches (not finding this site): {missed}",
        f"Organic visitors (estimated): {organic}",
        f"Total search volume (keyword set): {total_vol}",
        f"Search visibility %: {visibility_pct}%",
    ]
    if keyword_preview:
        user_parts.append("Top keywords (keyword, volume/mo):\n" + keyword_preview)
    if onpage_summaries:
        user_parts.append("On-page/technical issues: " + str(onpage_summaries))
    user_content = "Current SEO data:\n\n" + "\n".join(user_parts) + "\n\nReturn a JSON array of 6 next steps with \"label\" and \"tag\"."

    try:
        client = _get_client("OPEN_AI_SEO_API_KEY")
        model = _get_model()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown code block if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rstrip("`").strip()
        steps = json.loads(raw)
        if not isinstance(steps, list):
            return []
        out = []
        for i, item in enumerate(steps[:6]):
            if not isinstance(item, dict):
                continue
            label = (item.get("label") or "").strip()
            tag = (item.get("tag") or "Next step").strip()[:40]
            if label:
                out.append({"label": label, "tag": tag or "Next step"})
        return out[:6]
    except Exception:
        return []


def generate_keyword_action_suggestions(keywords: list[dict]) -> list[dict]:
    """
    Given a list of keyword dicts (from \"what people search for\" / top_keywords),
    ask OpenAI to propose one-sentence SEO improvement actions per keyword.

    Returns a list of {\"keyword\": str, \"suggestion\": str}.
    """
    if not keywords:
        return []

    # Limit to a reasonable number of keywords to keep the prompt small.
    top = []
    for k in keywords:
        kw = (k.get("keyword") or "").strip()
        if not kw:
            continue
        vol = int(k.get("search_volume") or 0)
        top.append((kw, vol))
    if not top:
        return []
    # Sort by volume descending and cap to top 20 for the prompt.
    top_sorted = sorted(top, key=lambda kv: kv[1], reverse=True)[:20]

    lines = [f"- {kw} ({vol}/mo)" for kw, vol in top_sorted]
    user_content = (
        "For each of the following search keyword phrases, suggest exactly ONE concise, "
        "one-sentence action the business owner should take to improve their SEO for that keyword.\n\n"
        "List of keywords with approximate monthly search volume:\n"
        + "\n".join(lines)
        + "\n\n"
        "Return a JSON array where each item has:\n"
        "- \"keyword\": the original keyword phrase string\n"
        "- \"suggestion\": one clear, actionable sentence describing what to do next.\n"
        "Do not include any other fields or text."
    )

    system = (
        "You are an SEO strategist. Given keyword phrases and their search volumes, "
        "you propose highly practical, concrete actions the business can take to rank higher "
        "for each keyword. Suggestions should be short (one sentence) and easy to do."
    )

    try:
        client = _get_client("OPEN_AI_SEO_API_KEY")
        model = _get_model()
        completion = client.chat.completions.create(
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
        if not isinstance(data, list):
            return []
        out: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            kw = (item.get("keyword") or "").strip()
            suggestion = (item.get("suggestion") or "").strip()
            if kw and suggestion:
                out.append({"keyword": kw, "suggestion": suggestion})
        return out
    except Exception:
        return []


def summarize_reviews_conversation(messages: list[ReviewsMessage]) -> str:
    """
    Ask OpenAI to summarize a Reviews conversation into concise memory notes.
    """
    payload: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Summarize the following Reviews/Reputation conversation into concise memory notes. "
                "Capture key goals, tone preferences, and decisions. 5-10 bullet points max."
            ),
        },
    ]
    for m in messages:
        payload.append({"role": m.role, "content": m.content})

    client = _get_client("OPEN_AI_REVIEWS_API_KEY")
    model = _get_model()
    completion = client.chat.completions.create(
        model=model,
        messages=payload,
    )
    return (completion.choices[0].message.content or "").strip()


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

    profile = BusinessProfile.objects.filter(user=request.user).first()

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
        conversation.summary = summarize_seo_conversation(summary_messages)
        conversation.save(update_fields=["summary", "updated_at"])

    return Response(
        {
            "conversation_id": conversation.id,
            "reply": assistant_reply,
        },
    )


def reviews_chat(request: HttpRequest) -> Response:
    """
    Core implementation of the Reviews Agent chat endpoint. Same pattern as SEO chat
    but uses ReviewsConversation and ReviewsMessage (separate tables) and a different system role.
    """
    data = request.data
    message = (data.get("message") or "").strip()
    if not message:
        return Response({"detail": "Message is required."}, status=400)

    conversation_id = data.get("conversation_id")

    conversation: ReviewsConversation | None = None
    if conversation_id:
        try:
            conversation = ReviewsConversation.objects.get(
                id=conversation_id,
                user=request.user,
            )
        except ReviewsConversation.DoesNotExist:
            conversation = None

    if not conversation:
        conversation = ReviewsConversation.objects.create(
            user=request.user,
            title="Reviews Agent Chat",
        )

    ReviewsMessage.objects.create(
        conversation=conversation,
        role="user",
        content=message,
    )

    recent_messages = list(
        conversation.messages.order_by("-created_at")[:20],
    )
    recent_messages.reverse()

    profile = BusinessProfile.objects.filter(user=request.user).first()
    system_prompt = build_reviews_system_prompt(request.user, profile)
    assistant_reply = _get_chat_reply(
        system_prompt,
        recent_messages,
        conversation_summary=conversation.summary or None,
        api_key_env="OPEN_AI_REVIEWS_API_KEY",
    )

    ReviewsMessage.objects.create(
        conversation=conversation,
        role="assistant",
        content=assistant_reply,
    )

    total_messages = conversation.messages.count()
    if total_messages > 40:
        summary_messages = list(
            conversation.messages.order_by("created_at")[:80],
        )
        conversation.summary = summarize_reviews_conversation(summary_messages)
        conversation.save(update_fields=["summary", "updated_at"])

    return Response(
        {
            "conversation_id": conversation.id,
            "reply": assistant_reply,
        },
    )
