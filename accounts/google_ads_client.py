from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from django.conf import settings

from .models import GoogleAdsKeywordIdea


@dataclass
class KeywordIdea:
    keyword: str
    avg_monthly_searches: int | None
    competition: str | None
    competition_index: int | None
    low_top_of_page_bid_micros: int | None
    high_top_of_page_bid_micros: int | None


def _has_ads_credentials() -> bool:
    required = [
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "GOOGLE_ADS_CUSTOMER_ID",
        "GOOGLE_ADS_REFRESH_TOKEN",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
    ]
    return all(getattr(settings, name, None) for name in required)


def classify_intent(keyword: str) -> str:
    """
    Simple rule-based intent classification.
    Does not rely on Google Ads-provided intent.
    """
    k = keyword.lower()
    high_triggers = [
        "buy",
        "price",
        "cost",
        "near me",
        "coupon",
        "deal",
        "best",
        "hire",
        "book",
        "quote",
        "service",
    ]
    low_triggers = [
        "what is",
        "definition",
        "how to",
        "tutorial",
        "examples",
        "guide",
        "meaning",
    ]

    if any(t in k for t in high_triggers):
        return "HIGH"
    if any(t in k for t in low_triggers):
        return "LOW"
    return "MEDIUM"


def fetch_keyword_ideas_for_user(
    user_id: int,
    keywords: Iterable[str],
    cache_ttl_days: int = 7,
    industry: str | None = None,
    description: str | None = None,
) -> dict[str, KeywordIdea]:
    """
    Fetch keyword ideas from Google Ads KeywordPlanIdeaService for the given user & keywords.

    Results are cached per user/keyword in GoogleAdsKeywordIdea to reduce API calls.
    If Google Ads credentials are missing, this returns only cached data.
    """
    from google.ads.googleads.client import GoogleAdsClient  # type: ignore[import]

    now = datetime.now(timezone.utc)

    # First, load cached ideas.
    cached: dict[str, KeywordIdea] = {}
    fresh_cutoff = now - timedelta(days=cache_ttl_days)

    for idea in GoogleAdsKeywordIdea.objects.filter(
        user_id=user_id, last_fetched_at__gte=fresh_cutoff
    ):
        cached[idea.keyword] = KeywordIdea(
            keyword=idea.keyword,
            avg_monthly_searches=idea.avg_monthly_searches,
            competition=idea.competition,
            competition_index=idea.competition_index,
            low_top_of_page_bid_micros=idea.low_top_of_page_bid_micros,
            high_top_of_page_bid_micros=idea.high_top_of_page_bid_micros,
        )

    remaining = [k for k in keywords if k not in cached]

    # Use business profile context (industry / description) as additional seeds
    # to help Google Ads suggest better ideas, without changing intent logic.
    extra_seeds: list[str] = []
    if industry:
        extra_seeds.append(industry)
    if description:
        # Take a short prefix of the description as a seed (avoid flooding the request).
        snippet = description.strip()
        if len(snippet) > 80:
            snippet = snippet[:80]
        if snippet:
            extra_seeds.append(snippet)

    if (not remaining and not extra_seeds) or not _has_ads_credentials():
        return cached

    # Always use the company's global Google Ads credentials, not per-user tokens.
    refresh_token = getattr(settings, "GOOGLE_ADS_REFRESH_TOKEN", None)
    customer_id = getattr(settings, "GOOGLE_ADS_CUSTOMER_ID", "")
    if not refresh_token or not customer_id:
        return cached

    # Build Google Ads client from company-level settings.
    # Note: login_customer_id can match customer_id since we only use a single Ads account.
    config = {
        "developer_token": settings.GOOGLE_ADS_DEVELOPER_TOKEN,
        "login_customer_id": customer_id,
        "client_customer_id": customer_id,
        "use_proto_plus": True,
        "refresh_token": refresh_token,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
    }
    client = GoogleAdsClient.load_from_dict(config)
    service = client.get_service("KeywordPlanIdeaService")

    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = customer_id.replace("-", "")
    request.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
    # Always include the remaining keywords; optionally add extra business-context seeds.
    if remaining:
        request.keyword_seed.keywords.extend(remaining)
    if extra_seeds:
        request.keyword_seed.keywords.extend(extra_seeds)

    try:
        response = service.generate_keyword_ideas(request=request)
    except Exception:
        # On any error, just return cached data and avoid breaking the flow.
        return cached

    for idea in response:
        text = idea.text
        if text not in remaining:
            continue

        metrics = idea.keyword_idea_metrics
        avg_monthly_searches = (
            int(metrics.avg_monthly_searches) if metrics.avg_monthly_searches else None
        )
        competition_enum = metrics.competition.name if metrics.competition else None
        competition_index = (
            int(metrics.competition_index) if metrics.competition_index else None
        )
        low_bid = (
            int(metrics.low_top_of_page_bid_micros)
            if metrics.low_top_of_page_bid_micros
            else None
        )
        high_bid = (
            int(metrics.high_top_of_page_bid_micros)
            if metrics.high_top_of_page_bid_micros
            else None
        )

        obj, _ = GoogleAdsKeywordIdea.objects.update_or_create(
            user_id=user_id,
            keyword=text,
            defaults={
                "avg_monthly_searches": avg_monthly_searches or 0,
                "competition": competition_enum or "",
                "competition_index": competition_index or 0,
                "low_top_of_page_bid_micros": low_bid or 0,
                "high_top_of_page_bid_micros": high_bid or 0,
                "last_fetched_at": now,
            },
        )

        cached[text] = KeywordIdea(
            keyword=text,
            avg_monthly_searches=avg_monthly_searches,
            competition=competition_enum,
            competition_index=competition_index,
            low_top_of_page_bid_micros=low_bid,
            high_top_of_page_bid_micros=high_bid,
        )

    return cached

