"""
Google Business Profile (My Business) API client for Reviews Agent.
Fetches aggregate star rating (average across all reviews), total reviews, and
live response rate (reviews with owner reply / total) for the connected account/location.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from django.conf import settings

logger = logging.getLogger(__name__)

# Star rating enum from API: ONE, TWO, THREE, FOUR, FIVE (STAR_RATING_UNSPECIFIED excluded)
STAR_MAP = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5}
VALID_STAR_RATINGS = frozenset(STAR_MAP)


def _get_gbp_access_token(user) -> str | None:
    """Return a valid access token for the user's GBP connection, refreshing if needed."""
    from .models import GoogleBusinessProfileConnection

    try:
        conn = GoogleBusinessProfileConnection.objects.get(user=user)
    except GoogleBusinessProfileConnection.DoesNotExist:
        logger.info("[GBP] _get_gbp_access_token user_id=%s: no GoogleBusinessProfileConnection", user.id)
        return None

    now = datetime.now(timezone.utc)
    if conn.expires_at and conn.expires_at > now + timedelta(seconds=60) and conn.access_token:
        logger.info("[GBP] _get_gbp_access_token user_id=%s: using existing access token", user.id)
        return conn.access_token

    if not (conn.refresh_token or "").strip():
        logger.warning(
            "[GBP] _get_gbp_access_token user_id=%s: connection exists but refresh_token is empty",
            user.id,
        )
        return conn.access_token or None

    import requests

    token_data = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "refresh_token": conn.refresh_token,
        "grant_type": "refresh_token",
    }
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data=token_data,
        timeout=10,
    )
    if resp.status_code != 200:
        logger.warning("[GBP] _get_gbp_access_token user_id=%s token refresh failed status=%s", user.id, resp.status_code)
        return conn.access_token or None

    data = resp.json()
    access_token = data.get("access_token")
    expires_in = data.get("expires_in")
    if not access_token:
        logger.warning("[GBP] _get_gbp_access_token user_id=%s refresh response had no access_token", user.id)
        return conn.access_token or None

    logger.info("[GBP] _get_gbp_access_token user_id=%s refreshed access token", user.id)
    conn.access_token = access_token
    if isinstance(expires_in, int):
        conn.expires_at = now + timedelta(seconds=expires_in)
    conn.save(update_fields=["access_token", "expires_at"])
    return access_token


def fetch_gbp_overview(user) -> dict | None:
    """
    Fetch overview stats from Google Business Profile (accounts -> locations -> reviews).
    - Star rating: aggregate average of all reviews with a valid rating (from live API).
    - Response rate: live data (reviews with owner reply / total reviews).
    Returns dict with star_rating, previous_star_rating, total_reviews, new_reviews_this_month,
    response_rate_pct, industry_avg_response_pct, requests_sent, conversion_pct, or None on failure.
    """
    import requests

    access_token = _get_gbp_access_token(user)
    if not access_token:
        logger.warning("[GBP] No access token for user_id=%s (missing connection or refresh failed)", user.id)
        return None

    logger.info("[GBP] user_id=%s has access token, calling Account Management API", user.id)
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    # 1) List accounts (Account Management API)
    try:
        r = requests.get(
            "https://mybusinessaccountmanagement.googleapis.com/v1/accounts",
            headers=headers,
            timeout=10,
        )
        if r.status_code != 200:
            logger.warning(
                "[GBP] Accounts list failed user_id=%s status=%s body=%s",
                user.id,
                r.status_code,
                r.text[:500],
            )
            return None
        data = r.json()
        accounts = data.get("accounts") or []
        logger.info("[GBP] user_id=%s accounts response: count=%s", user.id, len(accounts))
        if not accounts:
            logger.warning("[GBP] No accounts found for user_id=%s raw_keys=%s", user.id, list(data.keys()))
            return None
        account_name = accounts[0].get("name")  # e.g. "accounts/123456789"
        if not account_name or not account_name.startswith("accounts/"):
            logger.warning("[GBP] Invalid first account name for user_id=%s: %s", user.id, account_name)
            return None
        account_id = account_name.split("/")[-1]
        logger.info("[GBP] user_id=%s using account_name=%s account_id=%s", user.id, account_name, account_id)
    except Exception as e:
        logger.exception("[GBP] Error listing accounts user_id=%s: %s", user.id, e)
        return None

    # 2) List locations (Business Information API)
    try:
        locations_url = f"https://mybusinessbusinessinformation.googleapis.com/v1/{account_name}/locations"
        r = requests.get(locations_url, headers=headers, timeout=10)
        if r.status_code != 200:
            logger.warning(
                "[GBP] Locations list failed user_id=%s status=%s body=%s",
                user.id,
                r.status_code,
                r.text[:500],
            )
            return None
        data = r.json()
        locations = data.get("locations") or []
        logger.info("[GBP] user_id=%s locations response: count=%s", user.id, len(locations))
        if not locations:
            logger.warning("[GBP] No locations for user_id=%s account_id=%s raw_keys=%s", user.id, account_id, list(data.keys()))
            return None
        location = locations[0]
        location_name = location.get("name")  # e.g. "accounts/123/locations/456"
        if not location_name:
            logger.warning("[GBP] First location has no name user_id=%s", user.id)
            return None
        parts = location_name.split("/")
        location_id = parts[-1] if len(parts) >= 4 else None
        if not location_id:
            logger.warning("[GBP] Could not parse location_id from %s user_id=%s", location_name, user.id)
            return None
        logger.info("[GBP] user_id=%s using location_name=%s location_id=%s", user.id, location_name, location_id)
    except Exception as e:
        logger.exception("[GBP] Error listing locations user_id=%s: %s", user.id, e)
        return None

    # 3) List reviews (My Business API v4)
    try:
        all_reviews = []
        page_token = None
        for page_num in range(20):  # cap pages
            url = (
                f"https://mybusiness.googleapis.com/v4/accounts/{account_id}/locations/{location_id}/reviews"
                f"?pageSize=50"
            )
            if page_token:
                url += f"&pageToken={page_token}"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                logger.warning(
                    "[GBP] Reviews list failed user_id=%s status=%s body=%s",
                    user.id,
                    r.status_code,
                    r.text[:500],
                )
                break
            data = r.json()
            reviews = data.get("reviews") or []
            all_reviews.extend(reviews)
            logger.info(
                "[GBP] user_id=%s reviews page=%s page_size=%s total_so_far=%s",
                user.id,
                page_num + 1,
                len(reviews),
                len(all_reviews),
            )
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        logger.exception("[GBP] Error listing reviews user_id=%s: %s", user.id, e)
        return None

    logger.info("[GBP] user_id=%s location_id=%s total_reviews=%s", user.id, location_id, len(all_reviews))

    if not all_reviews:
        logger.info("[GBP] No reviews for user_id=%s location_id=%s; returning zeros", user.id, location_id)
        # Still return a valid overview with zeros
        return _build_overview_from_reviews([], user)

    result = _build_overview_from_reviews(all_reviews, user)
    logger.info(
        "[GBP] user_id=%s overview result: total_reviews=%s star_rating=%s response_rate_pct=%s",
        user.id,
        result.get("total_reviews"),
        result.get("star_rating"),
        result.get("response_rate_pct"),
    )
    return result


def _build_overview_from_reviews(reviews: list, user) -> dict:
    """
    Compute overview stats from live review list from the Business Profile API.
    - Star rating: aggregate average over all reviews that have a valid starRating (ONE–FIVE).
    - Response rate: live count of reviews that have an owner reply (reviewReply.comment) / total.
    """
    from .models import ReviewsOverviewSnapshot

    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total = len(reviews)
    if total == 0:
        snapshot, _ = ReviewsOverviewSnapshot.objects.get_or_create(user=user)
        # Store zeros so the 1-hour cache is updated and we don't hit the API again.
        snapshot.star_rating = Decimal("0")
        snapshot.previous_star_rating = snapshot.previous_star_rating or Decimal("0")
        snapshot.total_reviews = 0
        snapshot.new_reviews_this_month = 0
        snapshot.response_rate_pct = Decimal("0")
        snapshot.industry_avg_response_pct = snapshot.industry_avg_response_pct or Decimal("45")
        snapshot.requests_sent = snapshot.requests_sent or 0
        snapshot.conversion_pct = snapshot.conversion_pct or Decimal("0")
        snapshot.save()
        return {
            "star_rating": 0.0,
            "previous_star_rating": float(snapshot.previous_star_rating or 0),
            "total_reviews": 0,
            "new_reviews_this_month": 0,
            "response_rate_pct": 0.0,
            "industry_avg_response_pct": float(snapshot.industry_avg_response_pct or 45),
            "requests_sent": snapshot.requests_sent or 0,
            "conversion_pct": float(snapshot.conversion_pct or 0),
        }

    # Aggregate star rating: average over all reviews with a valid rating (not a single review)
    sum_stars = 0
    count_rated = 0
    # Live response rate: count reviews that have an owner reply
    responded = 0
    new_this_month = 0
    for rev in reviews:
        star = rev.get("starRating")
        if star and star in VALID_STAR_RATINGS:
            sum_stars += STAR_MAP[star]
            count_rated += 1
        if rev.get("reviewReply") and rev["reviewReply"].get("comment"):
            responded += 1
        ct = rev.get("createTime")
        if ct:
            try:
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= start_of_month:
                    new_this_month += 1
            except (ValueError, TypeError):
                pass

    # Aggregate rating: only from reviews that have a star rating (avoid diluting with unspecified)
    avg_rating = round(sum_stars / count_rated, 2) if count_rated else 0
    # Live response rate from API data (reviews with reply / total reviews)
    response_rate = round((responded / total) * 100, 2) if total else 0

    snapshot, _ = ReviewsOverviewSnapshot.objects.get_or_create(user=user)
    prev_rating = snapshot.star_rating
    snapshot.previous_star_rating = prev_rating or Decimal(str(avg_rating))
    snapshot.star_rating = Decimal(str(avg_rating))
    snapshot.total_reviews = total
    snapshot.new_reviews_this_month = new_this_month
    snapshot.response_rate_pct = Decimal(str(response_rate))
    snapshot.industry_avg_response_pct = snapshot.industry_avg_response_pct or Decimal("45")
    # Leave requests_sent and conversion_pct as-is unless we have a source (e.g. CRM or internal)
    if snapshot.requests_sent == 0 and total > 0:
        snapshot.requests_sent = total * 2  # placeholder
        snapshot.conversion_pct = Decimal("16.8")
    snapshot.save()

    return {
        "star_rating": avg_rating,
        "previous_star_rating": float(prev_rating or avg_rating),
        "total_reviews": total,
        "new_reviews_this_month": new_this_month,
        "response_rate_pct": response_rate,
        "industry_avg_response_pct": float(snapshot.industry_avg_response_pct or 45),
        "requests_sent": snapshot.requests_sent or 0,
        "conversion_pct": float(snapshot.conversion_pct or 0),
    }
