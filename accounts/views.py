import secrets
from urllib.parse import urlencode, unquote, quote
from datetime import datetime, date, timedelta, timezone

import requests
from django.conf import settings
from django.contrib.auth import get_user_model, login, logout as django_logout
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authentication import SessionAuthentication
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import (
    BusinessProfile,
    GoogleSearchConsoleConnection,
    SEOOverviewSnapshot,
    AgentConversation,
    AgentMessage,
)
from .serializers import BusinessProfileSerializer
from .google_ads_client import classify_intent, fetch_keyword_ideas_for_user
from . import openai_utils


User = get_user_model()


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Use Django session auth without enforcing CSRF.

    This is safe here because the API is only accessible to already
    authenticated users and is called via our own frontend.
    """

    def enforce_csrf(self, request):
        return  # Skip the CSRF check performed by SessionAuthentication.


def health_check(_: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok"})


def google_login(request: HttpRequest) -> HttpResponse:
    state = secrets.token_urlsafe(32)
    frontend_base = getattr(settings, "FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")

    # Allow the frontend to pass either an absolute next URL or a relative path.
    raw_next = request.GET.get("next")
    next_url: str
    if raw_next:
        decoded_next = unquote(raw_next)
        if decoded_next.startswith("http://") or decoded_next.startswith("https://"):
            next_url = decoded_next
        else:
            # Treat as path, always send the user back to the frontend domain.
            if not decoded_next.startswith("/"):
                decoded_next = "/" + decoded_next
            next_url = frontend_base + decoded_next
    else:
        # Default after Google login
        next_url = frontend_base + "/app"

    request.session["oauth_state"] = state
    request.session["oauth_next"] = next_url

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": settings.GOOGLE_OAUTH_SCOPE,
        "access_type": "offline",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "consent",
    }

    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return redirect(auth_url)


def google_callback(request: HttpRequest) -> HttpResponse:
    stored_state = request.session.get("oauth_state")
    frontend_base = getattr(settings, "FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")
    session_next = request.session.get("oauth_next")
    if session_next:
        decoded_next = unquote(str(session_next))
        if decoded_next.startswith("http://") or decoded_next.startswith("https://"):
            next_url = decoded_next
        else:
            if not decoded_next.startswith("/"):
                decoded_next = "/" + decoded_next
            next_url = frontend_base + decoded_next
    else:
        next_url = frontend_base + "/app"

    incoming_state = request.GET.get("state")
    if not stored_state or not incoming_state or stored_state != incoming_state:
        return HttpResponseBadRequest("Invalid OAuth state")

    code = request.GET.get("code")
    if not code:
        return HttpResponseBadRequest("Missing authorization code")

    token_data = {
        "code": code,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data=token_data,
        timeout=10,
    )
    if token_resp.status_code != 200:
        return HttpResponseBadRequest("Failed to exchange code for token")

    token_json = token_resp.json()
    access_token = token_json.get("access_token")
    if not access_token:
        return HttpResponseBadRequest("No access token received")

    userinfo_resp = requests.get(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if userinfo_resp.status_code != 200:
        return HttpResponseBadRequest("Failed to fetch user info")

    profile = userinfo_resp.json()
    email = profile.get("email")
    name = profile.get("name", "") or ""

    if not email:
        return HttpResponseBadRequest("Email is required from Google")

    first_name = profile.get("given_name") or ""
    last_name = profile.get("family_name") or ""

    user, _ = User.objects.get_or_create(
        username=email,
        defaults={
            "email": email,
            "first_name": first_name or name,
            "last_name": last_name,
        },
    )

    if user.email != email:
        user.email = email
        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        user.save(update_fields=["email", "first_name", "last_name"])

    login(request, user)

    return redirect(next_url)


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def gsc_status(request: HttpRequest) -> Response:
    """
    Return whether the current user has a Google Search Console connection.
    """
    connected = GoogleSearchConsoleConnection.objects.filter(user=request.user).exists()
    return Response({"connected": connected})


def gsc_connect_start(request: HttpRequest) -> HttpResponse:
    """
    Start Google OAuth flow for Search Console read-only access.
    """
    if not request.user.is_authenticated:
        # Rely on frontend middleware to enforce auth, but guard anyway.
        return redirect(settings.FRONTEND_BASE_URL + "/login")

    state = secrets.token_urlsafe(32)
    request.session["gsc_state"] = state
    next_url = request.GET.get("next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"
    request.session["gsc_next"] = next_url

    redirect_uri = request.build_absolute_uri("/integrations/google-search-console/callback/")
    scope = "https://www.googleapis.com/auth/webmasters.readonly"

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "access_type": "offline",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "consent",
    }

    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return redirect(auth_url)


def gsc_connect_callback(request: HttpRequest) -> HttpResponse:
    """
    Handle the Google OAuth callback for Search Console access and persist tokens.
    """
    state = request.GET.get("state")
    stored_state = request.session.get("gsc_state")
    next_url = request.session.get("gsc_next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"

    if not stored_state or state != stored_state:
        return HttpResponseBadRequest("Invalid OAuth state")

    code = request.GET.get("code")
    if not code:
        return HttpResponseBadRequest("Missing authorization code")

    redirect_uri = request.build_absolute_uri("/integrations/google-search-console/callback/")

    token_data = {
        "code": code,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data=token_data,
        timeout=10,
    )
    if token_resp.status_code != 200:
        return HttpResponseBadRequest("Failed to exchange code for token")

    token_json = token_resp.json()
    access_token = token_json.get("access_token", "")
    refresh_token = token_json.get("refresh_token", "")
    token_type = token_json.get("token_type", "")
    expires_in = token_json.get("expires_in")

    expires_at: datetime | None = None
    if isinstance(expires_in, int):
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    if not request.user.is_authenticated:
        # User should already be authenticated via Google SSO.
        return redirect(settings.FRONTEND_BASE_URL + "/login")

    conn, _created = GoogleSearchConsoleConnection.objects.get_or_create(user=request.user)
    conn.access_token = access_token
    if refresh_token:
        conn.refresh_token = refresh_token
    conn.token_type = token_type
    conn.expires_at = expires_at
    conn.save()

    # Clean up session keys
    request.session.pop("gsc_state", None)
    request.session.pop("gsc_next", None)

    return redirect(next_url)


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def ads_status(request: HttpRequest) -> Response:
    """
    Return whether the current user has a Google Ads connection.
    """
    from .models import GoogleAdsConnection

    connected = GoogleAdsConnection.objects.filter(
        user=request.user,
        refresh_token__isnull=False,
        customer_id__isnull=False,
    ).exclude(refresh_token="").exclude(customer_id="").exists()
    return Response({"connected": connected})


def ads_connect_start(request: HttpRequest) -> HttpResponse:
    """
    Start Google OAuth flow for Google Ads API access.
    """
    if not request.user.is_authenticated:
        return redirect(settings.FRONTEND_BASE_URL + "/login")

    state = secrets.token_urlsafe(32)
    request.session["gads_state"] = state
    next_url = request.GET.get("next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"
    request.session["gads_next"] = next_url

    redirect_uri = request.build_absolute_uri("/integrations/google-ads/callback/")
    scope = "https://www.googleapis.com/auth/adwords"

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "access_type": "offline",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "consent",
    }

    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return redirect(auth_url)


def ads_connect_callback(request: HttpRequest) -> HttpResponse:
    """
    Handle the Google OAuth callback for Google Ads access and persist tokens.
    """
    from .models import GoogleAdsConnection

    state = request.GET.get("state")
    stored_state = request.session.get("gads_state")
    next_url = request.session.get("gads_next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"

    if not stored_state or state != stored_state:
        return HttpResponseBadRequest("Invalid OAuth state")

    code = request.GET.get("code")
    if not code:
        return HttpResponseBadRequest("Missing authorization code")

    redirect_uri = request.build_absolute_uri("/integrations/google-ads/callback/")

    token_data = {
        "code": code,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data=token_data,
        timeout=10,
    )
    if token_resp.status_code != 200:
        return HttpResponseBadRequest("Failed to exchange code for token")

    token_json = token_resp.json()
    access_token = token_json.get("access_token", "")
    refresh_token = token_json.get("refresh_token", "")
    token_type = token_json.get("token_type", "")
    expires_in = token_json.get("expires_in")

    expires_at: datetime | None = None
    if isinstance(expires_in, int):
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    if not request.user.is_authenticated:
        return redirect(settings.FRONTEND_BASE_URL + "/login")

    conn, _created = GoogleAdsConnection.objects.get_or_create(user=request.user)
    conn.access_token = access_token
    if refresh_token:
        conn.refresh_token = refresh_token
    conn.token_type = token_type
    conn.expires_at = expires_at
    conn.save()

    # Try to determine the user's Ads customer ID using the newly created connection.
    try:
        from google.ads.googleads.client import GoogleAdsClient  # type: ignore[import]

        if conn.refresh_token:
            ads_config = {
                "developer_token": settings.GOOGLE_ADS_DEVELOPER_TOKEN,
                "refresh_token": conn.refresh_token,
                "client_id": settings.GOOGLE_ADS_CLIENT_ID,
                "client_secret": settings.GOOGLE_ADS_CLIENT_SECRET,
                "use_proto_plus": True,
            }
            ads_client = GoogleAdsClient.load_from_dict(ads_config)
            customer_service = ads_client.get_service("CustomerService")
            accessible = customer_service.list_accessible_customers()
            # Pick the first accessible customer as the default for now.
            if accessible.resource_names:
                resource_name = accessible.resource_names[0]  # "customers/1234567890"
                customer_id = resource_name.split("/")[-1]
                conn.customer_id = customer_id
                conn.save(update_fields=["customer_id"])
    except Exception:
        # If this fails, the integration still works using app-level customer ID fallback.
        pass

    request.session.pop("gads_state", None)
    request.session.pop("gads_next", None)

    return redirect(next_url)


def _get_gsc_access_token(user: User) -> str | None:
    """
    Return a fresh access token for the user's Google Search Console connection,
    refreshing it with the stored refresh token if needed.
    """
    try:
        conn = GoogleSearchConsoleConnection.objects.get(user=user)
    except GoogleSearchConsoleConnection.DoesNotExist:
        return None

    # If token is still valid (with small safety window), reuse it.
    now = datetime.now(timezone.utc)
    if conn.expires_at and conn.expires_at > now + timedelta(seconds=60) and conn.access_token:
        return conn.access_token

    if not conn.refresh_token:
        return conn.access_token or None

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
        return conn.access_token or None

    data = resp.json()
    access_token = data.get("access_token")
    expires_in = data.get("expires_in")
    if not access_token:
        return conn.access_token or None

    conn.access_token = access_token
    if isinstance(expires_in, int):
        conn.expires_at = now + timedelta(seconds=expires_in)
    conn.save(update_fields=["access_token", "expires_at"])
    return access_token


def _gsc_query(
    access_token: str,
    site_url: str,
    start: date,
    end: date,
) -> list[dict]:
    """
    Call the Search Console searchAnalytics.query endpoint for the given site and date range.
    """
    endpoint = (
        "https://searchconsole.googleapis.com/webmasters/v3/sites/"
        f"{quote(site_url, safe='')}/searchAnalytics/query"
    )
    body = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "dimensions": ["query"],
        "rowLimit": 25000,
    }
    resp = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        json=body,
        timeout=20,
    )
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data.get("rows", []) or []


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_overview(request: HttpRequest) -> Response:
    """
    Return SEO overview metrics for the dashboard, powered by Google Search Console:

    - Organic Visitors This Month – clicks this month
    - Keywords Ranking – count of distinct queries this month
    - Top 3 Positions – count of queries with average position <= 3 this month
    - Organic Growth – % change in clicks vs previous month
    """
    # Ensure we have a site URL for Search Console.
    profile = BusinessProfile.objects.filter(user=request.user).first()
    site_url = (profile.website_url if profile else None) or getattr(
        settings,
        "GOOGLE_SITE_URL",
        "",
    )
    if not site_url:
        return Response(
            {"detail": "No site URL configured for Search Console."},
            status=400,
        )

    access_token = _get_gsc_access_token(request.user)
    if not access_token:
        return Response(
            {"detail": "Google Search Console is not connected."},
            status=400,
        )

    today = datetime.now(timezone.utc).date()
    start_current = today.replace(day=1)
    # Previous month range
    if start_current.month == 1:
        prev_year = start_current.year - 1
        prev_month = 12
    else:
        prev_year = start_current.year
        prev_month = start_current.month - 1
    start_prev = date(prev_year, prev_month, 1)
    end_prev = start_current - timedelta(days=1)

    rows_current = _gsc_query(access_token, site_url, start_current, today)
    rows_prev = _gsc_query(access_token, site_url, start_prev, end_prev)

    organic_visitors = int(sum(float(r.get("clicks", 0)) for r in rows_current))
    keywords_ranking = len(rows_current)
    top3_positions = sum(1 for r in rows_current if float(r.get("position", 9999)) <= 3.0)

    prev_clicks = int(sum(float(r.get("clicks", 0)) for r in rows_prev))
    if prev_clicks == 0:
        organic_growth_pct = 100.0 if organic_visitors > 0 else 0.0
    else:
        organic_growth_pct = ((organic_visitors - prev_clicks) / prev_clicks) * 100.0

    # Persist snapshot for this user and period.
    snapshot, _ = SEOOverviewSnapshot.objects.get_or_create(
        user=request.user,
        period_start=start_current,
    )
    snapshot.organic_visitors = organic_visitors
    snapshot.prev_organic_visitors = prev_clicks
    snapshot.keywords_ranking = keywords_ranking
    snapshot.top3_positions = top3_positions
    snapshot.save()

    return Response(
        {
            "organic_visitors": organic_visitors,
            "keywords_ranking": keywords_ranking,
            "top3_positions": top3_positions,
            "organic_growth_pct": organic_growth_pct,
        },
    )


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_keywords(request: HttpRequest) -> Response:
    """
    Return a unified High-Intent Keywords dataset combining:

    - Google Search Console: query, clicks, impressions, ctr, position
    - Google Ads KeywordPlanIdeaService: avg_monthly_searches, competition, competition_index, bid range
    - Rule-based intent classification (HIGH / MEDIUM / LOW)
    """
    profile = BusinessProfile.objects.filter(user=request.user).first()
    site_url = (profile.website_url if profile else None) or getattr(
        settings,
        "GOOGLE_SITE_URL",
        "",
    )

    today = datetime.now(timezone.utc).date()

    # If GSC is connected, use it to get per-site ranking & position deltas.
    access_token = _get_gsc_access_token(request.user)
    results: list[dict] = []

    if site_url and access_token:
        # Current period: last 30 days
        start = today - timedelta(days=30)
        prev_start = start - timedelta(days=30)
        prev_end = start - timedelta(days=1)

        rows_current = _gsc_query(access_token, site_url, start, today)
        rows_prev = _gsc_query(access_token, site_url, prev_start, prev_end)

        # Index previous-period positions by query for quick lookup
        prev_positions: dict[str, float] = {}
        for row in rows_prev:
            keys = row.get("keys") or []
            if not keys:
                continue
            q = keys[0]
            prev_positions[q] = float(row.get("position", 0))

        # Sort by clicks descending and take top N.
        sorted_rows = sorted(
            rows_current,
            key=lambda r: float(r.get("clicks", 0)),
            reverse=True,
        )
        top_rows = sorted_rows[:50]

        keywords = [r["keys"][0] for r in top_rows if r.get("keys")]

        # Fetch Ads ideas with caching, using business context (industry / description)
        # to improve suggestions.
        industry = profile.industry if profile and profile.industry else None
        description = profile.description if profile and profile.description else None

        ads_ideas = {}
        try:
            ads_ideas = fetch_keyword_ideas_for_user(
                request.user.id,
                keywords,
                industry=industry,
                description=description,
            )
        except Exception:
            # If Ads auth is missing or fails, we still return GSC-only data.
            ads_ideas = {}

        for row in top_rows:
            query_keys = row.get("keys") or []
            if not query_keys:
                continue
            query = query_keys[0]

            ads = ads_ideas.get(query)

            clicks = float(row.get("clicks", 0))
            impressions = float(row.get("impressions", 0))
            ctr = float(row.get("ctr", 0))
            position = float(row.get("position", 0))

            prev_pos = prev_positions.get(query)
            # Positive delta means improvement (moved up).
            position_change = None
            if prev_pos is not None:
                position_change = prev_pos - position

            results.append(
                {
                    "keyword": query,
                    "avg_monthly_searches": (
                        int(ads.avg_monthly_searches)
                        if ads and ads.avg_monthly_searches is not None
                        else None
                    ),
                    "intent": classify_intent(query),
                    "current_position": position,
                    "position_change": position_change,
                    "impressions": int(impressions),
                    "clicks": int(clicks),
                    "ctr": ctr,
                },
            )

    else:
        # No GSC connection: fall back to any cached Ads ideas for this user.
        # These provide global-ish volume and competition, but no site-specific ranking.
        from .models import GoogleAdsKeywordIdea

        for idea in GoogleAdsKeywordIdea.objects.filter(user=request.user).order_by(
            "-avg_monthly_searches",
        )[:50]:
            results.append(
                {
                    "keyword": idea.keyword,
                    "avg_monthly_searches": idea.avg_monthly_searches,
                    "intent": classify_intent(idea.keyword),
                    "current_position": None,
                    "position_change": None,
                    "impressions": 0,
                    "clicks": 0,
                    "ctr": 0.0,
                },
            )

    return Response({"keywords": results})


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_chat(request: HttpRequest) -> Response:
    """SEO agent chat endpoint – delegates to OpenAI utils implementation."""
    return openai_utils.seo_chat(request)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def me(request: HttpRequest) -> JsonResponse:
    user = request.user
    return JsonResponse(
        {
            "id": user.id,
            "email": user.email,
            "first_name": getattr(user, "first_name", ""),
            "last_name": getattr(user, "last_name", ""),
        }
    )


@csrf_exempt
@api_view(["GET", "PATCH", "PUT"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def business_profile(request: HttpRequest) -> Response:
    """
    Retrieve or upsert the authenticated user's business profile.

    - GET: returns the current profile (creates an empty one if missing).
    - PATCH/PUT: updates existing profile fields; creates a profile if it does not exist yet.
    """
    profile, _created = BusinessProfile.objects.get_or_create(user=request.user)

    if request.method == "GET":
        serializer = BusinessProfileSerializer(profile)
        return Response(serializer.data)

    # For PATCH/PUT, apply partial updates
    serializer = BusinessProfileSerializer(
        profile,
        data=request.data,
        partial=True,
    )
    serializer.is_valid(raise_exception=True)
    serializer.save()
    return Response(serializer.data)


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def api_logout(request: HttpRequest) -> Response:
    """
    Log out the current user from the Django session (Google SSO).
    """
    django_logout(request)
    return Response({"success": True})

