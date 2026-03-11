import logging
import secrets
from urllib.parse import urlencode, unquote, quote, urlparse
from datetime import datetime, date, timedelta, timezone

import requests
from django.conf import settings
from django.core.cache import cache
from django.contrib.auth import get_user_model, login, logout as django_logout
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authentication import SessionAuthentication
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import (
    AgentActivityLog,
    BusinessProfile,
    GoogleAdsMetricsCache,
    GoogleSearchConsoleConnection,
    GoogleBusinessProfileConnection,
    MetaAdsConnection,
    SEOOverviewSnapshot,
    ReviewsOverviewSnapshot,
    AgentConversation,
    AgentMessage,
)

# Third-party API cache: only refetch from GSC, Google Ads, GBP if last fetch was >= this long ago.
THIRD_PARTY_CACHE_TTL = timedelta(hours=1)
# Google My Business Account Management API: 1 request per minute per project. Enforce per-user to avoid 429.
GBP_API_MIN_INTERVAL_SECONDS = 60
from .serializers import BusinessProfileSerializer
from .google_ads_client import (
    classify_intent,
    fetch_ads_metrics_for_user,
    fetch_ads_metrics_for_user_result,
)
from .meta_ads_utils import get_meta_ads_status_for_user
from .tiktok_ads_utils import get_tiktok_ads_status_for_user
from .dataforseo_utils import (
    get_keyword_gap_keywords,
    get_ranked_keywords_visibility,
)
from . import openai_utils
from . import debug_log as _debug

logger = logging.getLogger(__name__)
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
def gbp_status(request: HttpRequest) -> Response:
    """
    Return whether the current user has a Google Business Profile connection.
    """
    connected = GoogleBusinessProfileConnection.objects.filter(
        user=request.user,
    ).exclude(refresh_token="").exists()
    return Response({"connected": bool(connected)})


def gbp_connect_start(request: HttpRequest) -> HttpResponse:
    """
    Start Google OAuth flow for Google Business Profile (reviews, locations).
    """
    if not request.user.is_authenticated:
        return redirect(settings.FRONTEND_BASE_URL + "/login")

    state = secrets.token_urlsafe(32)
    request.session["gbp_state"] = state
    next_url = request.GET.get("next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"
    request.session["gbp_next"] = next_url

    redirect_uri = request.build_absolute_uri("/integrations/google-business-profile/callback/")
    scope = "https://www.googleapis.com/auth/business.manage"

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


def gbp_connect_callback(request: HttpRequest) -> HttpResponse:
    """
    Handle the Google OAuth callback for Google Business Profile and persist tokens.
    """
    state = request.GET.get("state")
    stored_state = request.session.get("gbp_state")
    next_url = request.session.get("gbp_next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"

    if not stored_state or state != stored_state:
        return HttpResponseBadRequest("Invalid OAuth state")

    code = request.GET.get("code")
    if not code:
        return HttpResponseBadRequest("Missing authorization code")

    redirect_uri = request.build_absolute_uri("/integrations/google-business-profile/callback/")

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

    expires_at = None
    if isinstance(expires_in, int):
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    if not request.user.is_authenticated:
        return redirect(settings.FRONTEND_BASE_URL + "/login")

    conn, _created = GoogleBusinessProfileConnection.objects.get_or_create(user=request.user)
    conn.access_token = access_token
    if refresh_token:
        conn.refresh_token = refresh_token
    conn.token_type = token_type
    conn.expires_at = expires_at
    conn.save()

    request.session.pop("gbp_state", None)
    request.session.pop("gbp_next", None)
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


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def meta_ads_status(request: HttpRequest) -> Response:
    """
    Return whether the current user has a Meta Ads connection.

    The underlying lookup lives in ``meta_ads_utils.get_meta_ads_status_for_user``
    so that the storage / API details can evolve independently of this view.
    """

    status = get_meta_ads_status_for_user(request.user.id)
    return Response({"connected": bool(status.connected)})


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def tiktok_ads_status(request: HttpRequest) -> Response:
    """
    Return whether the current user has a TikTok Ads connection.

    The underlying lookup lives in ``tiktok_ads_utils.get_tiktok_ads_status_for_user``
    so that implementation details are kept out of the HTTP layer.
    """

    status = get_tiktok_ads_status_for_user(request.user.id)
    return Response({"connected": bool(status.connected)})


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def agent_activity_feed(request: HttpRequest) -> Response:
    """
    Return the current user's agent activity log for the dashboard "What your agents did today".
    Only returns records from the last 30 days (same window as cleanup).
    """
    from django.utils import timezone
    from datetime import timedelta

    cutoff = timezone.now() - timedelta(days=30)
    logs = (
        AgentActivityLog.objects.filter(user=request.user, created_at__gte=cutoff)
        .order_by("-created_at")[:100]
    )
    return Response({
        "activities": [
            {
                "id": log.id,
                "agent": log.agent,
                "description": log.description,
                "account_name": log.account_name or "",
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
    })


# Meta (Facebook) Marketing API OAuth: scopes for campaigns, ad sets, ads, creatives, audiences, insights, pages.
META_ADS_OAUTH_SCOPES = [
    "ads_management",       # create and manage campaigns, ad sets, ads
    "ads_read",             # read ad accounts and reports
    "business_management",  # manage business assets and ad accounts
    "pages_show_list",      # list pages (for Page Posts / Page Insights)
    "pages_read_engagement",  # read page insights
    "pages_manage_posts",   # optionally post content to pages
]


def meta_ads_connect_start(request: HttpRequest) -> HttpResponse:
    """
    Start Meta (Facebook) OAuth flow for Marketing API access.

    Requires META_ADS_APP_ID and META_ADS_APP_SECRET. Redirect URI must be whitelisted in the Meta app
    (e.g. https://yourdomain.com/integrations/meta-ads/callback/).
    """
    next_url = request.GET.get("next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"
    frontend_base = (settings.FRONTEND_BASE_URL or "http://localhost:3000").rstrip("/")

    if not request.user.is_authenticated:
        # Send user to frontend login; after login they can hit Connect again with a session.
        login_url = f"{frontend_base}/login?next={quote(next_url, safe='')}"
        return redirect(login_url)

    app_id = getattr(settings, "META_ADS_APP_ID", None)
    app_secret = getattr(settings, "META_ADS_APP_SECRET", None)
    if not app_id or not (app_secret and str(app_secret).strip()):
        logger.warning("[Meta Ads] META_ADS_APP_ID or META_ADS_APP_SECRET not set; skipping OAuth redirect.")
        sep = "&" if "?" in next_url else "?"
        return redirect(f"{next_url}{sep}meta_ads_error=not_configured")

    state = secrets.token_urlsafe(32)
    request.session["meta_ads_state"] = state
    request.session["meta_ads_next"] = next_url

    redirect_uri = request.build_absolute_uri("/integrations/meta-ads/callback/")
    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": ",".join(META_ADS_OAUTH_SCOPES),
        "response_type": "code",
    }

    auth_url = "https://www.facebook.com/v18.0/dialog/oauth?" + urlencode(params)
    return redirect(auth_url)


def meta_ads_connect_callback(request: HttpRequest) -> HttpResponse:
    """
    Handle Meta OAuth callback: exchange code for short-lived token, then for
    long-lived token (60 days), and store in MetaAdsConnection.
    """
    stored_state = request.session.get("meta_ads_state")
    next_url = request.session.get("meta_ads_next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"
    request.session.pop("meta_ads_state", None)
    request.session.pop("meta_ads_next", None)

    state = request.GET.get("state")
    if stored_state and state != stored_state:
        return HttpResponseBadRequest("Invalid OAuth state")

    code = request.GET.get("code")
    if not code:
        error = request.GET.get("error_description") or request.GET.get("error", "Missing authorization code")
        logger.warning("[Meta Ads] OAuth callback error: %s", error)
        return redirect(next_url)

    app_id = getattr(settings, "META_ADS_APP_ID", None)
    app_secret = getattr(settings, "META_ADS_APP_SECRET", None)
    if not app_id or not app_secret:
        return redirect(next_url)

    redirect_uri = request.build_absolute_uri("/integrations/meta-ads/callback/")

    # Exchange code for short-lived access token
    token_url = "https://graph.facebook.com/v18.0/oauth/access_token"
    token_params = {
        "client_id": app_id,
        "client_secret": app_secret,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    token_resp = requests.get(token_url, params=token_params, timeout=15)
    if token_resp.status_code != 200:
        logger.warning("[Meta Ads] Token exchange failed: %s %s", token_resp.status_code, token_resp.text[:300])
        return redirect(next_url)

    try:
        short_data = token_resp.json()
    except ValueError:
        logger.warning("[Meta Ads] Token response not JSON: %s", token_resp.text[:200])
        return redirect(next_url)

    short_token = short_data.get("access_token")
    if not short_token:
        logger.warning("[Meta Ads] No access_token in response: %s", short_data)
        return redirect(next_url)

    # Exchange short-lived for long-lived token (60 days)
    long_params = {
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": short_token,
    }
    long_resp = requests.get(token_url, params=long_params, timeout=15)
    if long_resp.status_code != 200:
        logger.warning("[Meta Ads] Long-lived token exchange failed: %s", long_resp.status_code)
        # Still save short-lived token so connection works until it expires
        access_token = short_token
        expires_in = short_data.get("expires_in", 3600)
    else:
        try:
            long_data = long_resp.json()
            access_token = long_data.get("access_token") or short_token
            expires_in = long_data.get("expires_in", 5184000)  # 60 days in seconds
        except ValueError:
            access_token = short_token
            expires_in = short_data.get("expires_in", 3600)

    expires_at = None
    if isinstance(expires_in, int) and expires_in > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    if not request.user.is_authenticated:
        return redirect(settings.FRONTEND_BASE_URL + "/login")

    conn, _ = MetaAdsConnection.objects.get_or_create(user=request.user)
    conn.access_token = access_token
    conn.token_type = "Bearer"
    conn.expires_at = expires_at
    conn.save(update_fields=["access_token", "token_type", "expires_at", "updated_at"])

    logger.info("[Meta Ads] Connection saved for user_id=%s, expires_at=%s", request.user.id, expires_at)
    return redirect(next_url)


def tiktok_ads_connect_start(request: HttpRequest) -> HttpResponse:
    """
    Start TikTok Ads OAuth flow. Redirects to next URL until TikTok OAuth is configured.
    """
    if not request.user.is_authenticated:
        return redirect(settings.FRONTEND_BASE_URL + "/login")
    next_url = request.GET.get("next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"
    # TODO: when TikTok OAuth is configured, redirect to TikTok and set session state/callback
    return redirect(next_url)


def tiktok_ads_connect_callback(request: HttpRequest) -> HttpResponse:
    """
    Handle TikTok OAuth callback. Redirects to next URL until TikTok OAuth is implemented.
    """
    next_url = request.session.get("tiktok_ads_next") or settings.FRONTEND_BASE_URL + "/app?tab=integrations"
    request.session.pop("tiktok_ads_state", None)
    request.session.pop("tiktok_ads_next", None)
    return redirect(next_url)


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def ads_metrics(request: HttpRequest) -> Response:
    """
    Return Google Ads performance metrics for the current user.
    Uses a 1-hour cache: if we have fresh cached metrics, return them without calling the Google Ads API.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - THIRD_PARTY_CACHE_TTL
    force_refresh = request.GET.get("refresh") == "1"

    if not force_refresh:
        try:
            cache = GoogleAdsMetricsCache.objects.get(user=request.user)
            if cache.fetched_at >= cutoff:
                return Response({
                    "new_customers_this_month": cache.new_customers_this_month,
                    "new_customers_previous_month": cache.new_customers_previous_month,
                    "avg_roas": cache.avg_roas,
                    "google_search_roas": cache.google_search_roas,
                    "cost_per_customer": cache.cost_per_customer,
                    "cost_per_customer_previous": cache.cost_per_customer_previous,
                    "active_campaigns_count": cache.active_campaigns_count,
                })
        except GoogleAdsMetricsCache.DoesNotExist:
            pass

    metrics, reason, detail = fetch_ads_metrics_for_user_result(request.user.id)
    if metrics is None:
        return Response(
            {
                "error": detail or "Google Ads not connected or metrics unavailable",
                "reason": reason or "not_connected",
            },
            status=404,
        )

    cache, _ = GoogleAdsMetricsCache.objects.update_or_create(
        user=request.user,
        defaults={
            "new_customers_this_month": metrics.new_customers_this_month,
            "new_customers_previous_month": metrics.new_customers_previous_month,
            "avg_roas": metrics.avg_roas,
            "google_search_roas": metrics.google_search_roas,
            "cost_per_customer": metrics.cost_per_customer,
            "cost_per_customer_previous": metrics.cost_per_customer_previous,
            "active_campaigns_count": metrics.active_campaigns_count,
        },
    )
    return Response({
        "new_customers_this_month": cache.new_customers_this_month,
        "new_customers_previous_month": cache.new_customers_previous_month,
        "avg_roas": cache.avg_roas,
        "google_search_roas": cache.google_search_roas,
        "cost_per_customer": cache.cost_per_customer,
        "cost_per_customer_previous": cache.cost_per_customer_previous,
        "active_campaigns_count": cache.active_campaigns_count,
    })


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
    # Keep refresh token from user auth only; never overwrite with empty (Google often
    # omits refresh_token on re-auth, so preserve the existing one).
    if refresh_token:
        conn.refresh_token = refresh_token
    conn.token_type = token_type
    conn.expires_at = expires_at
    conn.save()

    # Try to determine the user's Ads customer ID using the newly created connection.
    connection_ok = True
    connection_error_reason = None
    connection_error_detail = None
    try:
        from google.ads.googleads.client import GoogleAdsClient  # type: ignore[import]

        if conn.refresh_token:
            ads_config = {
                "developer_token": settings.GOOGLE_ADS_DEVELOPER_TOKEN,
                "refresh_token": conn.refresh_token,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "use_proto_plus": True,
            }
            ads_client = GoogleAdsClient.load_from_dict(ads_config)
            customer_service = ads_client.get_service("CustomerService")
            accessible = customer_service.list_accessible_customers()
            if accessible.resource_names:
                resource_name = accessible.resource_names[0]
                customer_id = resource_name.split("/")[-1]
                conn.customer_id = customer_id
                conn.save(update_fields=["customer_id"])
            else:
                connection_ok = False
                connection_error_reason = "no_accounts"
                connection_error_detail = (
                    "No Google Ads accounts were found for this login. "
                    "Create or get access to a Google Ads account at ads.google.com, then try connecting again."
                )
        else:
            connection_ok = False
            connection_error_reason = "missing_refresh_token"
            connection_error_detail = (
                "Google did not return a refresh token. Try disconnecting and connecting again, "
                "and make sure to approve all requested permissions."
            )
    except Exception as e:
        logger.exception("[Google Ads] list_accessible_customers failed: %s", e)
        connection_ok = False
        connection_error_reason = "api_error"
        connection_error_detail = (
            f"We couldn't load your Google Ads account: {str(e)}. "
            "Check that your Google account has access to Google Ads and try reconnecting."
        )

    request.session.pop("gads_state", None)
    request.session.pop("gads_next", None)

    if not connection_ok and connection_error_reason and connection_error_detail:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        parsed = urlparse(next_url)
        params = parse_qs(parsed.query)
        params["google_ads_error"] = [connection_error_reason]
        params["google_ads_error_detail"] = [connection_error_detail]
        new_query = urlencode(params, doseq=True)
        next_url = urlunparse(parsed._replace(query=new_query))

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
    Return SEO overview metrics for the dashboard, powered by DataForSEO Labs.

    We treat DataForSEO's \"visibility\" metric as our primary signal (mapped to
    organic_visitors for backwards compatibility with the frontend), and
    keywords_count / top3_positions as supporting metrics.

    Uses a 1-hour cache: if we have fresh snapshot data, return it without calling the API.
    """
    # #region agent log
    _debug.log("views.py:seo_overview:entry", "seo_overview called", {"user_id": getattr(request.user, "id", None), "query_refresh": request.GET.get("refresh")}, "H1")
    # #endregion
    today = datetime.now(timezone.utc).date()
    start_current = today.replace(day=1)
    now = datetime.now(timezone.utc)
    cutoff = now - THIRD_PARTY_CACHE_TTL
    cache_ttl = int(THIRD_PARTY_CACHE_TTL.total_seconds())
    force_refresh = request.GET.get("refresh") == "1"

    # Serve from cache if we have a snapshot for this period fetched within the last hour (unless refresh=1).
    if not force_refresh:
        try:
            snapshot = SEOOverviewSnapshot.objects.get(
                user=request.user,
                period_start=start_current,
            )
            if snapshot.last_fetched_at >= cutoff:
                prev_clicks = snapshot.prev_organic_visitors or 0
                organic_visitors = snapshot.organic_visitors or 0
                if prev_clicks == 0:
                    organic_growth_pct = 100.0 if organic_visitors > 0 else 0.0
                else:
                    organic_growth_pct = ((organic_visitors - prev_clicks) / prev_clicks) * 100.0
                # #region agent log
                _debug.log("views.py:seo_overview:cache_hit", "returning from snapshot cache", {"organic_visitors": organic_visitors, "last_fetched_at": str(snapshot.last_fetched_at), "cutoff": str(cutoff)}, "H2")
                # #endregion
                return Response(
                    {
                        "organic_visitors": organic_visitors,
                        "keywords_ranking": snapshot.keywords_ranking or 0,
                        "top3_positions": snapshot.top3_positions or 0,
                        "organic_growth_pct": organic_growth_pct,
                    },
                )
        except SEOOverviewSnapshot.DoesNotExist:
            pass

    # #region agent log
    _debug.log("views.py:seo_overview:cache_miss", "cache miss or force_refresh; calling DataForSEO", {}, "H2")
    # #endregion
    try:
        # Cache miss, stale, or refresh=1: call DataForSEO Labs ranked_keywords API.
        profile = BusinessProfile.objects.filter(user=request.user).first()
        site_url = profile.website_url if profile and profile.website_url else ""
        if not site_url:
            return Response(
                {"detail": "No website URL configured for SEO overview."},
                status=400,
            )

        parsed = urlparse(site_url)
        domain = (parsed.netloc or parsed.path or "").lower()
        if domain.startswith("www."):
            domain = domain[4:]

        if not domain:
            return Response(
                {"detail": "Could not determine domain from website URL."},
                status=400,
            )

        # If the website domain has changed since the last fetch, force a fresh
        # DataForSEO call instead of using cached snapshots.
        domain_cache_key = f"seo_overview_domain:{request.user.id}"
        previous_domain = cache.get(domain_cache_key)
        if previous_domain and previous_domain != domain:
            force_refresh = True
        # Store/update the current domain stamp for future comparisons.
        cache.set(domain_cache_key, domain, cache_ttl)

        location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))  # default: US

        logger.info(
            "[seo_overview] user_id=%s calling DataForSEO ranked_keywords for domain=%s location_code=%s",
            request.user.id,
            domain,
            location_code,
        )
        # #region agent log
        _debug.log("views.py:seo_overview:before_dataforseo", "domain and url sent", {"domain": domain, "site_url": site_url}, "H3")
        # #endregion
        visibility_data = get_ranked_keywords_visibility(
            domain,
            location_code=location_code,
            language_code=getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en"),
        )
        # #region agent log
        _debug.log("views.py:seo_overview:after_dataforseo", "visibility_data result", {"is_none": visibility_data is None, "has_keys": list(visibility_data.keys()) if visibility_data else []}, "H4")
        # #endregion
        # When DataForSEO returns no result (e.g. new/small domain with no ranking data yet),
        # return 200 with zeros so the dashboard still works instead of 502.
        if not visibility_data:
            # #region agent log
            _debug.log("views.py:seo_overview:returning_zeros", "returning 200 with zeros (fix branch)", {"domain": domain}, "H1")
            # #endregion
            logger.info(
                "[seo_overview] user_id=%s no DataForSEO result for domain=%s; returning zeros",
                request.user.id,
                domain,
            )
            try:
                snapshot = SEOOverviewSnapshot.objects.get(
                    user=request.user,
                    period_start=start_current,
                )
                prev_vis = snapshot.organic_visitors or 0
            except SEOOverviewSnapshot.DoesNotExist:
                prev_vis = 0
            organic_growth_pct = 0.0
            return Response(
                {
                    "organic_visitors": 0,
                    "keywords_ranking": 0,
                    "top3_positions": 0,
                    "organic_growth_pct": organic_growth_pct,
                },
            )

        current_visibility = int(round(visibility_data.get("visibility", 0.0) or 0.0))
        keywords_ranking = int(visibility_data.get("keywords_count", 0) or 0)
        top3_positions = int(visibility_data.get("top3_positions", 0) or 0)

        # Compute growth vs previous snapshot visibility (stored in organic_visitors).
        try:
            snapshot = SEOOverviewSnapshot.objects.get(
                user=request.user,
                period_start=start_current,
            )
            prev_vis = snapshot.organic_visitors or 0
        except SEOOverviewSnapshot.DoesNotExist:
            prev_vis = 0

        if prev_vis == 0:
            organic_growth_pct = 100.0 if current_visibility > 0 else 0.0
        else:
            organic_growth_pct = ((current_visibility - prev_vis) / prev_vis) * 100.0

        snapshot, _ = SEOOverviewSnapshot.objects.get_or_create(
            user=request.user,
            period_start=start_current,
        )
        snapshot.organic_visitors = current_visibility
        snapshot.prev_organic_visitors = prev_vis
        snapshot.keywords_ranking = keywords_ranking
        snapshot.top3_positions = top3_positions
        snapshot.save()

        return Response(
            {
                # For backwards compatibility with the frontend naming, map current visibility
                # to organic_visitors. The label in the UI can describe this as visibility.
                "organic_visitors": current_visibility,
                "keywords_ranking": keywords_ranking,
                "top3_positions": top3_positions,
                "organic_growth_pct": organic_growth_pct,
            },
        )
    except Exception as e:
        # #region agent log
        _debug.log("views.py:seo_overview:exception", "exception in seo_overview", {"exc_type": type(e).__name__, "exc_msg": str(e)[:300]}, "H4")
        # #endregion
        raise


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_keywords(request: HttpRequest) -> Response:
    """
    Return a High-Intent Keywords dataset for the SEO Agent using DataForSEO Labs
    domain_intersection to surface keyword gaps vs competitors.
    """
    force_refresh = request.GET.get("refresh") == "1"

    profile = BusinessProfile.objects.filter(user=request.user).first()
    site_url = profile.website_url if profile and profile.website_url else ""
    if not site_url:
        return Response({"keywords": []})

    parsed = urlparse(site_url)
    domain = (parsed.netloc or parsed.path or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]

    if not domain:
        return Response({"keywords": []})

    # Use per-user cache so we only call DataForSEO at most once per hour,
    # unless the caller explicitly asks for a refresh via ?refresh=1 or the
    # business website domain has changed.
    cache_key = f"seo_keywords:{request.user.id}"
    domain_cache_key = f"seo_keywords_domain:{request.user.id}"
    previous_domain = cache.get(domain_cache_key)
    if previous_domain and previous_domain != domain:
        force_refresh = True

    if not force_refresh:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(
                "[SEO keywords] Returning cached keyword gap items for user_id=%s (count=%s).",
                request.user.id,
                len(cached),
            )
            return Response({"keywords": cached})

    # Competitor domains can be configured globally for now; later we can make this
    # per-user if needed. Provide a sensible default set of common directories so
    # competitor analysis works even if the env var is not set.
    raw_competitors = getattr(
        settings,
        "DATAFORSEO_DEFAULT_COMPETITORS",
        "yelp.com,angi.com,thumbtack.com,homeadvisor.com",
    )
    competitor_domains = [c.strip() for c in raw_competitors.split(",") if c.strip()]

    location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    language_code = getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en")

    logger.info(
        "[SEO keywords] user_id=%s domain=%s competitors=%s location_code=%s",
        request.user.id,
        domain,
        ",".join(competitor_domains) or "(none)",
        location_code,
    )

    gap_items = get_keyword_gap_keywords(
        domain,
        competitor_domains,
        location_code=location_code,
        language_code=language_code,
    )

    results: list[dict] = []
    for item in gap_items:
        kw = item.get("keyword")
        if not kw:
            continue
        search_volume = item.get("search_volume")
        competitors = item.get("competitors") or []
        results.append(
            {
                "keyword": kw,
                "avg_monthly_searches": int(search_volume) if search_volume is not None else None,
                "intent": classify_intent(kw),
                # For keyword gap items the site typically does not rank yet.
                "current_position": 0,
                "position_change": None,
                "impressions": 0,
                "clicks": 0,
                "ctr": 0,
                "competitors": [
                    {
                        "domain": c.get("domain") or "",
                        "url": c.get("url") or "",
                        "rank": c.get("rank"),
                    }
                    for c in competitors
                    if c.get("url")
                ],
            },
        )

    # Sort by highest opportunity: HIGH intent, then MEDIUM, then LOW, and by search volume desc.
    def sort_key(item: dict) -> tuple[int, int]:
        intent = item.get("intent") or "MEDIUM"
        intent_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(intent, 1)
        volume = item.get("avg_monthly_searches") or 0
        return intent_rank, -volume

    results.sort(key=sort_key)
    top_results = results[:100]

    # Cache the computed keyword gaps for this user for the same window as other
    # third-party data, so page loads reuse this instead of hitting DataForSEO.
    cache_ttl = int(THIRD_PARTY_CACHE_TTL.total_seconds())
    cache.set(cache_key, top_results, cache_ttl)
    cache.set(domain_cache_key, domain, cache_ttl)

    logger.info(
        "[SEO keywords] Returning %s fresh DataForSEO keyword gap items for user_id=%s.",
        len(top_results),
        request.user.id,
    )

    return Response({"keywords": top_results})


def _reviews_overview_response_from_snapshot(snapshot: ReviewsOverviewSnapshot) -> Response:
    """Build the same JSON shape as fetch_gbp_overview for consistency."""
    return Response({
        "star_rating": float(snapshot.star_rating or 0),
        "previous_star_rating": float(snapshot.previous_star_rating or 0),
        "total_reviews": snapshot.total_reviews or 0,
        "new_reviews_this_month": snapshot.new_reviews_this_month or 0,
        "response_rate_pct": float(snapshot.response_rate_pct or 0),
        "industry_avg_response_pct": float(snapshot.industry_avg_response_pct or 45),
        "requests_sent": snapshot.requests_sent or 0,
        "conversion_pct": float(snapshot.conversion_pct or 0),
        "connected": True,
    })


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def reviews_overview(request: HttpRequest) -> Response:
    """
    Return Reviews Agent overview: star rating, total reviews, response rate, requests sent.
    Uses a 1-hour cache: if we have fresh GBP snapshot data, return it without calling the API.
    """
    from .gbp_client import fetch_gbp_overview

    user_id = request.user.id
    gbp_qs = GoogleBusinessProfileConnection.objects.filter(user=request.user)
    has_connection = gbp_qs.exists()
    connected = gbp_qs.exclude(refresh_token="").exists()

    logger.info(
        "[reviews_overview] user_id=%s has_connection=%s connected=%s (non-empty refresh_token)",
        user_id,
        has_connection,
        connected,
    )

    if connected:
        now = datetime.now(timezone.utc)
        cutoff = now - THIRD_PARTY_CACHE_TTL
        force_refresh = request.GET.get("refresh") == "1"
        if not force_refresh:
            try:
                snapshot = ReviewsOverviewSnapshot.objects.get(user=request.user)
                if snapshot.last_fetched_at >= cutoff:
                    logger.info(
                        "[reviews_overview] user_id=%s returning cached snapshot total_reviews=%s star_rating=%s",
                        user_id,
                        snapshot.total_reviews,
                        snapshot.star_rating,
                    )
                    return _reviews_overview_response_from_snapshot(snapshot)
            except ReviewsOverviewSnapshot.DoesNotExist:
                logger.info("[reviews_overview] user_id=%s no snapshot yet, will call API", user_id)

        # At most one call per user per minute (Google My Business quota: requests per minute).
        lock_key = f"gbp_api_lock:{user_id}"
        if not cache.add(lock_key, "1", timeout=GBP_API_MIN_INTERVAL_SECONDS):
            logger.info(
                "[reviews_overview] user_id=%s rate limited (API called in last %ss), returning cached",
                user_id,
                GBP_API_MIN_INTERVAL_SECONDS,
            )
            try:
                snapshot = ReviewsOverviewSnapshot.objects.get(user=request.user)
                return _reviews_overview_response_from_snapshot(snapshot)
            except ReviewsOverviewSnapshot.DoesNotExist:
                return Response({
                    "star_rating": 0,
                    "previous_star_rating": 0,
                    "total_reviews": 0,
                    "new_reviews_this_month": 0,
                    "response_rate_pct": 0,
                    "industry_avg_response_pct": 45,
                    "requests_sent": 0,
                    "conversion_pct": 0,
                    "connected": True,
                })

        try:
            logger.info("[reviews_overview] user_id=%s calling fetch_gbp_overview", user_id)
            data = fetch_gbp_overview(request.user)
            if data:
                data["connected"] = True
                logger.info(
                    "[reviews_overview] user_id=%s API returned total_reviews=%s star_rating=%s",
                    user_id,
                    data.get("total_reviews"),
                    data.get("star_rating"),
                )
                return Response(data)
            logger.warning("[reviews_overview] user_id=%s fetch_gbp_overview returned None", user_id)
        except Exception as e:
            logger.exception("[reviews_overview] fetch_gbp_overview failed: %s", e)

    # Fallback: cached snapshot or defaults
    try:
        snapshot = ReviewsOverviewSnapshot.objects.get(user=request.user)
        logger.info(
            "[reviews_overview] user_id=%s fallback to existing snapshot total_reviews=%s",
            user_id,
            snapshot.total_reviews,
        )
        return _reviews_overview_response_from_snapshot(snapshot)
    except ReviewsOverviewSnapshot.DoesNotExist:
        logger.warning(
            "[reviews_overview] user_id=%s no connection or API failed; returning zeros connected=False",
            user_id,
        )
        return Response({
            "star_rating": 0,
            "previous_star_rating": 0,
            "total_reviews": 0,
            "new_reviews_this_month": 0,
            "response_rate_pct": 0,
            "industry_avg_response_pct": 45,
            "requests_sent": 0,
            "conversion_pct": 0,
            "connected": False,
        })


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_chat(request: HttpRequest) -> Response:
    """SEO agent chat endpoint – delegates to OpenAI utils implementation."""
    return openai_utils.seo_chat(request)


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def reviews_chat(request: HttpRequest) -> Response:
    """Reviews agent chat endpoint – same pattern as SEO, different system role and tables."""
    return openai_utils.reviews_chat(request)


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

