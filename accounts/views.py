import logging
import secrets
from urllib.parse import urlencode, unquote, quote, urlparse
from datetime import datetime, date, timedelta, timezone

import requests
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
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
    GoogleSearchConsoleConnection,
    GoogleBusinessProfileConnection,
    MetaAdsConnection,
    SEOOverviewSnapshot,
    ReviewsOverviewSnapshot,
    AgentConversation,
    AgentMessage,
)

# Third-party API cache: only refetch from GSC/GBP if last fetch was >= this long ago.
THIRD_PARTY_CACHE_TTL = timedelta(hours=1)
# Google My Business Account Management API: 1 request per minute per project. Enforce per-user to avoid 429.
GBP_API_MIN_INTERVAL_SECONDS = 60
from .serializers import (
    BusinessProfileAEOSerializer,
    BusinessProfileSEOSerializer,
    BusinessProfileSerializer,
)
from .meta_ads_utils import get_meta_ads_status_for_user
from .tiktok_ads_utils import get_tiktok_ads_status_for_user
from .dataforseo_utils import (
    get_ranked_keywords_visibility,
    compute_professional_seo_score,
    get_or_refresh_seo_score_for_user,
    get_profile_location_code,
    normalize_domain,
)
from .constants import SEO_SNAPSHOT_TTL
from . import openai_utils
from . import debug_log as _debug
from .aeo.aeo_utils import (
    AEO_ONBOARDING_PROMPT_COUNT,
    aeo_business_input_from_profile,
    build_full_aeo_prompt_plan,
    plan_items_from_saved_prompt_strings,
)

logger = logging.getLogger(__name__)
User = get_user_model()


def _seo_snapshot_context_for_profile(profile: BusinessProfile | None) -> tuple[str, int]:
    """Resolve snapshot identity context for SEO snapshots."""
    mode = str(getattr(profile, "seo_location_mode", "organic") or "organic").strip().lower()
    if mode != "local":
        return "organic", 0
    default_location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    resolved_code, _fallback, _label = get_profile_location_code(profile, default_location_code)
    return "local", int(resolved_code or 0)


def classify_intent(keyword: str) -> str:
    """
    Simple keyword intent classifier (kept local to avoid non-OAuth Google Ads dependency).
    """
    k = (keyword or "").lower()
    high_triggers = [
        "buy", "price", "cost", "near me", "coupon", "deal", "best",
        "hire", "book", "quote", "service",
    ]
    low_triggers = [
        "what is", "definition", "how to", "tutorial", "examples", "guide", "meaning",
    ]
    if any(t in k for t in high_triggers):
        return "HIGH"
    if any(t in k for t in low_triggers):
        return "LOW"
    return "MEDIUM"


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

    # Require a filled BusinessProfile before entering the app.
    # Django's business_profile endpoint auto-creates a blank row; treat that as "not onboarded".
    try:
        qs = BusinessProfile.objects.filter(user=user)
        profile = qs.filter(is_main=True).first() or qs.first()
        if not profile:
            next_url = frontend_base + "/onboarding"
        else:
            has_business_name = bool((profile.business_name or "").strip())
            has_website_url = bool((profile.website_url or "").strip())
            has_business_address = bool((profile.business_address or "").strip())
            if not has_business_name or not has_website_url or not has_business_address:
                next_url = frontend_base + "/onboarding"
    except Exception:
        logger.exception("[auth] business_profile existence check failed; continuing to next_url")

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

    profile_for_context = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    snapshot_mode, snapshot_location_code = _seo_snapshot_context_for_profile(profile_for_context)

    # Serve from cache if we have a snapshot for this period fetched within the last hour (unless refresh=1).
    if not force_refresh:
        try:
            snapshot = SEOOverviewSnapshot.objects.get(
                user=request.user,
                period_start=start_current,
                cached_location_mode=snapshot_mode,
                cached_location_code=snapshot_location_code,
            )
            if snapshot.last_fetched_at >= cutoff:
                prev_clicks = snapshot.prev_organic_visitors or 0
                organic_visitors = snapshot.organic_visitors or 0
                if prev_clicks == 0:
                    organic_growth_pct = 100.0 if organic_visitors > 0 else 0.0
                else:
                    organic_growth_pct = ((organic_visitors - prev_clicks) / prev_clicks) * 100.0

                seo_score = compute_professional_seo_score(
                    estimated_traffic=organic_visitors,
                    keywords_count=snapshot.keywords_ranking or 0,
                    top3_positions=snapshot.top3_positions or 0,
                    top10_positions=snapshot.top3_positions or 0,
                    avg_keyword_difficulty=None,
                    competitor_avg_traffic=0.0,
                )
                # #region agent log
                _debug.log("views.py:seo_overview:cache_hit", "returning from snapshot cache", {"organic_visitors": organic_visitors, "last_fetched_at": str(snapshot.last_fetched_at), "cutoff": str(cutoff)}, "H2")
                # #endregion
                return Response(
                    {
                        "organic_visitors": organic_visitors,
                        "keywords_ranking": snapshot.keywords_ranking or 0,
                        "top3_positions": snapshot.top3_positions or 0,
                        "organic_growth_pct": organic_growth_pct,
                        "seo_score": seo_score,
                    },
                )
        except SEOOverviewSnapshot.DoesNotExist:
            pass

    # #region agent log
    _debug.log("views.py:seo_overview:cache_miss", "cache miss or force_refresh; calling DataForSEO", {}, "H2")
    # #endregion
    try:
        # Cache miss, stale, or refresh=1: call DataForSEO Labs ranked_keywords API.
        # Always use the user's main business profile (selected in Settings).
        profile = (
            BusinessProfile.objects.filter(user=request.user, is_main=True).first()
            or BusinessProfile.objects.filter(user=request.user).first()
        )
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
                    cached_location_mode=snapshot_mode,
                    cached_location_code=snapshot_location_code,
                )
                prev_vis = snapshot.organic_visitors or 0
            except SEOOverviewSnapshot.DoesNotExist:
                prev_vis = 0
            organic_growth_pct = 0.0
            seo_score = compute_professional_seo_score(
                estimated_traffic=0.0,
                keywords_count=0,
                top3_positions=0,
                top10_positions=0,
                avg_keyword_difficulty=None,
                competitor_avg_traffic=0.0,
            )
            return Response(
                {
                    "organic_visitors": 0,
                    "keywords_ranking": 0,
                    "top3_positions": 0,
                    "organic_growth_pct": organic_growth_pct,
                    "seo_score": seo_score,
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
                cached_location_mode=snapshot_mode,
                cached_location_code=snapshot_location_code,
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
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
        )
        snapshot.organic_visitors = current_visibility
        snapshot.prev_organic_visitors = prev_vis
        snapshot.keywords_ranking = keywords_ranking
        snapshot.top3_positions = top3_positions
        snapshot.save()

        seo_score = compute_professional_seo_score(
            estimated_traffic=current_visibility,
            keywords_count=keywords_ranking,
            top3_positions=top3_positions,
            top10_positions=top3_positions,
            avg_keyword_difficulty=None,
            competitor_avg_traffic=0.0,
        )

        return Response(
            {
                # For backwards compatibility with the frontend naming, map current visibility
                # to organic_visitors. The label in the UI can describe this as visibility.
                "organic_visitors": current_visibility,
                "keywords_ranking": keywords_ranking,
                "top3_positions": top3_positions,
                "organic_growth_pct": organic_growth_pct,
                "seo_score": seo_score,
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

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    site_url = profile.website_url if profile and profile.website_url else ""
    if not site_url:
        return Response({"keywords": []})

    domain = normalize_domain(site_url)

    if not domain:
        return Response({"keywords": []})

    now_utc = datetime.now(timezone.utc)
    start_current = now_utc.date().replace(day=1)
    snapshot_mode, snapshot_location_code = _seo_snapshot_context_for_profile(profile)
    snapshot = (
        SEOOverviewSnapshot.objects.filter(
            user=request.user,
            period_start=start_current,
            cached_domain__iexact=domain,
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
        )
        .order_by("-last_fetched_at")
        .first()
    )

    snapshot_fresh = bool(
        snapshot
        and getattr(snapshot, "refreshed_at", None)
        and (now_utc - snapshot.refreshed_at) <= SEO_SNAPSHOT_TTL
    )

    # Never call DataForSEO competitors endpoint on routine page loads.
    # Recompute snapshot only on explicit refresh or once snapshot is stale.
    if force_refresh or not snapshot_fresh:
        get_or_refresh_seo_score_for_user(
            request.user,
            site_url=site_url,
            force_refresh=force_refresh,
        )
        snapshot = (
            SEOOverviewSnapshot.objects.filter(
                user=request.user,
                period_start=start_current,
                cached_domain__iexact=domain,
                cached_location_mode=snapshot_mode,
                cached_location_code=snapshot_location_code,
            )
            .order_by("-last_fetched_at")
            .first()
        )

    results: list[dict] = []
    snapshot_keywords = list(getattr(snapshot, "top_keywords", None) or [])
    for item in snapshot_keywords:
        kw = item.get("keyword")
        if not kw:
            continue
        competitors = item.get("competitors") or []
        top_comp_domain = str(item.get("top_competitor_domain") or "").strip()
        # Gap rows are those enriched from domain_intersection competitor data.
        if not (top_comp_domain or competitors):
            continue
        search_volume = item.get("search_volume")
        results.append(
            {
                "keyword": kw,
                "avg_monthly_searches": int(search_volume) if search_volume is not None else None,
                "intent": classify_intent(kw),
                "current_position": int(item.get("rank") or 0),
                "position_change": None,
                "impressions": 0,
                "clicks": 0,
                "ctr": 0,
                "top_competitor": item.get("top_competitor"),
                "top_competitor_domain": top_comp_domain or None,
                "top_competitor_rank": item.get("top_competitor_rank"),
                "competitors": [
                    {
                        "domain": c.get("domain") or "",
                        "url": c.get("url") or "",
                        "rank": c.get("rank"),
                    }
                    for c in competitors
                    if isinstance(c, dict) and (c.get("url") or c.get("domain"))
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

    logger.info(
        "[SEO keywords] Returning %s snapshot keyword gap items for user_id=%s.",
        len(top_results),
        request.user.id,
    )

    return Response({"keywords": top_results})


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_keyword_debug(request: HttpRequest) -> Response:
    """
    Debug helper: return the saved SEOOverviewSnapshot top_keywords row for a keyword.

    Query params:
    - keyword (required): keyword text to match (case-insensitive + trimmed)
    """
    raw_keyword = request.GET.get("keyword") or ""
    keyword = raw_keyword.strip()
    if not keyword:
        return Response({"detail": "Missing required query param: keyword"}, status=400)

    normalized_target = keyword.lower().strip()

    today = datetime.now(timezone.utc).date()
    start_current = today.replace(day=1)
    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    snapshot_mode, snapshot_location_code = _seo_snapshot_context_for_profile(profile)

    snapshot = (
        SEOOverviewSnapshot.objects.filter(
            user=request.user,
            period_start=start_current,
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
        )
        .order_by("-last_fetched_at")
        .first()
    )
    if not snapshot:
        return Response({"detail": "No SEOOverviewSnapshot for this user this month."}, status=404)

    top_keywords = getattr(snapshot, "top_keywords", None) or []
    matches: list[dict] = []
    for row in top_keywords:
        row_kw = str((row or {}).get("keyword") or "").strip()
        if not row_kw:
            continue
        if row_kw.lower().strip() == normalized_target:
            matches.append(row)

    # If no exact normalized matches, try best-effort substring match to help triage.
    if not matches:
        for row in top_keywords:
            row_kw = str((row or {}).get("keyword") or "").strip()
            if not row_kw:
                continue
            if normalized_target in row_kw.lower().strip() or row_kw.lower().strip() in normalized_target:
                matches.append(row)

    def _ui_rank_display(r: Any) -> Any:
        try:
            if r is None:
                return "—"
            r_int = int(r)
            return f"#{r_int}" if r_int > 0 else "—"
        except Exception:
            return "—"

    def _competitor_outranking(your_rank: Any, comp_rank: Any) -> bool:
        try:
            if your_rank is None:
                return False
            your = int(your_rank)
            if your <= 0:
                return False
            if comp_rank is None:
                return False
            comp = int(comp_rank)
            if comp <= 0:
                return False
            return comp < your
        except Exception:
            return False

    enriched: list[dict] = []
    for m in matches[:5]:
        your_rank = m.get("rank")
        top_comp_domain = m.get("top_competitor_domain") or m.get("top_competitor") or None
        top_comp_rank = m.get("top_competitor_rank")
        competitors = m.get("competitors") if isinstance(m.get("competitors"), list) else []
        any_outranking = any(
            _competitor_outranking(your_rank, c.get("rank")) for c in competitors if isinstance(c, dict)
        )
        enriched.append(
            {
                "keyword": m.get("keyword"),
                "rank": your_rank,
                "ui_rank_display": _ui_rank_display(your_rank),
                "top_competitor_domain": top_comp_domain,
                "top_competitor_rank": top_comp_rank,
                "top_competitor_outranks_you": _competitor_outranking(your_rank, top_comp_rank),
                "competitors_count": len(competitors),
                "competitors": competitors[:5],
                "any_competitor_outranks_you": any_outranking,
            }
        )

    return Response(
        {
            "debug_keyword": keyword,
            "snapshot_id": snapshot.id,
            "snapshot_refreshed_at": snapshot.refreshed_at,
            "keywords_matches_count": len(matches),
            "matches": enriched,
        }
    )


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

    - GET: returns the user's main business profile (creates one if missing).
    - PATCH/PUT: updates main profile fields; creates a main profile if it does not exist yet.

    Query param ``skip_heavy=1`` (GET and PATCH): omit DataForSEO and on-the-fly AEO readiness work
    in the serializer (returns null/empty SEO fields and a stub AEO bundle). Use during onboarding
    to avoid paid API calls when only updating basic fields.
    """
    profile_qs = BusinessProfile.objects.filter(user=request.user)
    profile = profile_qs.filter(is_main=True).first()
    if profile is None:
        # If the user has no profiles yet, create a main profile for them.
        if not profile_qs.exists():
            profile = BusinessProfile.objects.create(user=request.user, is_main=True)
        else:
            # Fallback: use the first existing profile as the main one.
            profile = profile_qs.first()

    if request.method == "GET":
        force_aeo_refresh = str(request.GET.get("refresh_aeo", "")).strip().lower() in {"1", "true", "yes"}
        skip_heavy = str(request.GET.get("skip_heavy", "")).strip().lower() in {"1", "true", "yes"}
        serializer = BusinessProfileSerializer(
            profile,
            context={
                "force_aeo_refresh": force_aeo_refresh,
                "disable_seo_context_for_aeo": True,
                "skip_heavy_profile_metrics": skip_heavy,
            },
        )
        return Response(serializer.data)

    # For PATCH/PUT, apply partial updates
    skip_heavy = str(request.GET.get("skip_heavy", "")).strip().lower() in {"1", "true", "yes"}
    old_site_url = profile.website_url
    serializer = BusinessProfileSerializer(
        profile,
        data=request.data,
        partial=True,
    )
    serializer.is_valid(raise_exception=True)
    serializer.save()

    # If the website URL changed, proactively refresh the SEO score snapshot
    # so that the frontend (and SEO agent) immediately see metrics for the new site.
    new_site_url = serializer.instance.website_url
    if new_site_url and new_site_url != old_site_url:
        try:
            get_or_refresh_seo_score_for_user(
                request.user,
                site_url=new_site_url,
            )
        except Exception:
            # Never block profile saves on DataForSEO errors.
            pass

    out = BusinessProfileSerializer(
        serializer.instance,
        context={
            "disable_seo_context_for_aeo": True,
            "skip_heavy_profile_metrics": skip_heavy,
        },
    )
    return Response(out.data)


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_profile_data(request: HttpRequest) -> Response:
    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        profile = BusinessProfile.objects.create(user=request.user, is_main=True)
    serializer = BusinessProfileSEOSerializer(profile)
    payload = dict(serializer.data)

    # Expose SEO score trend from historical snapshots for dashboard charting.
    website = str(getattr(profile, "website_url", "") or "").strip()
    parsed_domain = normalize_domain(website) if website else ""

    snapshots_qs = SEOOverviewSnapshot.objects.filter(
        user=request.user,
    )
    if parsed_domain:
        snapshots_qs = snapshots_qs.filter(cached_domain__iexact=parsed_domain)

    seo_score_history = [
        {
            "period_started_at": snap.period_start.isoformat(),
            "seo_score": int(snap.search_performance_score),
        }
        for snap in snapshots_qs.order_by("period_start", "id")
        if snap.search_performance_score is not None
    ]
    payload["seo_score_history"] = seo_score_history
    return Response(payload)


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_score_history_data(request: HttpRequest) -> Response:
    """
    Return SEO score points (search_performance_score) from all matching
    SEOOverviewSnapshot rows for the authenticated user + current profile domain.
    """
    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"points": []})

    website = str(getattr(profile, "website_url", "") or "").strip()
    parsed_domain = normalize_domain(website) if website else ""
    start_current = datetime.now(timezone.utc).date().replace(day=1)

    # Ensure current-period snapshot is present in history (without forcing a refresh).
    if website:
        try:
            get_or_refresh_seo_score_for_user(
                request.user,
                site_url=website,
                force_refresh=False,
            )
        except Exception:
            logger.exception(
                "[seo_score_history] failed to warm current snapshot user_id=%s website=%s",
                getattr(request.user, "id", None),
                website,
            )

    snapshots_qs = SEOOverviewSnapshot.objects.filter(
        user=request.user,
    )
    if parsed_domain:
        snapshots_qs = snapshots_qs.filter(cached_domain__iexact=parsed_domain)

    points = [
        {
            "period_started_at": snap.period_start.isoformat(),
            "seo_score": int(snap.search_performance_score),
        }
        for snap in snapshots_qs.order_by("period_start", "id")
        if snap.search_performance_score is not None
    ]

    # Defensive fallback: if current period exists but score is missing from points,
    # append current seo_score from the warm call so chart includes "now".
    has_current_point = any(p.get("period_started_at") == start_current.isoformat() for p in points)
    if not has_current_point and website:
        try:
            current_bundle = get_or_refresh_seo_score_for_user(
                request.user,
                site_url=website,
                force_refresh=False,
            ) or {}
            current_score = current_bundle.get("seo_score")
            if current_score is not None:
                points.append(
                    {
                        "period_started_at": start_current.isoformat(),
                        "seo_score": int(current_score),
                    }
                )
                points.sort(key=lambda p: str(p.get("period_started_at") or ""))
        except Exception:
            logger.exception(
                "[seo_score_history] failed fallback current-point append user_id=%s",
                getattr(request.user, "id", None),
            )
    return Response({"points": points})


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_profile_data(request: HttpRequest) -> Response:
    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        profile = BusinessProfile.objects.create(user=request.user, is_main=True)
    serializer = BusinessProfileAEOSerializer(
        profile,
        context={"force_aeo_refresh": False, "disable_seo_context_for_aeo": True},
    )
    return Response(serializer.data)


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_prompt_coverage_data(request: HttpRequest) -> Response:
    """
    Cached-only prompt coverage read for AI Visibility UI.
    Returns monitored prompts from saved AEOResponseSnapshot rows plus latest extraction status.
    """
    from .models import AEOResponseSnapshot

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"prompts": [], "monitored_count": 0})

    responses = (
        AEOResponseSnapshot.objects.filter(profile=profile)
        .order_by("-created_at", "-id")
    )
    prompts: list[dict] = []
    for resp in responses:
        latest_ex = resp.extraction_snapshots.order_by("-created_at", "-id").first()
        competitors_count = 0
        cited = False
        if latest_ex is not None:
            cited = bool(latest_ex.brand_mentioned)
            comps = latest_ex.competitors_json or []
            competitors_count = len(comps) if isinstance(comps, list) else 0
        prompts.append(
            {
                "id": resp.id,
                "prompt": (resp.prompt_text or "").strip(),
                "prompt_type": str(resp.prompt_type or ""),
                "weight": float(resp.weight or 1.0),
                "cited": cited,
                "competitors_cited": competitors_count,
                "response_created_at": resp.created_at.isoformat() if resp.created_at else None,
            }
        )
    return Response({"prompts": prompts, "monitored_count": len(prompts)})


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_pipeline_status_data(request: HttpRequest) -> Response:
    """
    Cached-only AEO pipeline status endpoint.
    No OpenAI/DataForSEO calls are made in this read path.
    """
    from .models import AEOExecutionRun, AEORecommendationRun, AEOScoreSnapshot

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response(
            {
                "run": None,
                "score_snapshot": None,
                "recommendation_run": None,
                "freshness": {"run_age_days": None, "score_age_days": None, "recommendation_age_days": None},
            }
        )

    latest_run = (
        AEOExecutionRun.objects.filter(profile=profile)
        .order_by("-created_at")
        .first()
    )
    latest_score = (
        AEOScoreSnapshot.objects.filter(profile=profile)
        .order_by("-created_at")
        .first()
    )
    latest_reco = (
        AEORecommendationRun.objects.filter(profile=profile)
        .order_by("-created_at")
        .first()
    )
    now = datetime.now(timezone.utc)

    run_age_days = None
    if latest_run and latest_run.finished_at:
        run_age_days = max(0.0, (now - latest_run.finished_at).total_seconds() / 86400.0)
    score_age_days = None
    if latest_score:
        score_age_days = max(0.0, (now - latest_score.created_at).total_seconds() / 86400.0)
    reco_age_days = None
    if latest_reco:
        reco_age_days = max(0.0, (now - latest_reco.created_at).total_seconds() / 86400.0)

    return Response(
        {
            "run": None
            if not latest_run
            else {
                "id": latest_run.id,
                "status": latest_run.status,
                "fetch_mode": latest_run.fetch_mode,
                "cache_hit": latest_run.cache_hit,
                "prompt_count_requested": latest_run.prompt_count_requested,
                "prompt_count_executed": latest_run.prompt_count_executed,
                "prompt_count_failed": latest_run.prompt_count_failed,
                "extraction_status": latest_run.extraction_status,
                "scoring_status": latest_run.scoring_status,
                "recommendation_status": latest_run.recommendation_status,
                "extraction_count": latest_run.extraction_count,
                "score_snapshot_id": latest_run.score_snapshot_id,
                "recommendation_run_id": latest_run.recommendation_run_id,
                "seo_triggered_at": latest_run.seo_triggered_at.isoformat() if latest_run.seo_triggered_at else None,
                "seo_trigger_status": latest_run.seo_trigger_status or "",
                "started_at": latest_run.started_at.isoformat() if latest_run.started_at else None,
                "finished_at": latest_run.finished_at.isoformat() if latest_run.finished_at else None,
                "error_message": latest_run.error_message or "",
            },
            "score_snapshot": None
            if not latest_score
            else {
                "id": latest_score.id,
                "visibility_score": float(latest_score.visibility_score),
                "weighted_position_score": float(latest_score.weighted_position_score),
                "citation_share": float(latest_score.citation_share),
                "total_prompts": int(latest_score.total_prompts),
                "total_mentions": int(latest_score.total_mentions),
                "created_at": latest_score.created_at.isoformat(),
            },
            "recommendation_run": None
            if not latest_reco
            else {
                "id": latest_reco.id,
                "score_snapshot_id": latest_reco.score_snapshot_id,
                "recommendation_count": len(latest_reco.recommendations_json or []),
                "created_at": latest_reco.created_at.isoformat(),
            },
            "freshness": {
                "run_age_days": run_age_days,
                "score_age_days": score_age_days,
                "recommendation_age_days": reco_age_days,
            },
        }
    )


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def refresh_aeo_snapshot(request: HttpRequest) -> Response:
    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"detail": "No main business profile is configured."}, status=400)
    logger.info(
        "[refresh_endpoint] module=aeo user_id=%s profile_id=%s refresh_started=true external_api_called=true",
        getattr(request.user, "id", None),
        getattr(profile, "id", None),
    )
    serializer = BusinessProfileAEOSerializer(
        profile,
        context={"force_aeo_refresh": True, "disable_seo_context_for_aeo": True},
    )
    logger.info(
        "[refresh_endpoint] module=aeo user_id=%s profile_id=%s refresh_completed=true external_api_called=true",
        getattr(request.user, "id", None),
        getattr(profile, "id", None),
    )
    return Response(serializer.data)


def _serialize_aeo_prompt_items(items: list | None) -> list[dict]:
    out: list[dict] = []
    for p in items or []:
        if not isinstance(p, dict):
            continue
        text = (p.get("prompt") or "").strip()
        if not text:
            continue
        try:
            w = float(p.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        out.append(
            {
                "prompt": text,
                "type": str(p.get("type") or ""),
                "weight": w,
                "dynamic": bool(p.get("dynamic", False)),
            }
        )
    return out


def _truthy_openai_param(raw: str | None) -> bool:
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _onboarding_plan_include_openai(request: HttpRequest, body: dict) -> bool:
    if request.method == "GET":
        return _truthy_openai_param(request.GET.get("include_openai"))
    v = body.get("include_openai")
    if v is None:
        return True
    if isinstance(v, bool):
        return v
    return _truthy_openai_param(str(v))


def _onboarding_reuse_saved_prompts(request: HttpRequest, body: dict) -> bool:
    v = body.get("reuse_saved") if isinstance(body, dict) else None
    if isinstance(v, bool):
        return v
    if v is not None:
        return _truthy_openai_param(str(v))
    return _truthy_openai_param(request.GET.get("reuse_saved"))


def _aeo_prompt_target_count() -> int:
    testing_mode = bool(getattr(settings, "AEO_TESTING_MODE", False))
    if testing_mode:
        try:
            return max(1, int(getattr(settings, "AEO_TEST_PROMPT_COUNT", 10)))
        except (TypeError, ValueError):
            return 10
    try:
        return max(1, int(getattr(settings, "AEO_PROD_PROMPT_COUNT", 50)))
    except (TypeError, ValueError):
        return 50


def _first_nonempty_str(*vals) -> str | None:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


@csrf_exempt
@api_view(["GET", "POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_onboarding_prompt_plan(request: HttpRequest) -> Response:
    """
    AEO onboarding prompt plan: OpenAI-generated prompt set (target count from settings).

    Call after PATCH ``/api/business-profile/`` with name, URL, address, and optional industry.

    **GET** or **POST** (same behavior). Query params and JSON body are merged; body wins on conflict.

    Optional parameters (query and/or JSON):

    - ``include_openai`` — accepted for backward compatibility; prompt planning is OpenAI-only.
    - ``city`` — optional override for templates (overrides inference from ``business_address``; use
      ``business_address`` on the profile so prompts use city/region, not street-level detail).
    - ``industry`` — explicit industry override for prompt context.
    - ``reuse_saved`` — when ``true`` and the profile already has ``selected_aeo_prompts`` of length
      ``AEO_ONBOARDING_PROMPT_COUNT``, return that list without calling OpenAI or rebuilding templates.

    Response ``meta`` includes ``openai_status`` (``ok`` | ``partial`` | ``failed_empty`` |
    ``reused_saved``) and ``openai_message`` when generation cannot hit target count.
    """
    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        profile = BusinessProfile.objects.create(user=request.user, is_main=True)

    body = request.data if request.method == "POST" else {}
    if not isinstance(body, dict):
        body = {}

    city_override = _first_nonempty_str(body.get("city"), request.GET.get("city"))
    industry_override = _first_nonempty_str(body.get("industry"), request.GET.get("industry"))
    include_openai = _onboarding_plan_include_openai(request, body)
    reuse_saved = _onboarding_reuse_saved_prompts(request, body)

    target_prompt_count = _aeo_prompt_target_count()

    saved = profile.selected_aeo_prompts or []
    if reuse_saved and isinstance(saved, list) and len(saved) == target_prompt_count:
        raw_items = plan_items_from_saved_prompt_strings([str(x) for x in saved])
        if len(raw_items) == target_prompt_count:
            ser = _serialize_aeo_prompt_items(raw_items)
            biz = aeo_business_input_from_profile(
                profile,
                city=city_override,
                industry=industry_override,
            ).as_dict()
            try:
                from .models import AEOExecutionRun
                from .tasks import run_aeo_phase1_execution_task

                inflight = AEOExecutionRun.objects.filter(
                    profile=profile,
                    status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
                ).exists()
                if not inflight:
                    run = AEOExecutionRun.objects.create(
                        profile=profile,
                        prompt_count_requested=len(ser),
                        status=AEOExecutionRun.STATUS_PENDING,
                    )
                    transaction.on_commit(
                        lambda run_id=run.id, prompt_payload=ser: run_aeo_phase1_execution_task.delay(
                            run_id, prompt_payload
                        )
                    )
            except Exception:
                logger.exception("[AEO onboarding] failed to enqueue phase1 execution (reuse_saved)")
            return Response(
                {
                    "groups": {
                        "fixed": [],
                        "dynamic": [],
                        "openai_generated": [],
                        "saved": ser,
                    },
                    "combined": ser,
                    "business": biz,
                    "meta": {
                        "openai_status": "reused_saved",
                        "openai_message": "",
                        "openai_prompt_count": 0,
                        "combined_count": len(ser),
                        "combined_target": target_prompt_count,
                        "combined_shortfall": 0,
                    },
                }
            )

    plan = build_full_aeo_prompt_plan(
        profile,
        city=city_override,
        industry=industry_override,
        include_openai=include_openai,
        target_combined_count=target_prompt_count,
    )
    fixed = _serialize_aeo_prompt_items(plan.get("fixed") or [])
    dynamic = _serialize_aeo_prompt_items(plan.get("dynamic") or [])
    openai_generated = _serialize_aeo_prompt_items(plan.get("openai_generated") or [])
    combined = _serialize_aeo_prompt_items(plan.get("combined") or [])
    try:
        profile.selected_aeo_prompts = [str(x.get("prompt") or "") for x in combined if str(x.get("prompt") or "").strip()]
        profile.save(update_fields=["selected_aeo_prompts", "updated_at"])
    except Exception:
        logger.exception("[AEO onboarding] failed saving selected_aeo_prompts")

    try:
        from .models import AEOExecutionRun
        from .tasks import run_aeo_phase1_execution_task

        inflight = AEOExecutionRun.objects.filter(
            profile=profile,
            status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
        ).exists()
        if not inflight:
            run = AEOExecutionRun.objects.create(
                profile=profile,
                prompt_count_requested=min(len(combined), target_prompt_count),
                status=AEOExecutionRun.STATUS_PENDING,
            )
            transaction.on_commit(
                lambda run_id=run.id, prompt_payload=combined: run_aeo_phase1_execution_task.delay(
                    run_id, prompt_payload
                )
            )
    except Exception:
        logger.exception("[AEO onboarding] failed to enqueue phase1 execution")

    return Response(
        {
            "groups": {
                "fixed": fixed,
                "dynamic": dynamic,
                "openai_generated": openai_generated,
                "saved": [],
            },
            "combined": combined,
            "business": plan.get("business") or {},
            "meta": plan.get("meta") or {},
        }
    )


@csrf_exempt
@api_view(["GET", "POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def business_profile_list(request: HttpRequest) -> Response:
    """
    List or create business profiles for the authenticated user.

    - GET: return all business profiles (main first).
    - POST: create a new business profile for this user.
      The first profile for a user is always created as is_main=True.
    """
    if request.method == "GET":
        profiles = BusinessProfile.objects.filter(user=request.user).order_by("-is_main", "created_at", "id")
        serializer = BusinessProfileSerializer(profiles, many=True)
        return Response(serializer.data)

    # POST: create a new profile under this user
    existing_qs = BusinessProfile.objects.filter(user=request.user)
    is_first = not existing_qs.exists()

    data = request.data.copy()
    # Never allow client to set a different user
    data.pop("user", None)

    if is_first:
        # First profile for this user is main by definition
        data["is_main"] = True
    else:
        # For additional profiles, default is_main to False unless explicitly set
        if "is_main" not in data:
            data["is_main"] = False

    serializer = BusinessProfileSerializer(data=data)
    serializer.is_valid(raise_exception=True)
    profile = serializer.save(user=request.user)

    # If this profile is being set as main, unset others
    if profile.is_main:
        existing_qs.exclude(pk=profile.pk).update(is_main=False)

    return Response(serializer.data, status=201)


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def refresh_seo_next_steps(request: HttpRequest) -> Response:
    """
    Regenerate SEO next steps (\"Your action plan\" / \"Do these now\") for the current main business profile.

    This:
    - Resolves the user's main BusinessProfile (is_main=True, or first as fallback)
    - Uses its website_url to identify the SEOOverviewSnapshot for the current month
    - Recomputes seo_next_steps immediately (bypassing the usual 7-day TTL)
    - Stores the new steps on the snapshot and updates BusinessProfile serialization
    """
    today = datetime.now(timezone.utc).date()
    start_current = today.replace(day=1)
    user = request.user

    profile = (
        BusinessProfile.objects.filter(user=user, is_main=True).first()
        or BusinessProfile.objects.filter(user=user).first()
    )
    if not profile or not profile.website_url:
        return Response(
            {"detail": "No main business profile with a website URL is configured."},
            status=400,
        )

    site_url = profile.website_url
    domain = normalize_domain(site_url)
    if not domain:
        return Response(
            {"detail": "Could not determine domain from website URL."},
            status=400,
        )

    try:
        snapshot_mode, snapshot_location_code = _seo_snapshot_context_for_profile(profile)
        snapshot = SEOOverviewSnapshot.objects.filter(
            user=user,
            period_start=start_current,
            cached_domain__iexact=domain,
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
        ).first()
    except Exception:
        snapshot = None

    if not snapshot:
        # If we don't yet have a snapshot for this domain/period, trigger the main SEO score helper
        # which will create one and enqueue enrichment tasks; then try again.
        from .dataforseo_utils import get_or_refresh_seo_score_for_user

        get_or_refresh_seo_score_for_user(user, site_url=site_url)
        snapshot = SEOOverviewSnapshot.objects.filter(
            user=user,
            period_start=start_current,
            cached_domain__iexact=domain,
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
        ).first()

    if not snapshot:
        return Response(
            {"detail": "SEO snapshot not available yet for this site. Try again in a few minutes."},
            status=202,
        )

    # Build fresh seo_data for this snapshot and forcibly regenerate next steps.
    from .openai_utils import generate_seo_next_steps

    seo_data = {
        "seo_score": int(snapshot.search_performance_score or 0),
        "missed_searches_monthly": int(getattr(snapshot, "missed_searches_monthly", 0) or 0),
        "organic_visitors": int(snapshot.organic_visitors or 0),
        "total_search_volume": int(getattr(snapshot, "total_search_volume", 0) or 0),
        "search_visibility_percent": int(getattr(snapshot, "search_visibility_percent", 0) or 0),
        "top_keywords": getattr(snapshot, "top_keywords", None) or [],
        "business_name": getattr(profile, "business_name", "") or "",
        "website_url": site_url,
        "business_description": getattr(profile, "description", "") or "",
    }

    try:
        steps = generate_seo_next_steps(seo_data)
    except Exception:
        return Response(
            {"detail": "Failed to generate SEO next steps. Please try again later."},
            status=500,
        )

    steps_list = list(steps)[:10] if steps else []
    snapshot.seo_next_steps = steps_list
    snapshot.seo_next_steps_refreshed_at = datetime.now(timezone.utc)
    snapshot.save(update_fields=["seo_next_steps", "seo_next_steps_refreshed_at"])

    # Return the updated main profile (serializer will now include fresh seo_next_steps)
    serializer = BusinessProfileSerializer(profile)
    return Response(serializer.data)


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def refresh_seo_snapshot(request: HttpRequest) -> Response:
    """
    Force a refresh of the SEO snapshot (keywords, rankings, visibility) for the user's
    main business profile, then return the updated profile.

    This calls get_or_refresh_seo_score_for_user with the current website_url so that:
    - top_keywords (including rank and missed_searches_monthly) are up to date
    - search_visibility_percent, missed_searches_monthly, etc. are recomputed
    """
    user = request.user

    profile = (
        BusinessProfile.objects.filter(user=user, is_main=True).first()
        or BusinessProfile.objects.filter(user=user).first()
    )
    if not profile or not profile.website_url:
        return Response(
            {"detail": "No main business profile with a website URL is configured."},
            status=400,
        )

    site_url = profile.website_url
    logger.info(
        "[refresh_endpoint] module=seo user_id=%s profile_id=%s refresh_started=true external_api_called=false",
        getattr(user, "id", None),
        getattr(profile, "id", None),
    )
    external_api_called = False
    try:
        # IMPORTANT:
        # ranked_keywords/live is returning rank_absolute=null in our current pipeline.
        # If we recompute/save a fresh snapshot here, it overwrites previously-enriched
        # keyword ranks back to null.
        #
        # Instead, we ONLY re-run enrichment tasks (gap keywords + next steps) on the
        # existing snapshot, so ranks stay intact and refresh is meaningful.
        from .dataforseo_utils import (
            normalize_domain,
            enrich_with_gap_keywords,
            enrich_with_llm_keywords,
            enrich_keyword_ranks_from_labs,
            recompute_snapshot_metrics_from_keywords,
            normalize_seo_snapshot_metrics,
        )
        from .tasks import (
            generate_snapshot_next_steps_task,
            generate_keyword_action_suggestions_task,
        )

        domain = normalize_domain(site_url) or ""
        if not domain:
            raise ValueError("Could not normalize domain for refresh_seo_snapshot")

        today = datetime.now(timezone.utc).date()
        start_current = today.replace(day=1)
        snapshot_mode, snapshot_location_code = _seo_snapshot_context_for_profile(profile)

        snapshot = SEOOverviewSnapshot.objects.filter(
            user=user,
            period_start=start_current,
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
        ).order_by("-last_fetched_at").first()

        # If snapshot doesn't exist yet, fall back to full creation (it will enqueue tasks).
        if not snapshot:
            get_or_refresh_seo_score_for_user(user, site_url=site_url)
            snapshot = SEOOverviewSnapshot.objects.filter(
                user=user,
                period_start=start_current,
                cached_location_mode=snapshot_mode,
                cached_location_code=snapshot_location_code,
            ).order_by("-last_fetched_at").first()

        if snapshot:
            # Synchronously enrich only keywords so the UI immediately reflects
            # updated "You rank #X" pills instead of showing cached rank=null
            # while background tasks are still running.
            location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
            language_code = getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en")
            user = snapshot.user

            top_keywords: list[dict] = [dict(k) for k in (getattr(snapshot, "top_keywords", None) or [])]
            enrich_with_gap_keywords(
                domain=domain,
                location_code=location_code,
                language_code=language_code,
                user=user,
                top_keywords=top_keywords,
            )
            external_api_called = True
            enrich_with_llm_keywords(
                user=user,
                location_code=location_code,
                top_keywords=top_keywords,
            )
            rank_stats = enrich_keyword_ranks_from_labs(
                domain=domain,
                location_code=location_code,
                language_code=language_code,
                top_keywords=top_keywords,
                user=user,
            )
            external_api_called = True
            total = int(rank_stats.get("total") or 0)
            ranked_after = int(rank_stats.get("non_null_after") or 0)
            coverage = (ranked_after / total) if total > 0 else 0.0
            min_coverage = float(getattr(settings, "SEO_RANK_ENRICHMENT_MIN_COVERAGE", 0.05))
            logger.info(
                "[SEO refresh] rank enrichment coverage user_id=%s total=%s ranked_after=%s coverage=%.2f%% filled_ranked=%s filled_gap=%s",
                getattr(user, "id", None),
                total,
                ranked_after,
                coverage * 100.0,
                int(rank_stats.get("filled_from_ranked") or 0),
                int(rank_stats.get("filled_from_gap") or 0),
            )
            if total > 0 and coverage < min_coverage:
                refresh_warning = (
                    "Rank enrichment coverage is too low; refresh aborted to avoid returning all-none ranks."
                )
                logger.warning(
                    "[SEO refresh] %s user_id=%s coverage=%.2f%% threshold=%.2f%%",
                    refresh_warning,
                    getattr(user, "id", None),
                    coverage * 100.0,
                    min_coverage * 100.0,
                )
                return Response(
                    {
                        "detail": refresh_warning,
                        "rank_coverage_percent": round(coverage * 100.0, 2),
                    },
                    status=409,
                )

            top_keywords_sorted = sorted(
                top_keywords,
                key=lambda x: x.get("search_volume", 0),
                reverse=True,
            )[:20]
            total_keywords = len(top_keywords_sorted)
            keywords_with_rank = sum(
                1 for k in top_keywords_sorted if isinstance(k.get("rank"), int) and (k.get("rank") or 0) > 0
            )
            keywords_with_competitor = sum(
                1
                for k in top_keywords_sorted
                if (
                    (k.get("top_competitor_domain") or k.get("top_competitor"))
                    or (isinstance(k.get("competitors"), list) and len(k.get("competitors") or []) > 0)
                )
            )
            keywords_with_outranking_competitor = sum(
                1
                for k in top_keywords_sorted
                if (
                    isinstance(k.get("rank"), int)
                    and (k.get("rank") or 0) > 0
                    and (
                        (
                            isinstance(k.get("top_competitor_rank"), int)
                            and (k.get("top_competitor_rank") or 0) > 0
                            and int(k.get("top_competitor_rank")) < int(k.get("rank"))
                        )
                        or (
                            isinstance(k.get("competitors"), list)
                            and any(
                                isinstance(c.get("rank"), int)
                                and (c.get("rank") or 0) > 0
                                and int(c.get("rank")) < int(k.get("rank"))
                                for c in (k.get("competitors") or [])
                            )
                        )
                    )
                )
            )
            rank_pct = (keywords_with_rank / total_keywords * 100.0) if total_keywords > 0 else 0.0
            competitor_pct = (keywords_with_competitor / total_keywords * 100.0) if total_keywords > 0 else 0.0
            outranking_competitor_pct = (
                keywords_with_outranking_competitor / total_keywords * 100.0
                if total_keywords > 0
                else 0.0
            )
            logger.info(
                "[SEO refresh] keyword coverage user_id=%s total=%s rank_non_null_pct=%.2f competitor_data_pct=%.2f outranking_competitor_pct=%.2f",
                getattr(user, "id", None),
                total_keywords,
                rank_pct,
                competitor_pct,
                outranking_competitor_pct,
            )
            metrics = normalize_seo_snapshot_metrics(
                recompute_snapshot_metrics_from_keywords(
                    top_keywords=top_keywords_sorted,
                    domain=domain,
                    location_code=location_code,
                    language_code=language_code,
                    seo_location_mode=str(snapshot_mode or "organic"),
                )
            )
            logger.info(
                "[SEO refresh] recompute user_id=%s keywords_with_rank=%s estimated_traffic_before=%s estimated_traffic_after=%s appearances_before=%s appearances_after=%s total_search_volume_before=%s total_search_volume_after=%s visibility_before=%s visibility_after=%s missed_before=%s missed_after=%s",
                getattr(user, "id", None),
                keywords_with_rank,
                int(snapshot.organic_visitors or 0),
                int(metrics.get("estimated_traffic") or 0),
                int(snapshot.estimated_search_appearances_monthly or 0),
                int(metrics.get("estimated_search_appearances_monthly") or 0),
                int(snapshot.total_search_volume or 0),
                int(metrics.get("total_search_volume") or 0),
                int(snapshot.search_visibility_percent or 0),
                int(metrics.get("search_visibility_percent") or 0),
                int(snapshot.missed_searches_monthly or 0),
                int(metrics.get("missed_searches_monthly") or 0),
            )
            if (
                keywords_with_rank > 0
                and int(metrics.get("search_visibility_percent") or 0) == 0
                and int(metrics.get("total_search_volume") or 0) > 0
            ):
                logger.warning(
                    "[SEO refresh] consistency_check user_id=%s ranked_keywords=%s visibility_zero_with_volume=true",
                    getattr(user, "id", None),
                    keywords_with_rank,
                )
            with transaction.atomic():
                snapshot_context = {
                    "mode": snapshot_mode,
                    "code": snapshot_location_code,
                    "label": "",
                }
                if snapshot_mode == "local":
                    default_location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
                    resolved_code, _used_fallback, resolved_label = get_profile_location_code(profile, default_location_code)
                    snapshot_context["code"] = int(resolved_code or 0)
                    snapshot_context["label"] = str(resolved_label or "")
                snapshot.top_keywords = top_keywords_sorted
                snapshot.keywords_enriched_at = datetime.now(timezone.utc)
                snapshot.refreshed_at = datetime.now(timezone.utc)
                snapshot.organic_visitors = int(metrics["estimated_traffic"])
                snapshot.total_search_volume = int(metrics["total_search_volume"])
                snapshot.estimated_search_appearances_monthly = int(metrics["estimated_search_appearances_monthly"])
                snapshot.search_visibility_percent = int(metrics["search_visibility_percent"])
                snapshot.missed_searches_monthly = int(metrics["missed_searches_monthly"])
                snapshot.search_performance_score = int(metrics["search_performance_score"])
                snapshot.keywords_ranking = int(metrics["keywords_ranking"])
                snapshot.top3_positions = int(metrics["top3_positions"])
                snapshot.cached_location_mode = str(snapshot_context.get("mode") or "organic")
                snapshot.cached_location_code = int(snapshot_context.get("code") or 0)
                snapshot.cached_location_label = str(snapshot_context.get("label") or "")
                snapshot.local_verification_applied = any(
                    str((row or {}).get("rank_source") or "baseline") == "local_verified"
                    for row in top_keywords_sorted
                )
                snapshot.local_verified_keyword_count = sum(
                    1 for row in top_keywords_sorted if (row or {}).get("local_verified_rank") is not None
                )
                snapshot.save(
                    update_fields=[
                        "top_keywords",
                        "keywords_enriched_at",
                        "refreshed_at",
                        "organic_visitors",
                        "total_search_volume",
                        "estimated_search_appearances_monthly",
                        "search_visibility_percent",
                        "missed_searches_monthly",
                        "search_performance_score",
                        "keywords_ranking",
                        "top3_positions",
                        "cached_location_mode",
                        "cached_location_code",
                        "cached_location_label",
                        "local_verification_applied",
                        "local_verified_keyword_count",
                    ]
                )

            # Next steps / action suggestions can remain async.
            generate_snapshot_next_steps_task.delay(snapshot.id)
            generate_keyword_action_suggestions_task.delay(snapshot.id)
    except Exception:
        # Never block the UI on SEO refresh failures; return the profile anyway.
        pass

    logger.info(
        "[refresh_endpoint] module=seo user_id=%s profile_id=%s refresh_completed=true external_api_called=%s",
        getattr(user, "id", None),
        getattr(profile, "id", None),
        bool(external_api_called),
    )
    serializer = BusinessProfileSEOSerializer(profile)
    return Response(serializer.data)

@csrf_exempt
@api_view(["GET", "PATCH", "PUT", "DELETE"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def business_profile_detail(request: HttpRequest, pk: int) -> Response:
    """
    Retrieve, update, or delete a single business profile owned by the authenticated user.
    Primarily used by the Settings page to update a specific profile or mark it as main.
    """
    try:
        profile = BusinessProfile.objects.get(id=pk, user=request.user)
    except BusinessProfile.DoesNotExist:
        return Response({"detail": "Not found."}, status=404)

    if request.method == "GET":
        serializer = BusinessProfileSerializer(profile)
        return Response(serializer.data)

    if request.method in ("PATCH", "PUT"):
        old_site_url = profile.website_url
        serializer = BusinessProfileSerializer(
            profile,
            data=request.data,
            partial=(request.method == "PATCH"),
        )
        serializer.is_valid(raise_exception=True)
        profile = serializer.save()

        # If is_main was set to True on this profile, unset main on all others.
        if profile.is_main:
            BusinessProfile.objects.filter(user=request.user).exclude(pk=profile.pk).update(is_main=False)

        # If the website URL changed, refresh SEO snapshot for this site.
        new_site_url = profile.website_url
        if new_site_url and new_site_url != old_site_url:
            try:
                get_or_refresh_seo_score_for_user(
                    request.user,
                    site_url=new_site_url,
                )
            except Exception:
                pass

        return Response(serializer.data)

    # DELETE
    profile.delete()
    return Response(status=204)


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

