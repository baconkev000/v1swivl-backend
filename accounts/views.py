import logging
import secrets
from urllib.parse import urlencode, unquote, quote, urlparse
from datetime import datetime, date, timedelta, timezone

import requests
import stripe
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone as django_timezone
from django.contrib.auth import authenticate, get_user_model, login, logout as django_logout
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authentication import SessionAuthentication
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import (
    AEOCompetitorSnapshot,
    AgentActivityLog,
    BusinessProfile,
    SEOOverviewSnapshot,
    AgentConversation,
    AgentMessage,
    OnboardingOnPageCrawl,
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
from .tiktok_ads_utils import get_tiktok_ads_status_for_user
from .dataforseo_utils import (
    get_ranked_keywords_visibility,
    compute_professional_seo_score,
    get_or_refresh_seo_score_for_user,
    get_profile_location_code,
    normalize_domain,
    seo_snapshot_context_for_profile,
)
from .constants import SEO_SNAPSHOT_TTL
from .onboarding_completion import user_has_completed_full_onboarding
from . import openai_utils
from . import debug_log as _debug
from .aeo.aeo_utils import (
    AEO_ONBOARDING_PROMPT_COUNT,
    aeo_business_input_from_onboarding_payload,
    aeo_business_input_from_profile,
    build_full_aeo_prompt_plan,
    plan_items_from_saved_prompt_strings,
)
from .aeo.perplexity_execution_utils import perplexity_execution_enabled
from .gemini_utils import gemini_execution_enabled
from .domain_utils import normalize_tracked_competitor_domain
from .stripe_billing import sync_from_checkout_session
from .stripe_billing import sync_from_invoice_paid
from .stripe_billing import sync_from_subscription

logger = logging.getLogger(__name__)

# Prompt coverage + pending checks: stored AEOResponseSnapshot.platform values we surface in the UI.
_AEO_COVERAGE_PLATFORM_SET = frozenset({"openai", "gemini", "perplexity"})
_AEO_PENDING_PLATFORM_ORDER = ("openai", "perplexity", "gemini")
User = get_user_model()


def _seo_snapshot_context_for_profile(profile: BusinessProfile | None) -> tuple[str, int]:
    """Resolve snapshot identity context for SEO snapshots."""
    return seo_snapshot_context_for_profile(profile)


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


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([AllowAny])
def auth_status(request: HttpRequest) -> Response:
    """
    Session-aware status for the Next.js app: whether the browser has a Django session
    and whether the main BusinessProfile looks onboarded (matches google_callback rules).
    """
    if not request.user.is_authenticated:
        return Response({"authenticated": False, "onboarding_complete": False})
    try:
        onboarding_complete = user_has_completed_full_onboarding(request.user)
    except Exception:
        logger.exception("[auth_status] business profile check failed")
        onboarding_complete = False
    return Response(
        {"authenticated": True, "onboarding_complete": onboarding_complete},
    )


@csrf_exempt
@api_view(["POST"])
@authentication_classes([])
@permission_classes([AllowAny])
def stripe_webhook(request: HttpRequest) -> Response:
    """
    Stripe webhook endpoint. Stripe events are the billing source of truth.
    """
    secret = str(getattr(settings, "STRIPE_WEBHOOK_SECRET", "") or "").strip()
    api_key = str(getattr(settings, "STRIPE_SECRET_KEY", "") or "").strip()
    if not secret or not api_key:
        logger.error("[stripe webhook] missing STRIPE_WEBHOOK_SECRET / STRIPE_SECRET_KEY")
        return Response({"error": "Stripe webhook is not configured."}, status=503)

    stripe.api_key = api_key
    sig = request.META.get("HTTP_STRIPE_SIGNATURE", "")
    payload = request.body
    try:
        event = stripe.Webhook.construct_event(payload, sig, secret)
    except ValueError:
        return Response({"error": "Invalid payload."}, status=400)
    except stripe.error.SignatureVerificationError:
        return Response({"error": "Invalid signature."}, status=400)

    # Stripe SDK returns ``StripeObject`` (dict-like, but no ``.get``).
    event_dict = event.to_dict_recursive() if hasattr(event, "to_dict_recursive") else dict(event)
    event_type = str(event_dict.get("type") or "")
    data = event_dict.get("data") if isinstance(event_dict.get("data"), dict) else {}
    handled = False
    try:
        if event_type == "checkout.session.completed":
            handled = sync_from_checkout_session(data)
        elif event_type == "invoice.paid":
            handled = sync_from_invoice_paid(data)
        elif event_type in {"customer.subscription.updated", "customer.subscription.deleted"}:
            handled = sync_from_subscription(data)
        else:
            handled = True
    except Exception:
        logger.exception("[stripe webhook] handler failed for %s", event_type)
        return Response({"error": "Webhook handler failed."}, status=500)

    if not handled:
        logger.error("[stripe webhook] ignored event without profile match: %s", event_type)
    return Response({"received": True})


def _onboarding_domain_claimed_by_other_user(domain: str, user) -> bool:
    """True if any other user's business profile uses this normalized domain."""
    if not domain:
        return False
    for bp in BusinessProfile.objects.exclude(user=user).exclude(website_url="").only("id", "website_url"):
        if normalize_domain(bp.website_url or "") == domain:
            return True
    return False


def _onboarding_reusable_crawl_for_user(user, domain: str) -> OnboardingOnPageCrawl | None:
    """Latest completed onboarding crawl for this user/domain with keywords and/or review topics."""
    if not domain:
        return None
    qs = OnboardingOnPageCrawl.objects.filter(
        user=user,
        domain=domain,
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
    ).order_by("-created_at")
    for crawl in qs[:5]:
        if crawl.ranked_keywords or crawl.review_topics:
            return crawl
    return None


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def onboarding_onpage_crawl_start(request: HttpRequest) -> Response:
    """
    Queue DataForSEO On-Page crawl (max 10 pages) after onboarding step 1.

    If the domain is already on another user's business profile, returns 409.
    If this user already has a completed crawl with ranked keywords for the domain,
    returns that row without enqueueing a new crawl.
    """
    from .tasks import onboarding_onpage_crawl_task, onboarding_review_topics_backfill_task

    website_url = (request.data.get("website_url") or "").strip()
    business_name = (request.data.get("business_name") or "").strip()
    location = (request.data.get("location") or "").strip()
    domain = normalize_domain(website_url)
    if not domain:
        return Response({"error": "A valid website_url is required"}, status=400)

    if _onboarding_domain_claimed_by_other_user(domain, request.user):
        return Response(
            {
                "error": (
                    "This website is already linked to another account. "
                    "Use a different URL or sign in with the account that owns it."
                ),
            },
            status=409,
        )

    reused = _onboarding_reusable_crawl_for_user(request.user, domain)
    if reused is not None:
        if (reused.ranked_keywords or []) and not (reused.review_topics or []):

            def _enqueue_backfill() -> None:
                onboarding_review_topics_backfill_task.delay(reused.id)

            transaction.on_commit(_enqueue_backfill)
        return Response(
            {
                "id": reused.id,
                "status": reused.status,
                "domain": domain,
                "reused": True,
                "ranked_keywords": reused.ranked_keywords or [],
                "review_topics": reused.review_topics or [],
                "review_topics_error": reused.review_topics_error or "",
                "prompt_plan_status": reused.prompt_plan_status,
                "prompt_plan_prompt_count": int(reused.prompt_plan_prompt_count or 0),
                "prompt_plan_error": reused.prompt_plan_error or "",
            },
        )

    try:
        profile_qs = BusinessProfile.objects.filter(user=request.user)
        profile = profile_qs.filter(is_main=True).first() or profile_qs.first()
        if profile is None:
            profile = BusinessProfile.objects.create(user=request.user, is_main=True)

        crawl = OnboardingOnPageCrawl.objects.create(
            user=request.user,
            business_profile=profile,
            domain=domain,
            max_pages=10,
            context={"business_name": business_name, "location": location},
            status=OnboardingOnPageCrawl.STATUS_PENDING,
        )
        cid = crawl.id

        def _enqueue() -> None:
            onboarding_onpage_crawl_task.delay(cid)

        transaction.on_commit(_enqueue)
        return Response(
            {
                "id": crawl.id,
                "status": crawl.status,
                "domain": domain,
                "prompt_plan_status": crawl.prompt_plan_status,
                "prompt_plan_prompt_count": int(crawl.prompt_plan_prompt_count or 0),
                "prompt_plan_error": crawl.prompt_plan_error or "",
            }
        )
    except Exception:
        logger.exception(
            "[onboarding_onpage_crawl_start] failed user_id=%s domain=%s",
            getattr(request.user, "id", None),
            domain,
        )
        return Response(
            {"error": "Crawl could not be started. Contact your administrator."},
            status=500,
        )


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def onboarding_crawl_latest(request: HttpRequest) -> Response:
    """Latest onboarding crawl for polling (ranked keywords up to ONBOARDING_RANKED_KEYWORDS_LIMIT)."""
    domain_param = (request.GET.get("domain") or "").strip().lower()
    base = OnboardingOnPageCrawl.objects.filter(user=request.user)
    if domain_param:
        crawl = base.filter(domain=domain_param).order_by("-created_at").first()
    else:
        crawl = base.order_by("-created_at").first()
    if not crawl:
        return Response(
            {
                "id": None,
                "status": "none",
                "ranked_keywords": [],
                "review_topics": [],
                "review_topics_error": "",
                "domain": None,
            },
        )
    return Response(
        {
            "id": crawl.id,
            "status": crawl.status,
            "domain": crawl.domain,
            "ranked_keywords": crawl.ranked_keywords or [],
            "review_topics": crawl.review_topics or [],
            "review_topics_error": crawl.review_topics_error or "",
            "exit_reason": crawl.exit_reason,
            "ranked_keywords_error": crawl.ranked_keywords_error or "",
            "prompt_plan_status": crawl.prompt_plan_status,
            "prompt_plan_prompt_count": int(crawl.prompt_plan_prompt_count or 0),
            "prompt_plan_error": crawl.prompt_plan_error or "",
        },
    )


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([AllowAny])
def api_auth_login(request: HttpRequest) -> Response:
    """Email + password login; establishes Django session for the Next.js app."""
    email = (request.data.get("email") or "").strip()
    password = request.data.get("password") or ""
    if not email or not password:
        return Response({"error": "Email and password are required"}, status=400)
    user = User.objects.filter(email__iexact=email).first()
    if user is not None:
        if not user.check_password(password):
            return Response({"error": "Invalid email or password"}, status=400)
    else:
        user = authenticate(request, username=email, password=password)
        if user is None:
            return Response({"error": "Invalid email or password"}, status=400)
    if not user.is_active:
        return Response({"error": "Account is disabled"}, status=400)
    login(request, user)
    try:
        done = user_has_completed_full_onboarding(user)
    except Exception:
        logger.exception("[api_auth_login] onboarding check failed")
        done = False
    return Response({"ok": True, "redirect": "/app" if done else "/onboarding"})


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([AllowAny])
def api_auth_register(request: HttpRequest) -> Response:
    """Create a local user, main BusinessProfile, and session (SPA signup)."""
    email = (request.data.get("email") or "").strip().lower()
    password = request.data.get("password") or ""
    if not email or len(password) < 6:
        return Response(
            {"error": "Valid email and password (min 6 characters) required"},
            status=400,
        )
    if User.objects.filter(email__iexact=email).exists() or User.objects.filter(
        username__iexact=email
    ).exists():
        return Response({"error": "An account with this email already exists"}, status=400)
    user = User.objects.create_user(username=email, email=email, password=password)
    if not BusinessProfile.objects.filter(user=user).exists():
        BusinessProfile.objects.create(user=user, is_main=True)
    login(request, user)
    return Response({"ok": True, "redirect": "/onboarding"})


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

    # Align with auth/status: profile + keywords + saved AEO prompts + AEO responses + extractions.
    try:
        if user_has_completed_full_onboarding(user):
            next_url = frontend_base + "/app"
        elif not session_next:
            next_url = frontend_base + "/onboarding"
        # else keep ``next_url`` from session (set above)
    except Exception:
        logger.exception("[auth] onboarding check failed; sending to onboarding")
        next_url = frontend_base + "/onboarding"

    return redirect(next_url)


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
            business_profile=profile,
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

    - GET: returns the user's main business profile (creates one if missing).
    - PATCH/PUT: updates main profile fields; creates a main profile if it does not exist yet.

    Query param ``skip_heavy=1`` (GET and PATCH): omit DataForSEO and on-the-fly AEO readiness work
    in the serializer (returns null/empty SEO fields and a stub AEO bundle). Use during onboarding
    to avoid paid API calls when only updating basic fields.
    """
    profile_qs = BusinessProfile.objects.filter(user=request.user).prefetch_related(
        "tracked_competitors",
    )
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
        BusinessProfile.objects.filter(user=request.user, is_main=True)
        .prefetch_related("tracked_competitors")
        .first()
        or BusinessProfile.objects.filter(user=request.user)
        .prefetch_related("tracked_competitors")
        .first()
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


def _improvement_recommendations_for_prompt(
    prompt_key: str,
    response_ids: set[int],
    recs: list,
) -> list:
    """
    Map Phase-5 ``AEORecommendationRun.recommendations_json`` items to a single prompt row.

    Matches ``create_content`` by ``prompt`` text; ``acquire_citation`` by ``references.response_snapshot_id``
    belonging to this prompt's response snapshots.
    """
    out: list = []
    key = (prompt_key or "").strip()

    def _norm_text(s: str) -> str:
        return " ".join((s or "").split()).strip().lower()

    key_norm = _norm_text(key)
    seen_text: set[str] = set()
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        at = rec.get("action_type")
        matched = False
        refs = rec.get("references") if isinstance(rec.get("references"), dict) else {}
        matched_prompt_texts = refs.get("matched_prompt_texts")
        matched_response_snapshot_ids = refs.get("matched_response_snapshot_ids")

        # Precedence 1: explicit matched prompt membership.
        if isinstance(matched_prompt_texts, list) and matched_prompt_texts:
            norm_members = {_norm_text(str(x)) for x in matched_prompt_texts if str(x).strip()}
            if key_norm and key_norm in norm_members:
                matched = True
        # Precedence 2: explicit grouped response id membership.
        elif isinstance(matched_response_snapshot_ids, list) and matched_response_snapshot_ids:
            ids: set[int] = set()
            for x in matched_response_snapshot_ids:
                try:
                    v = int(x)
                except (TypeError, ValueError):
                    continue
                ids.add(v)
            if ids.intersection(response_ids):
                matched = True
        # Legacy behavior fallback.
        elif at == "create_content":
            p = (rec.get("prompt") or "").strip()
            if _norm_text(p) == key_norm:
                matched = True
        elif at == "acquire_citation":
            rid = refs.get("response_snapshot_id")
            try:
                rid_int = int(rid) if rid is not None else None
            except (TypeError, ValueError):
                rid_int = None
            if rid_int is not None and rid_int in response_ids:
                matched = True
        if not matched:
            continue
        body = (rec.get("nl_explanation") or rec.get("reason") or "").strip()
        if not body:
            continue
        body_key = _norm_text(body)
        if body_key in seen_text:
            continue
        seen_text.add(body_key)
        item = {
            "action_type": at,
            "priority": (rec.get("priority") or "medium"),
            "text": body,
        }
        src = rec.get("source")
        if isinstance(src, str) and src.strip():
            item["source"] = src.strip()
        out.append(item)
    return out


def _build_aeo_prompt_coverage_payload(profile: BusinessProfile) -> dict:
    """
    One row per monitored / seen prompt; per-platform (OpenAI / Gemini / Perplexity) cells from latest snapshot each.
    """
    from .aeo.aeo_extraction_utils import (
        brand_effectively_cited,
        citations_ranking_for_prompt_coverage,
        merge_citations_rankings_across_platform_cells,
        merged_target_url_position,
        unique_business_count_excluding_target,
    )
    from .models import AEOResponseSnapshot, AEORecommendationRun

    selected_prompt_count = len(
        [str(x).strip() for x in (profile.selected_aeo_prompts or []) if str(x).strip()]
    )
    profile_site = (getattr(profile, "website_url", None) or "").strip()
    profile_business_name = (getattr(profile, "business_name", None) or "").strip()

    responses = list(
        AEOResponseSnapshot.objects.filter(profile=profile).order_by("-created_at", "-id")
    )

    by_prompt: dict[str, list] = {}
    for resp in responses:
        key = (resp.prompt_text or "").strip()
        if not key:
            continue
        by_prompt.setdefault(key, []).append(resp)

    def _response_sort_key(x):
        c = x.created_at
        if c is None:
            return (datetime.min.replace(tzinfo=timezone.utc), x.id)
        if django_timezone.is_naive(c):
            c = django_timezone.make_aware(c, timezone.utc)
        return (c, x.id)

    def latest_snapshot_per_platform(rows: list) -> dict[str, object]:
        best: dict[str, object] = {}
        for r in sorted(rows, key=_response_sort_key, reverse=True):
            plat = str(r.platform or "").strip().lower()
            if plat not in _AEO_COVERAGE_PLATFORM_SET:
                continue
            if plat not in best:
                best[plat] = r
        return best

    def platform_cell(resp) -> dict:
        latest_ex = resp.extraction_snapshots.order_by("-created_at", "-id").first()
        competitors_count = 0
        cited = False
        ranking: list = []
        target_url_position = None
        if latest_ex is not None:
            cited = brand_effectively_cited(
                bool(latest_ex.brand_mentioned),
                latest_ex.competitors_json,
                tracked_website_url_or_domain=profile_site,
            )
            comps = latest_ex.competitors_json or []
            competitors_count = len(comps) if isinstance(comps, list) else 0
            ranking, target_url_position = citations_ranking_for_prompt_coverage(
                latest_ex.citations_json,
                latest_ex.competitors_json,
                tracked_website_url_or_domain=profile_site,
                brand_mentioned=bool(latest_ex.brand_mentioned),
                tracked_business_name=profile_business_name,
            )
        return {
            "has_data": True,
            "cited": cited,
            "competitors_cited": competitors_count,
            "response_created_at": resp.created_at.isoformat() if resp.created_at else None,
            "target_url_position": target_url_position,
            "citations_ranking": ranking,
        }

    empty_cell = {
        "has_data": False,
        "cited": None,
        "competitors_cited": None,
        "response_created_at": None,
        "target_url_position": None,
        "citations_ranking": [],
    }

    latest_reco = (
        AEORecommendationRun.objects.filter(profile=profile)
        .order_by("-created_at", "-id")
        .first()
    )
    reco_items: list = list(latest_reco.recommendations_json or []) if latest_reco else []

    ordered_keys: list[str] = []
    seen: set[str] = set()
    for raw in profile.selected_aeo_prompts or []:
        k = str(raw).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        ordered_keys.append(k)
    for k in sorted(by_prompt.keys()):
        if k not in seen:
            ordered_keys.append(k)

    prompts: list[dict] = []
    for key in ordered_keys:
        rows = by_prompt.get(key, [])
        response_ids = {int(r.id) for r in rows if getattr(r, "id", None) is not None}
        improvement_recommendations = _improvement_recommendations_for_prompt(
            key, response_ids, reco_items
        )
        plat_latest = latest_snapshot_per_platform(rows)
        platforms = {
            "openai": platform_cell(plat_latest["openai"]) if "openai" in plat_latest else dict(empty_cell),
            "gemini": platform_cell(plat_latest["gemini"]) if "gemini" in plat_latest else dict(empty_cell),
            "perplexity": platform_cell(plat_latest["perplexity"]) if "perplexity" in plat_latest else dict(empty_cell),
        }
        any_row = next(iter(rows), None)
        prompt_type = ""
        if any_row is not None:
            prompt_type = str(any_row.prompt_type or "")
        o_cell = platforms["openai"]
        g_cell = platforms["gemini"]
        p_cell = platforms["perplexity"]
        pos_candidates: list[int] = []
        if o_cell.get("has_data") and o_cell.get("cited") and o_cell.get("target_url_position") is not None:
            pos_candidates.append(int(o_cell["target_url_position"]))
        if g_cell.get("has_data") and g_cell.get("cited") and g_cell.get("target_url_position") is not None:
            pos_candidates.append(int(g_cell["target_url_position"]))
        if p_cell.get("has_data") and p_cell.get("cited") and p_cell.get("target_url_position") is not None:
            pos_candidates.append(int(p_cell["target_url_position"]))
        target_url_position_best = min(pos_candidates) if pos_candidates else None

        merged_ranking = merge_citations_rankings_across_platform_cells([o_cell, g_cell, p_cell])
        target_from_merged = merged_target_url_position(merged_ranking)
        combined_ranking = merged_ranking
        combined_target = (
            target_from_merged if target_from_merged is not None else target_url_position_best
        )
        comp_other_businesses = unique_business_count_excluding_target(merged_ranking)

        prompts.append(
            {
                "prompt": key,
                "prompt_type": prompt_type,
                "competitors_cited": comp_other_businesses,
                "platforms": platforms,
                "target_url_position": combined_target,
                "citations_ranking": combined_ranking,
                "improvement_recommendations": improvement_recommendations,
            }
        )

    tracked_name = profile_business_name
    return {
        "prompts": prompts,
        "monitored_count": selected_prompt_count,
        "tracked_business_name": tracked_name,
    }


def _aeo_profile_visibility_pending(profile: BusinessProfile) -> bool:
    """
    True while prompt LLM responses or extraction snapshots are still in flight for monitored
    prompts — dashboard should show a loading state instead of partial visibility %.
    """
    from .models import AEOResponseSnapshot, AEOExecutionRun

    monitored = [str(x).strip() for x in (profile.selected_aeo_prompts or []) if str(x).strip()]
    if not monitored:
        return False

    if AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exists():
        return True

    latest_run = (
        AEOExecutionRun.objects.filter(profile=profile).order_by("-created_at", "-id").first()
    )
    if latest_run is not None and latest_run.status == AEOExecutionRun.STATUS_COMPLETED:
        if latest_run.extraction_status in (
            AEOExecutionRun.STAGE_PENDING,
            AEOExecutionRun.STAGE_RUNNING,
        ):
            return True

    responses = list(
        AEOResponseSnapshot.objects.filter(profile=profile)
        .order_by("-created_at", "-id")
        .prefetch_related("extraction_snapshots")
    )
    by_prompt: dict[str, list] = {}
    for resp in responses:
        key = (resp.prompt_text or "").strip()
        if not key:
            continue
        by_prompt.setdefault(key, []).append(resp)

    def _response_sort_key(x):
        c = x.created_at
        if c is None:
            return (datetime.min.replace(tzinfo=timezone.utc), x.id)
        if django_timezone.is_naive(c):
            c = django_timezone.make_aware(c, timezone.utc)
        return (c, x.id)

    def latest_snapshot_per_platform(rows: list) -> dict[str, object]:
        best: dict[str, object] = {}
        for r in sorted(rows, key=_response_sort_key, reverse=True):
            plat = str(r.platform or "").strip().lower()
            if plat not in _AEO_COVERAGE_PLATFORM_SET:
                continue
            if plat not in best:
                best[plat] = r
        return best

    for key in monitored:
        rows = by_prompt.get(key, [])
        plat_latest = latest_snapshot_per_platform(rows)
        for plat in _AEO_PENDING_PLATFORM_ORDER:
            if plat not in plat_latest:
                continue
            resp = plat_latest[plat]
            if len(resp.extraction_snapshots.all()) == 0:
                return True
    return False


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_prompt_coverage_data(request: HttpRequest) -> Response:
    """
    Cached-only prompt coverage read for AI Visibility UI.
    One row per unique prompt text; per-platform (OpenAI / Gemini / Perplexity) citation from latest snapshot each.
    """
    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"prompts": [], "monitored_count": 0, "tracked_business_name": ""})
    return Response(_build_aeo_prompt_coverage_payload(profile))


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_platform_visibility_data(request: HttpRequest) -> Response:
    """
    Per-LLM visibility % from extractions: share of prompts with a scan where the brand is cited.

    OpenAI (ChatGPT), Gemini, and Perplexity are computed from stored snapshots when present.
    Other platforms return 0% until integrations exist (frontend still shows them).
    """
    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"platforms": []})

    payload = _build_aeo_prompt_coverage_payload(profile)
    prompts = payload.get("prompts") or []

    def aggregate(api_key: str) -> tuple[int, int]:
        with_data = 0
        cited = 0
        for row in prompts:
            cell = (row.get("platforms") or {}).get(api_key) or {}
            if not cell.get("has_data"):
                continue
            with_data += 1
            if cell.get("cited"):
                cited += 1
        return with_data, cited

    def pct(with_data: int, cited_count: int) -> float:
        if with_data <= 0:
            return 0.0
        return round(100.0 * float(cited_count) / float(with_data), 1)

    o_w, o_c = aggregate("openai")
    p_w, p_c = aggregate("perplexity")
    g_w, g_c = aggregate("gemini")

    platforms_out = [
        {
            "key": "openai",
            "label": "ChatGPT",
            "visibility_pct": pct(o_w, o_c),
            "has_data": o_w > 0,
            "prompts_with_data": o_w,
            "prompts_cited": o_c,
            "has_backend": True,
        },
        {
            "key": "perplexity",
            "label": "Perplexity",
            "visibility_pct": pct(p_w, p_c),
            "has_data": p_w > 0,
            "prompts_with_data": p_w,
            "prompts_cited": p_c,
            "has_backend": True,
        },
        {
            "key": "gemini",
            "label": "Gemini",
            "visibility_pct": pct(g_w, g_c),
            "has_data": g_w > 0,
            "prompts_with_data": g_w,
            "prompts_cited": g_c,
            "has_backend": True,
        },
        {
            "key": "grok",
            "label": "Grok",
            "visibility_pct": 0.0,
            "has_data": False,
            "prompts_with_data": 0,
            "prompts_cited": 0,
            "has_backend": False,
        },
    ]
    visibility_pending = _aeo_profile_visibility_pending(profile)
    return Response({"platforms": platforms_out, "visibility_pending": visibility_pending})


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_share_of_voice_data(request: HttpRequest) -> Response:
    """
    Aggregated AI share-of-voice from latest AEO extraction per response snapshot:
    your mention units vs competitor name mentions (top 3 names + Others).
    """
    from .aeo.aeo_scoring_utils import aggregate_aeo_share_of_voice

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response(
            {
                "total_prompts": 0,
                "total_mention_units": 0,
                "your_mention_units": 0,
                "competitor_mention_units": 0,
                "has_data": False,
                "rows": [],
            }
        )
    payload = aggregate_aeo_share_of_voice(profile)
    return Response(payload)


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_onboarding_competitors_data(request: HttpRequest) -> Response:
    """
    Onboarding Search Analytics: up to 3 brands (top 2 competitors + target) with visibility %
    from AEO extraction snapshots across all LLM platforms.
    """
    from .aeo.aeo_scoring_utils import aeo_onboarding_competitors_visibility

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"has_data": False, "total_prompts": 0, "rows": []})
    return Response(aeo_onboarding_competitors_visibility(profile))


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_competitors_data(request: HttpRequest) -> Response:
    """
    Competitors page payload (tracked + suggested) using cached competitor snapshots.
    Lazy-computes the snapshot when absent for the requested scope.
    """
    from .aeo.competitor_snapshots import compute_and_save_competitor_snapshot

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True)
        .prefetch_related("tracked_competitors")
        .first()
        or BusinessProfile.objects.filter(user=request.user)
        .prefetch_related("tracked_competitors")
        .first()
    )
    if not profile:
        return Response(
            {
                "tracked_competitors": [],
                "suggested_competitors": [],
                "total_slots": 0,
                "snapshot_updated_at": None,
                "has_data": False,
            },
        )

    platform_scope = str(request.GET.get("platform", "all") or "all").strip().lower() or "all"
    window_days_raw = str(request.GET.get("window_days", "") or "").strip()
    window_start = None
    window_end = None
    if window_days_raw:
        try:
            days = max(1, int(window_days_raw))
            window_end = django_timezone.now()
            window_start = window_end - timedelta(days=days)
        except (TypeError, ValueError):
            window_start = None
            window_end = None

    snapshot = (
        AEOCompetitorSnapshot.objects.filter(
            profile=profile,
            platform_scope=platform_scope,
            window_start=window_start,
            window_end=window_end,
        )
        .order_by("-updated_at")
        .first()
    )
    if snapshot is None:
        try:
            snapshot = compute_and_save_competitor_snapshot(
                profile,
                platform_scope=platform_scope,
                window_start=window_start,
                window_end=window_end,
            )
        except Exception:
            logger.exception(
                "[aeo_competitors_data] lazy snapshot compute failed profile_id=%s",
                profile.id,
            )
            snapshot = None

    rows = list(getattr(snapshot, "rows_json", []) or []) if snapshot is not None else []
    by_domain: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        dom = str(row.get("domain") or "").strip().lower()
        if not dom:
            continue
        by_domain[dom] = row

    tracked = []
    tracked_domains: set[str] = set()
    for c in profile.tracked_competitors.order_by("domain").all():
        dom = normalize_tracked_competitor_domain(c.domain or "") or ""
        if dom:
            tracked_domains.add(dom)
        stats = by_domain.get(dom, {})
        tracked.append(
            {
                "id": c.id,
                "name": c.name,
                "domain": c.domain,
                "appearances": int(stats.get("appearances") or 0) if stats else 0,
                "visibility_pct": float(stats.get("visibility_pct") or 0.0) if stats else 0.0,
                "rank": int(stats.get("rank") or 0) if stats else None,
                "last_seen_at": stats.get("last_seen_at") if stats else None,
            },
        )

    suggested = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dom = str(row.get("domain") or "").strip().lower()
        if not dom or dom in tracked_domains:
            continue
        suggested.append(
            {
                "domain": dom,
                "display_name": str(row.get("display_name") or dom),
                "appearances": int(row.get("appearances") or 0),
                "visibility_pct": float(row.get("visibility_pct") or 0.0),
                "rank": int(row.get("rank") or 0),
                "last_seen_at": row.get("last_seen_at"),
            },
        )
        if len(suggested) >= 5:
            break

    total_slots = int(getattr(snapshot, "total_slots", 0) or 0) if snapshot is not None else 0
    return Response(
        {
            "tracked_competitors": tracked,
            "suggested_competitors": suggested,
            "total_slots": total_slots,
            "snapshot_updated_at": snapshot.updated_at.isoformat() if snapshot else None,
            "has_data": total_slots > 0,
        },
    )


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_pipeline_status_data(request: HttpRequest) -> Response:
    """
    Cached-only AEO pipeline status endpoint.
    No OpenAI/DataForSEO calls are made in this read path.
    """
    from .aeo.aeo_scoring_utils import composite_aeo_score_from_snapshot
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
                "aeo_score": composite_aeo_score_from_snapshot(latest_score),
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
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_pass_count_analytics_data(request: HttpRequest) -> Response:
    """Staff-only analytics for provider pass counts and conditional third-pass stability."""
    if not bool(getattr(request.user, "is_staff", False)):
        return Response({"detail": "Forbidden"}, status=403)
    from .third_party_usage import build_aeo_pass_count_analytics_context

    run_raw = request.GET.get("run_id")
    profile_raw = request.GET.get("profile_id")
    run_id: int | None = None
    profile_id: int | None = None
    try:
        if run_raw not in (None, "", "all"):
            run_id = int(str(run_raw))
    except (TypeError, ValueError):
        run_id = None
    try:
        if profile_raw not in (None, "", "all"):
            profile_id = int(str(profile_raw))
    except (TypeError, ValueError):
        profile_id = None
    payload = build_aeo_pass_count_analytics_context(
        execution_run_id=run_id,
        profile_id=profile_id,
    )
    return Response(payload)


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


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_refresh_execution(request: HttpRequest) -> Response:
    """
    Re-run Phase 2 (LLM answers) and downstream extraction/scoring for prompts already saved on the profile.

    JSON body: ``{"platform": "openai" | "gemini" | "perplexity"}`` — refresh one provider only.
    Does not regenerate the prompt list (onboarding / OpenAI prompt planning unchanged).
    """
    from .models import AEOExecutionRun
    from .tasks import _aeo_target_prompt_count, run_aeo_phase1_execution_task

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"detail": "No main business profile is configured."}, status=400)

    body = request.data if isinstance(request.data, dict) else {}
    platform = str(body.get("platform") or "").strip().lower()
    if platform not in ("openai", "gemini", "perplexity"):
        return Response(
            {"detail": 'Invalid or missing "platform"; expected "openai", "gemini", or "perplexity".'},
            status=400,
        )

    if platform == "gemini" and not gemini_execution_enabled():
        return Response(
            {"detail": "Gemini is not configured (set GEMINI_API_KEY)."},
            status=400,
        )

    if platform == "perplexity" and not perplexity_execution_enabled():
        return Response(
            {"detail": "Perplexity is not configured (set PERPLEXITY_API_KEY)."},
            status=400,
        )

    saved = profile.selected_aeo_prompts or []
    if not isinstance(saved, list) or not any(str(x).strip() for x in saved):
        return Response(
            {"detail": "No saved AEO prompts to refresh. Complete onboarding first."},
            status=400,
        )

    raw_items = plan_items_from_saved_prompt_strings([str(x) for x in saved])
    if not raw_items:
        return Response({"detail": "Could not build prompt list from saved prompts."}, status=400)

    target = _aeo_target_prompt_count()
    selected = raw_items[:target]

    inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exists()
    if inflight:
        return Response(
            {"detail": "An AEO run is already in progress. Try again when it finishes."},
            status=409,
        )

    providers = {
        "openai": ["openai"],
        "gemini": ["gemini"],
        "perplexity": ["perplexity"],
    }[platform]

    run = AEOExecutionRun.objects.create(
        profile=profile,
        prompt_count_requested=len(selected),
        status=AEOExecutionRun.STATUS_PENDING,
    )
    transaction.on_commit(
        lambda rid=run.id, payload=selected, prov=providers: run_aeo_phase1_execution_task.delay(
            rid,
            payload,
            providers=prov,
            force_refresh=True,
        )
    )

    logger.info(
        "[AEO refresh_execution] user_id=%s profile_id=%s platform=%s run_id=%s prompts=%s",
        getattr(request.user, "id", None),
        profile.id,
        platform,
        run.id,
        len(selected),
    )

    return Response(
        {
            "run_id": run.id,
            "platform": platform,
            "prompt_count": len(selected),
            "status": "pending",
        },
        status=202,
    )


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def refresh_aeo_gemini(request: HttpRequest) -> Response:
    """
    Dedicated Gemini-only AEO refresh endpoint.

    Reuses prompt strings stored in BusinessProfile.selected_aeo_prompts and enqueues
    the Gemini-only execution/extraction path.
    """
    from .models import AEOExecutionRun
    from .tasks import run_aeo_gemini_refresh_task

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"detail": "No main business profile is configured."}, status=400)

    if not gemini_execution_enabled():
        return Response({"detail": "Gemini is not configured (set GEMINI_API_KEY)."}, status=400)

    saved = profile.selected_aeo_prompts or []
    if not isinstance(saved, list):
        saved = []
    saved_nonempty = [str(x).strip() for x in saved if str(x).strip()]
    if not saved_nonempty:
        return Response(
            {"detail": "No saved AEO prompts to refresh. Complete onboarding first."},
            status=400,
        )

    target = _aeo_prompt_target_count()
    selected = plan_items_from_saved_prompt_strings(saved_nonempty)[:target]
    if not selected:
        return Response({"detail": "Could not build prompt list from saved prompts."}, status=400)

    marker = "refresh_provider=gemini"
    inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
        error_message__icontains=marker,
    ).exists()
    if inflight:
        return Response(
            {"detail": "A Gemini refresh is already in progress for this profile."},
            status=409,
        )

    run = AEOExecutionRun.objects.create(
        profile=profile,
        prompt_count_requested=len(selected),
        status=AEOExecutionRun.STATUS_PENDING,
        error_message=marker,
    )
    transaction.on_commit(
        lambda rid=run.id, payload=selected: run_aeo_gemini_refresh_task.delay(rid, payload)
    )

    logger.info(
        "[AEO refresh_gemini] provider=gemini run_id=%s profile_id=%s prompts=%s status=pending",
        run.id,
        profile.id,
        len(selected),
    )
    return Response(
        {
            "run_id": run.id,
            "status": "pending",
            "provider": "gemini",
            "prompt_count_requested": len(selected),
            "message": "Gemini refresh queued.",
        },
        status=202,
    )


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def refresh_aeo_perplexity(request: HttpRequest) -> Response:
    """
    Dedicated Perplexity-only AEO refresh endpoint (same pattern as refresh-gemini).
    """
    from .models import AEOExecutionRun
    from .tasks import run_aeo_perplexity_refresh_task

    profile = (
        BusinessProfile.objects.filter(user=request.user, is_main=True).first()
        or BusinessProfile.objects.filter(user=request.user).first()
    )
    if not profile:
        return Response({"detail": "No main business profile is configured."}, status=400)

    if not perplexity_execution_enabled():
        return Response({"detail": "Perplexity is not configured (set PERPLEXITY_API_KEY)."}, status=400)

    saved = profile.selected_aeo_prompts or []
    if not isinstance(saved, list):
        saved = []
    saved_nonempty = [str(x).strip() for x in saved if str(x).strip()]
    if not saved_nonempty:
        return Response(
            {"detail": "No saved AEO prompts to refresh. Complete onboarding first."},
            status=400,
        )

    target = _aeo_prompt_target_count()
    selected = plan_items_from_saved_prompt_strings(saved_nonempty)[:target]
    if not selected:
        return Response({"detail": "Could not build prompt list from saved prompts."}, status=400)

    marker = "refresh_provider=perplexity"
    inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
        error_message__icontains=marker,
    ).exists()
    if inflight:
        return Response(
            {"detail": "A Perplexity refresh is already in progress for this profile."},
            status=409,
        )

    run = AEOExecutionRun.objects.create(
        profile=profile,
        prompt_count_requested=len(selected),
        status=AEOExecutionRun.STATUS_PENDING,
        error_message=marker,
    )
    transaction.on_commit(
        lambda rid=run.id, payload=selected: run_aeo_perplexity_refresh_task.delay(rid, payload)
    )

    logger.info(
        "[AEO refresh_perplexity] provider=perplexity run_id=%s profile_id=%s prompts=%s status=pending",
        run.id,
        profile.id,
        len(selected),
    )
    return Response(
        {
            "run_id": run.id,
            "status": "pending",
            "provider": "perplexity",
            "prompt_count_requested": len(selected),
            "message": "Perplexity refresh queued.",
        },
        status=202,
    )


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


def _onboarding_skip_execution(request: HttpRequest, body: dict) -> bool:
    """When true with reuse_saved, return saved prompts without enqueueing phase 1."""
    v = body.get("skip_execution") if isinstance(body, dict) else None
    if isinstance(v, bool):
        return v
    if v is not None:
        return _truthy_openai_param(str(v))
    return _truthy_openai_param(request.GET.get("skip_execution"))


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


def _normalize_onboarding_topic_details(
    selected_topics: list[str],
    topic_details_raw: object,
) -> list[dict[str, object]]:
    """Merge client topic rows with optional Labs/AEO metadata keyed by keyword."""
    by_kw: dict[str, dict[str, object]] = {}
    if isinstance(topic_details_raw, list):
        for item in topic_details_raw:
            if isinstance(item, dict):
                kw = str(item.get("keyword") or "").strip()
                if kw:
                    by_kw[kw.casefold()] = item
    out: list[dict[str, object]] = []
    for t in selected_topics:
        t = str(t).strip()
        if not t:
            continue
        base: dict[str, object] = {"keyword": t}
        extra = by_kw.get(t.casefold())
        if extra:
            for key in ("search_volume", "rank", "rank_group", "aeo_score", "aeo_category", "aeo_reason"):
                if key in extra and extra[key] is not None and str(extra[key]).strip() != "":
                    base[key] = extra[key]
        out.append(base)
    return out


def _saved_prompts_domain_matches_onboarding_context(
    profile: BusinessProfile,
    website_url_from_context: str,
) -> bool:
    """True when stored profile URL and onboarding context URL normalize to the same domain."""
    prof = normalize_domain(profile.website_url or "") or ""
    ctx = normalize_domain(website_url_from_context or "") or ""
    if not prof or not ctx:
        return False
    return prof == ctx


def _assign_saved_prompts_to_selected_topics(
    prompt_texts: list[str],
    topics: list[str],
) -> dict[str, list[str]]:
    """Round-robin assign flat saved prompts onto topic tabs (same shape as onboarding UI)."""
    topics_list = [str(t).strip() for t in topics if str(t).strip()]
    out: dict[str, list[str]] = {t: [] for t in topics_list}
    if not topics_list:
        return out
    clean = [str(p).strip() for p in prompt_texts if str(p).strip()]
    for i, p in enumerate(clean):
        out[topics_list[i % len(topics_list)]].append(p)
    return out


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
    - ``skip_execution`` — with ``reuse_saved``, do not enqueue AEO phase 1 (read-only for onboarding resume).

    **Onboarding step 2 → prompts** (POST JSON): set ``onboarding_step2_prompt_plan`` to ``true`` and send
    ``selected_topics`` (strings), optional ``topic_details`` (per-keyword metadata), and
    ``onboarding_context`` with ``business_name``, ``website_url``, ``location``, ``language``.
    If the profile already has a full ``selected_aeo_prompts`` list for the **same** normalized domain as
    ``website_url`` in context, returns those prompts (with ``prompts_by_topic``) without OpenAI or
    DataForSEO. Otherwise builds a new plan via OpenAI.

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
    skip_execution = _onboarding_skip_execution(request, body)

    onboarding_step2 = bool(body.get("onboarding_step2_prompt_plan"))
    selected_topics_raw = body.get("selected_topics")

    target_prompt_count = _aeo_prompt_target_count()
    saved_raw = list(profile.selected_aeo_prompts or [])

    selected_topics: list[str] = []
    if onboarding_step2:
        if not isinstance(selected_topics_raw, list):
            return Response(
                {"error": "selected_topics must be a non-empty list when onboarding_step2_prompt_plan is set."},
                status=400,
            )
        selected_topics = [str(s).strip() for s in selected_topics_raw if str(s).strip()]
        if not selected_topics:
            return Response(
                {"error": "selected_topics must be a non-empty list when onboarding_step2_prompt_plan is set."},
                status=400,
            )
        oc_early = body.get("onboarding_context") if isinstance(body.get("onboarding_context"), dict) else {}
        website_url_ctx = str(oc_early.get("website_url") or "").strip()
        if (
            len(saved_raw) == target_prompt_count
            and _saved_prompts_domain_matches_onboarding_context(profile, website_url_ctx)
        ):
            raw_items = plan_items_from_saved_prompt_strings([str(x) for x in saved_raw])
            if len(raw_items) == target_prompt_count:
                ser = _serialize_aeo_prompt_items(raw_items)
                prompt_texts = [
                    str(x.get("prompt") or "").strip() for x in ser if str(x.get("prompt") or "").strip()
                ]
                if len(prompt_texts) == target_prompt_count:
                    pbt = _assign_saved_prompts_to_selected_topics(prompt_texts, selected_topics)
                    if all(len(pbt.get(t, [])) > 0 for t in selected_topics):
                        business_input = aeo_business_input_from_onboarding_payload(
                            business_name=str(oc_early.get("business_name") or ""),
                            website_url=website_url_ctx,
                            location=str(oc_early.get("location") or ""),
                            language=str(oc_early.get("language") or ""),
                            selected_topics=selected_topics,
                        )
                        return Response(
                            {
                                "groups": {
                                    "fixed": [],
                                    "dynamic": [],
                                    "openai_generated": [],
                                    "saved": ser,
                                },
                                "combined": ser,
                                "business": business_input.as_dict(),
                                "meta": {
                                    "openai_status": "reused_saved",
                                    "openai_message": "",
                                    "openai_prompt_count": 0,
                                    "combined_count": len(ser),
                                    "combined_target": target_prompt_count,
                                    "combined_shortfall": 0,
                                },
                                "prompts_by_topic": pbt,
                            }
                        )

    if onboarding_step2:
        reuse_saved = False

    if reuse_saved and isinstance(saved_raw, list) and len(saved_raw) == target_prompt_count:
        raw_items = plan_items_from_saved_prompt_strings([str(x) for x in saved_raw])
        if len(raw_items) == target_prompt_count:
            ser = _serialize_aeo_prompt_items(raw_items)
            biz = aeo_business_input_from_profile(
                profile,
                city=city_override,
                industry=industry_override,
            ).as_dict()
            if not skip_execution:
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
                    "prompts_by_topic": {},
                }
            )

    if onboarding_step2:
        oc = body.get("onboarding_context") if isinstance(body.get("onboarding_context"), dict) else {}
        details_norm = _normalize_onboarding_topic_details(selected_topics, body.get("topic_details"))
        business_input = aeo_business_input_from_onboarding_payload(
            business_name=str(oc.get("business_name") or ""),
            website_url=str(oc.get("website_url") or ""),
            location=str(oc.get("location") or ""),
            language=str(oc.get("language") or ""),
            selected_topics=selected_topics,
        )
        plan = build_full_aeo_prompt_plan(
            profile,
            business_input=business_input,
            onboarding_topic_details=details_norm,
            include_openai=include_openai,
            target_combined_count=target_prompt_count,
        )
    else:
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

    prompts_by_topic = plan.get("prompts_by_topic") or {}
    if not isinstance(prompts_by_topic, dict):
        prompts_by_topic = {}

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
            "prompts_by_topic": prompts_by_topic,
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
        profiles = (
            BusinessProfile.objects.filter(user=request.user)
            .prefetch_related("tracked_competitors")
            .order_by("-is_main", "created_at", "id")
        )
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
        from accounts.third_party_usage import usage_profile_context

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
            with usage_profile_context(profile):
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
                    business_profile=profile,
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
                    business_profile=profile,
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
        profile = BusinessProfile.objects.prefetch_related("tracked_competitors").get(
            id=pk,
            user=request.user,
        )
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
@permission_classes([AllowAny])
def api_logout(request: HttpRequest) -> Response:
    """
    Log out the current user from the Django session (Google SSO).
    """
    if request.user.is_authenticated:
        django_logout(request)
    return Response({"success": True})

