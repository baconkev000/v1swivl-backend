import json
import logging
import secrets
from typing import Optional
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

from .business_profile_access import (
    accessible_business_profiles_queryset,
    attach_organization_for_additional_profile,
    ensure_organization_for_first_owned_profile,
    get_business_profile_for_user,
    get_membership,
    remove_organization_membership_for_main_team_leave,
    resolve_main_business_profile_for_user_with_source,
    resolve_workspace_business_profile_for_request,
    resolve_workspace_business_profile_for_request_with_source,
    set_session_active_business_profile_for_user,
    should_create_owned_main_business_profile_for_user,
    sync_organization_membership_for_main_team_invite,
    viewer_team_access,
    workspace_data_user,
)
from .models import (
    ActionsGeneratedPageSnapshot,
    AEOCompetitorSnapshot,
    AEODashboardBundleCache,
    AgentActivityLog,
    BusinessProfile,
    BusinessProfileMembership,
    Organization,
    SEOOverviewSnapshot,
    AgentConversation,
    AgentMessage,
    OnboardingOnPageCrawl,
    ThirdPartyApiErrorLog,
)

# SEO overview cache: only refetch DataForSEO after snapshot TTL (7 days).
THIRD_PARTY_CACHE_TTL = timedelta(days=7)
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
    sort_top_keywords_for_display,
)
from .seo_snapshot_refresh import (
    run_full_seo_snapshot_for_profile,
    sync_enrich_current_period_seo_snapshot_for_profile,
)
from .constants import SEO_SNAPSHOT_TTL
from .onboarding_completion import (
    user_has_completed_full_onboarding,
    user_may_create_additional_business_profile,
)
from .user_identity_reconciliation import (
    authenticate_by_email_candidates,
    reconcile_user_identity_for_email,
)
from . import openai_utils
from . import debug_log as _debug
from .aeo.prompt_scan_progress import monitored_prompt_keys_in_order, prompt_scan_completed_count
from .aeo.aeo_plan_targets import (
    aeo_effective_custom_prompt_cap_for_profile,
    aeo_effective_monitored_target_for_profile,
    aeo_fallback_global_target_count,
    aeo_monitored_prompt_cap_for_plan_slug,
    aeo_onboarding_complete_min_prompts,
    aeo_should_run_post_payment_expansion,
)
from .aeo.aeo_utils import (
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
from .stripe_billing import normalize_stripe_payload
from .stripe_billing import extract_sync_debug_fields
from .stripe_billing import infer_sync_failure_reason
from .stripe_billing import mask_email
from .third_party_usage import record_third_party_api_error

logger = logging.getLogger(__name__)
_STRIPE_EVENT_SHAPE_LOGGED = False


def _stripe_get_scalar(obj, key: str, default=""):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def _stripe_get_nested(obj, *path, default=None):
    cur = obj
    for segment in path:
        if isinstance(segment, int):
            if not isinstance(cur, list) or segment < 0 or segment >= len(cur):
                return default
            cur = cur[segment]
            continue
        cur = _stripe_get_scalar(cur, segment, None)
        if cur is None:
            return default
    return cur


# Prompt coverage + pending checks: stored AEOResponseSnapshot.platform values we surface in the UI.
_AEO_COVERAGE_PLATFORM_SET = frozenset({"openai", "gemini", "perplexity"})
User = get_user_model()


def _reconcile_request_user_identity(request: HttpRequest, *, reason: str) -> None:
    user = getattr(request, "user", None)
    if user is None or not getattr(user, "is_authenticated", False):
        return
    result = reconcile_user_identity_for_email(
        getattr(user, "email", ""),
        preferred_user=user,
        reason=reason,
    )
    canonical = result.user
    if canonical is None or canonical.id == user.id:
        return
    backend = getattr(user, "backend", None) or getattr(canonical, "backend", None)
    if not backend:
        backends = list(getattr(settings, "AUTHENTICATION_BACKENDS", []) or [])
        backend = backends[0] if backends else None
    if backend:
        login(request, canonical, backend=backend)
    else:
        login(request, canonical)
    request.user = canonical
    logger.info(
        "[identity_reconcile] reason=%s switched_user_id=%s->%s reconciled=%s memberships=%s owned_profiles=%s",
        reason,
        user.id,
        canonical.id,
        result.reconciled,
        result.membership_count,
        result.owned_profile_count,
    )


def _money_minor_to_decimal_str(amount_minor: int | None) -> str:
    if amount_minor is None:
        return "0.00"
    return f"{(int(amount_minor) / 100):.2f}"


def _plan_label_from_slug(raw_plan: str) -> str:
    k = (raw_plan or "").strip().lower()
    if k == "pro":
        return "Pro"
    if k in {"advanced", "enterprise", "scale"}:
        return "Advanced"
    if k == "starter":
        return "Starter"
    return "Starter"


def _billing_payment_method_from_object(raw_pm: object) -> dict | None:
    """
    Extract safe card metadata from a Stripe PaymentMethod-like object.

    Returns only non-sensitive fields for API responses:
    ``brand``, ``last4``, ``exp_month``, ``exp_year``, optional ``funding``.
    """
    pm = normalize_stripe_payload(raw_pm)
    if not isinstance(pm, dict):
        return None
    card = pm.get("card")
    if not isinstance(card, dict):
        return None
    brand = str(card.get("brand") or "").strip().lower()
    last4 = str(card.get("last4") or "").strip()
    exp_month = card.get("exp_month")
    exp_year = card.get("exp_year")
    funding = str(card.get("funding") or "").strip().lower()
    try:
        em = int(exp_month)
        ey = int(exp_year)
    except (TypeError, ValueError):
        return None
    if not brand or not last4:
        return None
    out = {
        "brand": brand,
        "last4": last4[-4:],
        "exp_month": em,
        "exp_year": ey,
    }
    if funding:
        out["funding"] = funding
    return out


def _billing_payment_method_from_id(pm_id: object) -> dict | None:
    pid = str(pm_id or "").strip()
    if not pid:
        return None
    try:
        pm = stripe.PaymentMethod.retrieve(pid)
        return _billing_payment_method_from_object(pm)
    except Exception:
        logger.exception("[billing_summary] failed to retrieve payment method id=%s", pid)
        return None


def _billing_resolve_payment_method(
    *,
    customer_id: str,
    subscription_dict: dict | None,
    invoice_rows: list[dict] | None,
) -> dict | None:
    """
    Resolve card metadata in priority order:
    1) subscription default payment method
    2) customer invoice settings default payment method
    3) latest paid invoice's payment intent payment method
    """
    # 1) Subscription default payment method.
    sub_default_pm = (
        subscription_dict.get("default_payment_method")
        if isinstance(subscription_dict, dict)
        else None
    )
    if sub_default_pm:
        from_sub = (
            _billing_payment_method_from_object(sub_default_pm)
            if isinstance(sub_default_pm, dict)
            else _billing_payment_method_from_id(sub_default_pm)
        )
        if from_sub:
            return from_sub

    # 2) Customer invoice_settings.default_payment_method.
    try:
        cust_obj = stripe.Customer.retrieve(
            customer_id,
            expand=["invoice_settings.default_payment_method"],
        )
        cust = normalize_stripe_payload(cust_obj)
        default_pm = _stripe_get_nested(cust, "invoice_settings", "default_payment_method", default=None)
        if default_pm:
            from_customer = (
                _billing_payment_method_from_object(default_pm)
                if isinstance(default_pm, dict)
                else _billing_payment_method_from_id(default_pm)
            )
            if from_customer:
                return from_customer
    except Exception:
        logger.exception("[billing_summary] failed to retrieve Stripe customer id=%s", customer_id)

    # 3) Latest paid invoice payment intent payment method.
    rows = invoice_rows if isinstance(invoice_rows, list) else []
    paid_rows = [
        r for r in rows if isinstance(r, dict) and str(r.get("status") or "").strip().lower() == "paid"
    ]
    for raw in paid_rows:
        pi_ref = raw.get("payment_intent")
        if isinstance(pi_ref, dict):
            pm_candidate = pi_ref.get("payment_method")
            if pm_candidate:
                from_pi_obj = (
                    _billing_payment_method_from_object(pm_candidate)
                    if isinstance(pm_candidate, dict)
                    else _billing_payment_method_from_id(pm_candidate)
                )
                if from_pi_obj:
                    return from_pi_obj
            continue
        pi_id = str(pi_ref or "").strip()
        if not pi_id:
            continue
        try:
            pi_obj = stripe.PaymentIntent.retrieve(pi_id, expand=["payment_method"])
            pi = normalize_stripe_payload(pi_obj)
            pm = pi.get("payment_method") if isinstance(pi, dict) else None
            if pm:
                from_pi = (
                    _billing_payment_method_from_object(pm)
                    if isinstance(pm, dict)
                    else _billing_payment_method_from_id(pm)
                )
                if from_pi:
                    return from_pi
        except Exception:
            logger.exception(
                "[billing_summary] failed to retrieve Stripe payment intent id=%s",
                pi_id,
            )
    return None


def _monthly_price_from_price_obj(price_obj: dict) -> tuple[str, int | None]:
    """
    Return display and integer cents/month from Stripe price.
    """
    unit_amount = price_obj.get("unit_amount")
    recurring = price_obj.get("recurring") or {}
    if not isinstance(unit_amount, int) or unit_amount < 0:
        return "0.00", None
    interval = str(recurring.get("interval") or "").strip().lower()
    interval_count_raw = recurring.get("interval_count")
    interval_count = int(interval_count_raw) if isinstance(interval_count_raw, int) and interval_count_raw > 0 else 1
    monthly_minor = unit_amount
    if interval == "year":
        # Convert yearly to monthly equivalent.
        monthly_minor = round(unit_amount / float(12 * interval_count))
    elif interval == "month":
        monthly_minor = round(unit_amount / float(interval_count))
    return _money_minor_to_decimal_str(monthly_minor), int(monthly_minor)


def _subscription_rank_for_billing(sub: dict) -> tuple[int, int]:
    status = str(sub.get("status") or "").strip().lower()
    status_rank = {
        "active": 0,
        "trialing": 1,
        "past_due": 2,
        "incomplete": 3,
        "paused": 4,
        "canceled": 5,
        "unpaid": 6,
    }.get(status, 9)
    created = sub.get("created")
    created_int = int(created) if isinstance(created, int) else 0
    return status_rank, -created_int


def _price_obj_from_subscription_dict(subscription_dict: dict) -> dict:
    items = subscription_dict.get("items", {}).get("data", [])
    first_item = items[0] if isinstance(items, list) and items else {}
    if not isinstance(first_item, dict):
        return {}
    price_obj = first_item.get("price")
    if isinstance(price_obj, dict):
        return price_obj
    # Legacy Stripe payloads can still expose ``plan`` under items.
    plan_obj = first_item.get("plan")
    if isinstance(plan_obj, dict):
        return plan_obj
    return {}


def _safe_dt_from_unix(ts: int | None) -> datetime | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except Exception:
        return None


def _stripe_http_status_from_exception(exc: Exception) -> int | None:
    raw = getattr(exc, "http_status", None)
    if isinstance(raw, int):
        return raw
    try:
        parsed = int(raw)
        return parsed
    except Exception:
        return None


def _record_billing_stripe_error(
    *,
    billing_profile: BusinessProfile | None,
    operation: str,
    exc: Exception,
) -> None:
    name = type(exc).__name__
    http_status = _stripe_http_status_from_exception(exc)
    lname = name.lower()
    if "timeout" in lname:
        kind = ThirdPartyApiErrorLog.ErrorKind.TIMEOUT
    elif "connection" in lname:
        kind = ThirdPartyApiErrorLog.ErrorKind.CONNECTION_ERROR
    elif http_status is not None:
        kind = ThirdPartyApiErrorLog.ErrorKind.HTTP_ERROR
    else:
        kind = ThirdPartyApiErrorLog.ErrorKind.UNKNOWN_EXCEPTION
    record_third_party_api_error(
        provider="stripe",
        operation=operation,
        error_kind=kind,
        http_status=http_status,
        message=f"{name}: {str(exc)[:400]}",
        detail=f"billing_summary operation={operation}",
        business_profile=billing_profile,
    )


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
    and whether onboarding gates pass for app access.

    For account holders with multiple owned profiles, this is true if the resolved main
    profile is fully onboarded or any other owned profile is (so switching main to a
    profile still in progress does not force /onboarding).
    """
    if not request.user.is_authenticated:
        return Response({"authenticated": False, "onboarding_complete": False})
    _reconcile_request_user_identity(request, reason="auth_status")
    try:
        onboarding_complete = user_has_completed_full_onboarding(request.user)
    except Exception:
        logger.exception("[auth_status] business profile check failed")
        onboarding_complete = False
    profile, resolution_source = resolve_workspace_business_profile_for_request_with_source(request)
    logger.info(
        "[auth_status] user_id=%s onboarding_complete=%s resolved_profile_id=%s resolution_source=%s",
        request.user.id,
        onboarding_complete,
        getattr(profile, "id", None),
        resolution_source,
    )
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
        logger.error(
            "stripe.webhook.skipped reason_code=%s event_id=%s event_type=%s",
            "missing_required_env",
            "",
            "",
        )
        return Response({"error": "Stripe webhook is not configured."}, status=503)

    stripe.api_key = api_key
    sig = request.META.get("HTTP_STRIPE_SIGNATURE", "")
    payload = request.body
    try:
        event = stripe.Webhook.construct_event(payload, sig, secret)
    except ValueError:
        logger.error(
            "stripe.webhook.skipped reason_code=%s event_id=%s event_type=%s",
            "invalid_payload",
            "",
            "",
        )
        return Response({"error": "Invalid payload."}, status=400)
    except stripe.error.SignatureVerificationError:
        logger.error(
            "stripe.webhook.skipped reason_code=%s event_id=%s event_type=%s",
            "invalid_signature",
            "",
            "",
        )
        return Response({"error": "Invalid signature."}, status=400)

    global _STRIPE_EVENT_SHAPE_LOGGED
    if not _STRIPE_EVENT_SHAPE_LOGGED:
        _STRIPE_EVENT_SHAPE_LOGGED = True
        logger.info(
            "stripe.webhook.event_shape type=%s has_to_dict_recursive=%s has_id=%s has_type=%s repr=%s",
            str(type(event)),
            hasattr(event, "to_dict_recursive"),
            hasattr(event, "id"),
            hasattr(event, "type"),
            repr(event)[:500],
        )

    try:
        event_id = str(_stripe_get_scalar(event, "id", "") or "")
        event_type = str(_stripe_get_scalar(event, "type", "") or "")
        livemode = bool(_stripe_get_scalar(event, "livemode", False))
        api_version = str(_stripe_get_scalar(event, "api_version", "") or "")
        event_data = _stripe_get_scalar(event, "data", {})
        raw_data_obj = _stripe_get_scalar(event_data, "object", {})
        obj = normalize_stripe_payload(raw_data_obj)
        if not isinstance(obj, dict):
            obj = {
                "id": _stripe_get_scalar(raw_data_obj, "id", ""),
                "object": _stripe_get_scalar(raw_data_obj, "object", ""),
                "client_reference_id": _stripe_get_scalar(raw_data_obj, "client_reference_id", ""),
                "customer": _stripe_get_scalar(raw_data_obj, "customer", ""),
                "subscription": _stripe_get_scalar(raw_data_obj, "subscription", ""),
                "customer_details": normalize_stripe_payload(_stripe_get_scalar(raw_data_obj, "customer_details", {})),
                "customer_email": _stripe_get_scalar(raw_data_obj, "customer_email", ""),
                "receipt_email": _stripe_get_scalar(raw_data_obj, "receipt_email", ""),
                "payment_link": _stripe_get_scalar(raw_data_obj, "payment_link", ""),
                "invoice": _stripe_get_scalar(raw_data_obj, "invoice", ""),
                "lines": normalize_stripe_payload(_stripe_get_scalar(raw_data_obj, "lines", {})),
                "items": normalize_stripe_payload(_stripe_get_scalar(raw_data_obj, "items", {})),
                "price": _stripe_get_scalar(raw_data_obj, "price", ""),
                "status": _stripe_get_scalar(raw_data_obj, "status", ""),
                "current_period_end": _stripe_get_scalar(raw_data_obj, "current_period_end", None),
                "cancel_at_period_end": _stripe_get_scalar(raw_data_obj, "cancel_at_period_end", None),
            }
        event_dict = {
            "id": event_id,
            "type": event_type,
            "livemode": livemode,
            "api_version": api_version,
            "data": {"object": obj},
        }
    except Exception:
        logger.exception(
            "stripe.webhook.skipped reason_code=%s event_id=%s event_type=%s",
            "parse_failed",
            "",
            "",
        )
        return Response({"error": "Could not parse Stripe event."}, status=400)
    if not event_id or not event_type:
        logger.error(
            "stripe.webhook.skipped reason_code=%s event_id=%s event_type=%s top_level_keys=%s",
            "parse_failed",
            event_id,
            event_type,
            ",".join(sorted(event_dict.keys())),
        )
        return Response({"error": "Stripe event missing required fields."}, status=400)
    logger.info(
        "stripe.webhook.received event_id=%s event_type=%s livemode=%s api_version=%s request_path=%s has_signature_header=%s",
        event_id,
        event_type,
        livemode,
        api_version,
        request.path,
        bool(sig),
    )
    data = {"object": obj}
    details = _stripe_get_scalar(obj, "customer_details", {})
    details = details if isinstance(details, dict) else {}
    logger.info(
        "stripe.webhook.parsed event_id=%s event_type=%s livemode=%s object_id=%s object_type=%s client_reference_id=%s customer=%s subscription=%s customer_details_email=%s payment_link=%s invoice=%s top_level_keys=%s",
        event_id,
        event_type,
        livemode,
        str(_stripe_get_scalar(obj, "id") or ""),
        str(_stripe_get_scalar(obj, "object") or ""),
        str(_stripe_get_scalar(obj, "client_reference_id") or ""),
        str(_stripe_get_scalar(obj, "customer") or ""),
        str(_stripe_get_scalar(obj, "subscription") or ""),
        mask_email(str(_stripe_get_scalar(details, "email") or "")),
        str(_stripe_get_scalar(obj, "payment_link") or ""),
        str(_stripe_get_scalar(obj, "invoice") or ""),
        ",".join(sorted(event_dict.keys())),
    )
    dbg_identity = extract_sync_debug_fields(data, event_type=event_type, did_update=False)
    extracted_price_id = str(
        _stripe_get_scalar(obj, "price")
        or _stripe_get_nested(obj, "lines", "data", 0, "price", "id", default="")
        or _stripe_get_nested(obj, "items", "data", 0, "price", "id", default="")
        or ""
    )
    extracted_status = str(_stripe_get_scalar(obj, "status") or "")
    logger.info(
        "stripe.webhook.identity event_id=%s event_type=%s client_reference_id=%s customer_id=%s subscription_id=%s customer_details_email=%s payment_link_id=%s stripe_object_id=%s invoice_ref=%s",
        event_id,
        event_type,
        dbg_identity["client_reference_id"],
        dbg_identity["customer"],
        dbg_identity["subscription"],
        dbg_identity["email"],
        str(_stripe_get_scalar(obj, "payment_link") or ""),
        str(_stripe_get_scalar(obj, "id") or ""),
        str(_stripe_get_scalar(obj, "invoice") or ""),
    )
    handled = False
    result = None
    try:
        if event_type == "checkout.session.completed":
            result = sync_from_checkout_session(data, event_id=event_id)
            handled = result.handled
        elif event_type == "invoice.paid":
            result = sync_from_invoice_paid(data, event_id=event_id)
            handled = result.handled
        elif event_type in {"customer.subscription.updated", "customer.subscription.deleted"}:
            result = sync_from_subscription(data, event_id=event_id)
            handled = result.handled
        else:
            handled = True
    except Exception:
        logger.exception(
            "stripe.webhook.skipped reason_code=%s event_id=%s event_type=%s",
            "handler_exception",
            event_id,
            event_type,
        )
        return Response({"error": "Webhook handler failed."}, status=500)

    if result is not None:
        profile_email = None
        profile_user_id = None
        if result.matched_profile_id is not None:
            p = BusinessProfile.objects.filter(id=result.matched_profile_id).only("id", "user_id", "user__email").first()
            if p is not None:
                profile_user_id = p.user_id
                profile_email = mask_email(p.user.email if p.user else "")
        logger.info(
            "stripe.webhook.profile_resolution event_id=%s event_type=%s matched_profile_id=%s matched_by=%s profile_user_id=%s profile_email=%s",
            event_id,
            event_type,
            result.matched_profile_id if result.matched_profile_id is not None else "",
            result.matched_by or "none",
            profile_user_id if profile_user_id is not None else "",
            profile_email or "",
        )
        logger.info(
            "stripe.webhook.update_result event_id=%s event_type=%s did_update=%s updated_fields=%s stripe_customer_id_present=%s stripe_subscription_id_present=%s stripe_price_id_present=%s stripe_subscription_status_present=%s",
            event_id,
            event_type,
            result.did_update,
            ",".join(result.updated_fields) if result.updated_fields else "",
            bool(dbg_identity["customer"]),
            bool(dbg_identity["subscription"]),
            bool(extracted_price_id),
            bool(extracted_status),
        )
    if not handled:
        dbg = extract_sync_debug_fields(data, event_type=event_type, did_update=False)
        reason = (result.reason_code if result is not None and result.reason_code else None) or infer_sync_failure_reason(event_type, data)
        logger.error(
            "stripe.webhook.skipped reason_code=%s event_id=%s event_type=%s client_reference_id=%s customer=%s subscription=%s email=%s matched_profile_id=%s did_update=%s",
            reason,
            event_id,
            dbg["event_type"],
            dbg["client_reference_id"],
            dbg["customer"],
            dbg["subscription"],
            dbg["email"],
            dbg["matched_profile_id"],
            dbg["did_update"],
        )
    return Response({"received": True})


def _onboarding_domain_claimed_by_other_user(domain: str, user) -> bool:
    """True if any other user's business profile uses this normalized domain."""
    if not domain:
        return False
    for bp in BusinessProfile.objects.exclude(user=user).exclude(website_url="").only("id", "website_url"):
        if normalize_domain(bp.website_url or "") == domain:
            return True
    return False


def _onboarding_reusable_crawl_for_user(
    user,
    domain: str,
    *,
    business_profile_id: int | None = None,
) -> OnboardingOnPageCrawl | None:
    """Latest completed onboarding crawl for this user/domain with keywords and/or review topics."""
    if not domain:
        return None
    qs = OnboardingOnPageCrawl.objects.filter(
        user=user,
        domain=domain,
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
    ).order_by("-created_at")
    if business_profile_id is not None:
        qs = qs.filter(business_profile_id=int(business_profile_id))
    for crawl in qs[:5]:
        if crawl.ranked_keywords or crawl.review_topics:
            return crawl
    return None


def _clone_onboarding_crawl_for_profile(
    source: OnboardingOnPageCrawl,
    *,
    profile: BusinessProfile,
    context: dict,
) -> OnboardingOnPageCrawl:
    """
    Clone completed crawl payloads onto another profile so onboarding can reuse cached
    DataForSEO/Gemini results without re-hitting third-party APIs.
    """
    return OnboardingOnPageCrawl.objects.create(
        user=source.user,
        business_profile=profile,
        domain=source.domain,
        status=source.status,
        max_pages=source.max_pages,
        pages=source.pages or [],
        ranked_keywords=source.ranked_keywords or [],
        topic_clusters=source.topic_clusters or {},
        crawl_topic_seeds=source.crawl_topic_seeds or [],
        ranked_keywords_error=source.ranked_keywords_error or "",
        ranked_keywords_fetch_status=(
            OnboardingOnPageCrawl.RANKED_FETCH_COMPLETE
            if (source.ranked_keywords or [])
            else getattr(source, "ranked_keywords_fetch_status", "") or ""
        ),
        review_topics=source.review_topics or [],
        review_topics_error=source.review_topics_error or "",
        context=context,
        task_id=source.task_id or "",
        exit_reason=source.exit_reason or "",
        error_message=source.error_message or "",
        prompt_plan_status=source.prompt_plan_status,
        prompt_plan_prompt_count=int(source.prompt_plan_prompt_count or 0),
        prompt_plan_error=source.prompt_plan_error or "",
        prompt_plan_task_id=source.prompt_plan_task_id or "",
        prompt_plan_started_at=source.prompt_plan_started_at,
        prompt_plan_finished_at=source.prompt_plan_finished_at,
    )


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def onboarding_onpage_crawl_start(request: HttpRequest) -> Response:
    """
    Queue DataForSEO On-Page crawl (max 10 pages) after onboarding step 1.

    If this user already has a completed crawl with ranked keywords for the domain,
    returns that row without enqueueing a new crawl.
    """
    from .tasks import onboarding_onpage_crawl_task, onboarding_review_topics_backfill_task

    website_url = (request.data.get("website_url") or "").strip()
    business_name = (request.data.get("business_name") or "").strip()
    location = (request.data.get("location") or "").strip()
    profile_id_raw = request.data.get("profile_id")
    profile_id: int | None = None
    if profile_id_raw not in (None, ""):
        try:
            profile_id = int(profile_id_raw)
        except (TypeError, ValueError):
            return Response({"error": "profile_id must be an integer."}, status=400)
    customer_reach = str(request.data.get("customer_reach") or BusinessProfile.CUSTOMER_REACH_ONLINE).strip().lower()
    customer_reach_state = str(request.data.get("customer_reach_state") or "").strip()
    customer_reach_city = str(request.data.get("customer_reach_city") or "").strip()
    if customer_reach not in {
        BusinessProfile.CUSTOMER_REACH_ONLINE,
        BusinessProfile.CUSTOMER_REACH_LOCAL,
    }:
        customer_reach = BusinessProfile.CUSTOMER_REACH_ONLINE
    if (
        customer_reach == BusinessProfile.CUSTOMER_REACH_LOCAL
        and not customer_reach_state
    ):
        return Response(
            {"error": "customer_reach_state is required when customer_reach is local."},
            status=400,
        )
    if customer_reach != BusinessProfile.CUSTOMER_REACH_LOCAL:
        customer_reach_state = ""
        customer_reach_city = ""
    domain = normalize_domain(website_url)
    if not domain:
        return Response({"error": "A valid website_url is required"}, status=400)

    reused = _onboarding_reusable_crawl_for_user(
        request.user,
        domain,
        business_profile_id=profile_id,
    )
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
        if profile_id is not None:
            profile = BusinessProfile.objects.filter(
                id=profile_id,
                user=request.user,
            ).first()
            if profile is None:
                return Response(
                    {"error": "Business profile not found for this account."},
                    status=404,
                )
        else:
            profile = resolve_workspace_business_profile_for_request(request)
            if profile is None:
                if not should_create_owned_main_business_profile_for_user(request.user):
                    return Response(
                        {"error": "No business profile for this account. Ask an admin to add you to the team."},
                        status=403,
                    )
                profile = BusinessProfile.objects.create(user=request.user, is_main=True)
                BusinessProfileMembership.objects.get_or_create(
                    business_profile=profile,
                    user=request.user,
                    defaults={
                        "role": BusinessProfileMembership.ROLE_ADMIN,
                        "is_owner": True,
                    },
                )
        # Cross-profile cache reuse: if this profile has no prior crawl for the domain,
        # clone the latest completed crawl from another profile under the same account.
        cross_profile_reuse = _onboarding_reusable_crawl_for_user(
            request.user,
            domain,
            business_profile_id=None,
        )
        if (
            cross_profile_reuse is not None
            and int(getattr(cross_profile_reuse, "business_profile_id", 0) or 0) != int(profile.id)
            and (cross_profile_reuse.ranked_keywords or cross_profile_reuse.review_topics)
        ):
            cloned = _clone_onboarding_crawl_for_profile(
                cross_profile_reuse,
                profile=profile,
                context={
                    "business_name": business_name,
                    "location": location,
                    "customer_reach": customer_reach,
                    "customer_reach_state": customer_reach_state,
                    "customer_reach_city": customer_reach_city,
                },
            )
            return Response(
                {
                    "id": cloned.id,
                    "status": cloned.status,
                    "domain": domain,
                    "reused": True,
                    "ranked_keywords": cloned.ranked_keywords or [],
                    "review_topics": cloned.review_topics or [],
                    "review_topics_error": cloned.review_topics_error or "",
                    "prompt_plan_status": cloned.prompt_plan_status,
                    "prompt_plan_prompt_count": int(cloned.prompt_plan_prompt_count or 0),
                    "prompt_plan_error": cloned.prompt_plan_error or "",
                },
            )

        crawl = OnboardingOnPageCrawl.objects.create(
            user=request.user,
            business_profile=profile,
            domain=domain,
            max_pages=10,
            context={
                "business_name": business_name,
                "location": location,
                "customer_reach": customer_reach,
                "customer_reach_state": customer_reach_state,
                "customer_reach_city": customer_reach_city,
            },
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
    profile_id_param = (request.GET.get("profile_id") or "").strip()
    base = OnboardingOnPageCrawl.objects.filter(user=request.user)
    if profile_id_param:
        try:
            base = base.filter(business_profile_id=int(profile_id_param))
        except (TypeError, ValueError):
            return Response({"error": "profile_id must be an integer."}, status=400)
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
            "ranked_keywords_pending": getattr(
                crawl, "ranked_keywords_fetch_status", ""
            )
            == OnboardingOnPageCrawl.RANKED_FETCH_PENDING,
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
    authenticated_user = authenticate_by_email_candidates(email, password)
    if authenticated_user is None:
        return Response({"error": "Invalid email or password"}, status=400)
    reconcile = reconcile_user_identity_for_email(
        email,
        preferred_user=authenticated_user,
        reason="api_auth_login",
    )
    user = reconcile.user or authenticated_user
    if not user.is_active:
        return Response({"error": "Account is disabled"}, status=400)
    login(request, user)
    profile, resolution_source = resolve_main_business_profile_for_user_with_source(user)
    logger.info(
        "[api_auth_login] user_id=%s reconciled=%s merged_user_ids=%s membership_count=%s owned_profile_count=%s resolved_profile_id=%s resolution_source=%s",
        user.id,
        reconcile.reconciled,
        ",".join(str(x) for x in reconcile.merged_user_ids),
        reconcile.membership_count,
        reconcile.owned_profile_count,
        getattr(profile, "id", None),
        resolution_source,
    )
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
    existing_email_rows = list(User.objects.filter(email__iexact=email).order_by("id"))
    if existing_email_rows:
        placeholder = next((u for u in existing_email_rows if not u.has_usable_password()), None)
        if placeholder is None:
            return Response({"error": "An account with this email already exists"}, status=400)
        placeholder.set_password(password)
        if not (placeholder.username or "").strip():
            placeholder.username = email
            placeholder.save(update_fields=["password", "username"])
        else:
            placeholder.save(update_fields=["password"])
        reconcile = reconcile_user_identity_for_email(
            email,
            preferred_user=placeholder,
            reason="api_auth_register_placeholder",
        )
        user = reconcile.user or placeholder
    else:
        if User.objects.filter(username__iexact=email).exists():
            return Response({"error": "An account with this email already exists"}, status=400)
        user = User.objects.create_user(username=email, email=email, password=password)
    if should_create_owned_main_business_profile_for_user(user):
        bp = BusinessProfile.objects.create(user=user, is_main=True)
        BusinessProfileMembership.objects.get_or_create(
            business_profile=bp,
            user=user,
            defaults={
                "role": BusinessProfileMembership.ROLE_ADMIN,
                "is_owner": True,
            },
        )
    login(request, user)
    profile, resolution_source = resolve_main_business_profile_for_user_with_source(user)
    logger.info(
        "[api_auth_register] user_id=%s resolved_profile_id=%s resolution_source=%s memberships=%s owned_profiles=%s",
        user.id,
        getattr(profile, "id", None),
        resolution_source,
        BusinessProfileMembership.objects.filter(user=user).count(),
        BusinessProfile.objects.filter(user=user).count(),
    )
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

    Uses a 7-day snapshot cache: if we have fresh snapshot data, return it without calling DataForSEO.
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

    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response(
            {"detail": "No active business profile."},
            status=400,
        )
    profile_for_context = profile
    data_user = workspace_data_user(profile) or request.user
    snapshot_mode, snapshot_location_code = _seo_snapshot_context_for_profile(profile_for_context)

    # Serve from cache if we have a snapshot for this period fetched within the last hour (unless refresh=1).
    if not force_refresh:
        try:
            snapshot = SEOOverviewSnapshot.objects.get(
                business_profile=profile,
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
        domain_cache_key = f"seo_overview_domain:{profile.pk}"
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
                    business_profile=profile,
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
                business_profile=profile,
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
            business_profile=profile,
            period_start=start_current,
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
            defaults={"user": profile.user},
        )
        if snapshot.user_id != profile.user_id:
            snapshot.user = profile.user
            snapshot.save(update_fields=["user"])
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

    profile = resolve_workspace_business_profile_for_request(request)
    data_user = workspace_data_user(profile) or request.user
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
            business_profile=profile,
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
            data_user,
            site_url=site_url,
            force_refresh=force_refresh,
            business_profile=profile,
        )
        snapshot = (
            SEOOverviewSnapshot.objects.filter(
                business_profile=profile,
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
    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response({"detail": "No active business profile."}, status=404)
    snapshot_mode, snapshot_location_code = _seo_snapshot_context_for_profile(profile)

    snapshot = (
        SEOOverviewSnapshot.objects.filter(
            business_profile=profile,
            period_start=start_current,
            cached_location_mode=snapshot_mode,
            cached_location_code=snapshot_location_code,
        )
        .order_by("-last_fetched_at")
        .first()
    )
    if not snapshot:
        return Response({"detail": "No SEOOverviewSnapshot for this profile this month."}, status=404)

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
    _reconcile_request_user_identity(request, reason="business_profile")
    profile = resolve_workspace_business_profile_for_request(request)
    if profile is None:
        if not should_create_owned_main_business_profile_for_user(request.user):
            return Response({"error": "No active business profile."}, status=404)
        profile = BusinessProfile.objects.create(user=request.user, is_main=True)
        BusinessProfileMembership.objects.get_or_create(
            business_profile=profile,
            user=request.user,
            defaults={
                "role": BusinessProfileMembership.ROLE_ADMIN,
                "is_owner": True,
            },
        )

    profile = (
        BusinessProfile.objects.filter(pk=profile.pk)
        .prefetch_related("tracked_competitors")
        .first()
        or profile
    )

    access = viewer_team_access(request.user, profile)
    base_ctx = {"request": request, "viewer_access": access}

    if request.method == "GET":
        force_aeo_refresh = str(request.GET.get("refresh_aeo", "")).strip().lower() in {"1", "true", "yes"}
        skip_heavy = str(request.GET.get("skip_heavy", "")).strip().lower() in {"1", "true", "yes"}
        serializer = BusinessProfileSerializer(
            profile,
            context={
                **base_ctx,
                "force_aeo_refresh": force_aeo_refresh,
                "disable_seo_context_for_aeo": True,
                "skip_heavy_profile_metrics": skip_heavy,
            },
        )
        return Response(serializer.data)

    # For PATCH/PUT, apply partial updates
    if not access["viewer_can_edit_company_profile"]:
        return Response(
            {"detail": "You do not have permission to edit company settings."},
            status=403,
        )

    skip_heavy = str(request.GET.get("skip_heavy", "")).strip().lower() in {"1", "true", "yes"}
    old_site_url = profile.website_url
    serializer = BusinessProfileSerializer(
        profile,
        data=request.data,
        partial=True,
        context=base_ctx,
    )
    serializer.is_valid(raise_exception=True)
    serializer.save()

    # If the website URL changed, proactively refresh the SEO score snapshot
    # so that the frontend (and SEO agent) immediately see metrics for the new site.
    new_site_url = serializer.instance.website_url
    if new_site_url and new_site_url != old_site_url:
        try:
            inst = serializer.instance
            data_user = workspace_data_user(inst) or request.user
            run_full_seo_snapshot_for_profile(
                inst,
                data_user_fallback=data_user,
                abort_on_low_coverage=True,
            )
        except Exception:
            # Never block profile saves on DataForSEO errors.
            pass

    out = BusinessProfileSerializer(
        serializer.instance,
        context={
            **base_ctx,
            "disable_seo_context_for_aeo": True,
            "skip_heavy_profile_metrics": skip_heavy,
        },
    )
    return Response(out.data)


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def business_profile_set_active(request: HttpRequest) -> Response:
    """
    Choose which accessible BusinessProfile the app shell uses (session-scoped).

    Org-wide team members can switch to any site in the org without mutating ``is_main``.

    If the caller is ``Organization.owner_user`` for the profile's organization, we also
    move ``is_main`` to that site so billing / ordering stay aligned with the dropdown.
    """
    _reconcile_request_user_identity(request, reason="business_profile_set_active")
    body = getattr(request, "data", None) or {}
    if not isinstance(body, dict):
        body = {}
    raw = body.get("business_profile_id", body.get("id"))
    if raw is None:
        return Response({"detail": "business_profile_id is required."}, status=400)
    try:
        pk = int(raw)
    except (TypeError, ValueError):
        return Response({"detail": "Invalid business_profile_id."}, status=400)

    profile = get_business_profile_for_user(request.user, pk)
    if profile is None:
        return Response({"detail": "Not found."}, status=404)

    oid = getattr(profile, "organization_id", None)
    if oid:
        org = Organization.objects.filter(pk=int(oid)).first()
        if org is not None and int(org.owner_user_id) == int(request.user.id):
            BusinessProfile.objects.filter(organization_id=int(oid)).exclude(pk=profile.pk).update(
                is_main=False
            )
            BusinessProfile.objects.filter(pk=profile.pk).update(is_main=True)

    if not set_session_active_business_profile_for_user(request, request.user, int(profile.pk)):
        return Response({"detail": "Not found."}, status=404)

    return Response({"ok": True, "business_profile_id": int(profile.pk)})


@csrf_exempt
@api_view(["GET", "POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def business_profile_team(request: HttpRequest) -> Response:
    profile = resolve_workspace_business_profile_for_request(request)
    if profile is None:
        return Response({"error": "No active business profile."}, status=404)

    if request.method == "GET":
        m_self = get_membership(request.user, profile)
        if m_self is None and profile.user_id != request.user.id:
            return Response({"detail": "Forbidden."}, status=403)
        rows = (
            BusinessProfileMembership.objects.filter(
                business_profile=profile,
                hidden_from_team_ui=False,
            )
            .select_related("user")
            .order_by("-is_owner", "role", "id")
        )
        return Response(
            {
                "members": [
                    {
                        "user_id": r.user_id,
                        "email": r.user.email or "",
                        "role": r.role,
                        "is_owner": r.is_owner,
                        "pending_sign_in": not r.user.has_usable_password(),
                    }
                    for r in rows
                ],
            },
        )

    if not viewer_team_access(request.user, profile)["viewer_can_manage_team"]:
        return Response({"detail": "Forbidden."}, status=403)

    email = (request.data.get("email") or "").strip().lower()
    role_raw = (request.data.get("role") or "").strip().lower()
    if not email or "@" not in email:
        return Response({"error": "Valid email is required."}, status=400)
    if role_raw not in (BusinessProfileMembership.ROLE_ADMIN, BusinessProfileMembership.ROLE_MEMBER):
        return Response({"error": "role must be admin or member."}, status=400)

    owner = profile.user
    if owner and (owner.email or "").strip().lower() == email:
        return Response({"error": "That email is already the account owner."}, status=400)

    User = get_user_model()
    invited = User.objects.filter(email__iexact=email).first()
    if invited is None:
        invited = User(username=email, email=email)
        invited.set_unusable_password()
        invited.save()

    if BusinessProfileMembership.objects.filter(business_profile=profile, user=invited).exists():
        return Response({"error": "That user is already on this team."}, status=400)

    BusinessProfileMembership.objects.create(
        business_profile=profile,
        user=invited,
        role=(
            BusinessProfileMembership.ROLE_ADMIN
            if role_raw == BusinessProfileMembership.ROLE_ADMIN
            else BusinessProfileMembership.ROLE_MEMBER
        ),
        is_owner=False,
    )
    sync_organization_membership_for_main_team_invite(profile, invited, role_raw)
    return Response({"ok": True, "user_id": invited.id}, status=201)


@csrf_exempt
@api_view(["DELETE"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def business_profile_team_member(request: HttpRequest, user_id: int) -> Response:
    profile = resolve_workspace_business_profile_for_request(request)
    if profile is None:
        return Response({"error": "No active business profile."}, status=404)
    if not viewer_team_access(request.user, profile)["viewer_can_manage_team"]:
        return Response({"detail": "Forbidden."}, status=403)

    row = BusinessProfileMembership.objects.filter(
        business_profile=profile,
        user_id=int(user_id),
    ).first()
    if row is None:
        return Response({"error": "Not a team member."}, status=404)
    if row.is_owner:
        return Response({"error": "Cannot remove the primary account holder."}, status=400)
    removed_uid = int(row.user_id)
    row.delete()
    remove_organization_membership_for_main_team_leave(profile, removed_uid)
    return Response(status=204)


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def business_profile_checkout_identity(request: HttpRequest) -> Response:
    profile = resolve_workspace_business_profile_for_request(request)
    if profile is None:
        return Response({"error": "No active business profile."}, status=404)
    access = viewer_team_access(request.user, profile)
    if not access["viewer_can_access_billing"]:
        return Response({"error": "Billing is not available for this account."}, status=403)
    return Response(
        {
            "profile_id": profile.id,
            "user_id": request.user.id,
            "email": request.user.email,
            "is_main": bool(profile.is_main),
        },
    )


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def billing_summary(request: HttpRequest) -> Response:
    """
    Stripe-backed billing summary for the authenticated user's main business profile.
    """
    profile = resolve_workspace_business_profile_for_request(request)
    if profile is None:
        return Response({"error": "No active business profile."}, status=404)
    access = viewer_team_access(request.user, profile)
    if not access["viewer_can_access_billing"]:
        return Response({"error": "Billing is not available for this account."}, status=403)

    out = {
        "plan_label": _plan_label_from_slug(str(getattr(profile, "plan", "") or "")),
        "price_month": "0.00",
        "renewal_date": None,
        "currency": "usd",
        "invoices": [],
        "payment_method": None,
    }

    billing_profile = profile
    customer_id = str(getattr(billing_profile, "stripe_customer_id", "") or "").strip()
    subscription_id = str(getattr(billing_profile, "stripe_subscription_id", "") or "").strip()
    period_end = getattr(billing_profile, "stripe_current_period_end", None)
    if not customer_id:
        if getattr(billing_profile, "organization_id", None):
            owned_qs = BusinessProfile.objects.filter(
                organization_id=billing_profile.organization_id,
                user_id=billing_profile.user_id,
            )
        else:
            owned_qs = BusinessProfile.objects.filter(user=request.user)
        owned_with_customer = (
            owned_qs.exclude(pk=billing_profile.pk)
            .exclude(stripe_customer_id__isnull=True)
            .exclude(stripe_customer_id="")
            .order_by("-is_main", "id")
            .first()
        )
        if owned_with_customer is not None:
            billing_profile = owned_with_customer
            customer_id = str(getattr(billing_profile, "stripe_customer_id", "") or "").strip()
            subscription_id = str(getattr(billing_profile, "stripe_subscription_id", "") or "").strip()
            period_end = getattr(billing_profile, "stripe_current_period_end", None)
            out["plan_label"] = _plan_label_from_slug(str(getattr(billing_profile, "plan", "") or ""))
            logger.info(
                "[billing_summary] using owned profile fallback resolved_profile_id=%s billing_profile_id=%s user_id=%s",
                profile.id,
                billing_profile.id,
                request.user.id,
            )
    if period_end is not None:
        out["renewal_date"] = period_end.isoformat()

    secret = str(getattr(settings, "STRIPE_SECRET_KEY", "") or "").strip()
    if not secret or not customer_id:
        logger.info(
            "[billing_summary] skipped_stripe profile_id=%s billing_profile_id=%s has_secret=%s has_customer_id=%s",
            profile.id,
            billing_profile.id,
            bool(secret),
            bool(customer_id),
        )
        return Response(out)

    stripe.api_key = secret

    subscription_dict: dict | None = None
    subscription_retrieve_ok = False
    subscription_list_ok = False
    invoice_list_ok = False
    if subscription_id:
        try:
            subscription = stripe.Subscription.retrieve(
                subscription_id,
                expand=["items.data.price", "default_payment_method"],
            )
            nsub = normalize_stripe_payload(subscription)
            if isinstance(nsub, dict):
                subscription_dict = nsub
                subscription_retrieve_ok = True
        except Exception as exc:
            logger.exception("[billing_summary] failed to retrieve Stripe subscription id=%s", subscription_id)
            _record_billing_stripe_error(
                billing_profile=billing_profile,
                operation="stripe.subscription.retrieve.billing_summary",
                exc=exc,
            )

    if subscription_dict is None:
        try:
            listed = stripe.Subscription.list(
                customer=customer_id,
                status="all",
                limit=5,
                expand=["data.items.data.price", "data.default_payment_method"],
            )
            nlisted = normalize_stripe_payload(listed)
            rows = nlisted.get("data", []) if isinstance(nlisted, dict) else []
            if isinstance(rows, list) and rows:
                sub_rows = [r for r in rows if isinstance(r, dict)]
                if sub_rows:
                    sub_rows.sort(key=_subscription_rank_for_billing)
                    subscription_dict = sub_rows[0]
            subscription_list_ok = True
        except Exception as exc:
            logger.exception("[billing_summary] failed to list Stripe subscriptions customer=%s", customer_id)
            _record_billing_stripe_error(
                billing_profile=billing_profile,
                operation="stripe.subscription.list.billing_summary",
                exc=exc,
            )

    if isinstance(subscription_dict, dict):
        price_obj = _price_obj_from_subscription_dict(subscription_dict)
        if isinstance(price_obj, dict):
            monthly_price, _monthly_minor = _monthly_price_from_price_obj(price_obj)
            out["price_month"] = monthly_price
            out["currency"] = str(price_obj.get("currency") or "usd").lower()
        sub_period_end = subscription_dict.get("current_period_end")
        if isinstance(sub_period_end, int):
            dt = _safe_dt_from_unix(sub_period_end)
            if dt is not None:
                out["renewal_date"] = dt.isoformat()
                if getattr(billing_profile, "stripe_current_period_end", None) is None:
                    try:
                        billing_profile.stripe_current_period_end = dt
                        billing_profile.save(update_fields=["stripe_current_period_end", "updated_at"])
                    except Exception:
                        logger.exception("[billing_summary] failed to persist fallback renewal profile_id=%s", billing_profile.id)

    latest_paid_at: datetime | None = None
    invoice_rows: list[dict] = []
    try:
        inv_list = stripe.Invoice.list(customer=customer_id, limit=12)
        ninv = normalize_stripe_payload(inv_list)
        rows = ninv.get("data", []) if isinstance(ninv, dict) else []
        if (not isinstance(rows, list) or len(rows) == 0) and subscription_id:
            inv_list_sub = stripe.Invoice.list(subscription=subscription_id, limit=12)
            ninv_sub = normalize_stripe_payload(inv_list_sub)
            rows_sub = ninv_sub.get("data", []) if isinstance(ninv_sub, dict) else []
            if isinstance(rows_sub, list) and rows_sub:
                rows = rows_sub
        invoice_rows = rows if isinstance(rows, list) else []
        invoice_list_ok = True
        invoices_out: list[dict] = []
        if isinstance(rows, list):
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                created_unix = raw.get("created")
                created_dt = _safe_dt_from_unix(created_unix if isinstance(created_unix, int) else None)
                status = str(raw.get("status") or "").strip().lower() or "unknown"
                paid_at_unix = raw.get("status_transitions", {}).get("paid_at")
                paid_at = _safe_dt_from_unix(paid_at_unix if isinstance(paid_at_unix, int) else None)
                if status == "paid" and paid_at is not None and (
                    latest_paid_at is None or paid_at > latest_paid_at
                ):
                    latest_paid_at = paid_at
                amount_minor = raw.get("amount_paid")
                if not isinstance(amount_minor, int):
                    amount_minor = raw.get("amount_due") if isinstance(raw.get("amount_due"), int) else 0
                invoices_out.append(
                    {
                        "id": str(raw.get("number") or raw.get("id") or "").strip(),
                        "date": created_dt.isoformat() if created_dt else None,
                        "amount": _money_minor_to_decimal_str(amount_minor),
                        "currency": str(raw.get("currency") or out["currency"] or "usd").lower(),
                        "status": status.title(),
                    }
                )
        out["invoices"] = invoices_out
    except Exception as exc:
        logger.exception("[billing_summary] failed to list Stripe invoices customer=%s", customer_id)
        _record_billing_stripe_error(
            billing_profile=billing_profile,
            operation="stripe.invoice.list.billing_summary",
            exc=exc,
        )

    out["payment_method"] = _billing_resolve_payment_method(
        customer_id=customer_id,
        subscription_dict=subscription_dict if isinstance(subscription_dict, dict) else None,
        invoice_rows=invoice_rows,
    )

    # Fallback renewal date: 30 days from latest successful payment.
    if out.get("renewal_date") is None and latest_paid_at is not None:
        fallback = latest_paid_at + timedelta(days=30)
        out["renewal_date"] = fallback.isoformat()
        try:
            billing_profile.stripe_current_period_end = fallback
            billing_profile.save(update_fields=["stripe_current_period_end", "updated_at"])
        except Exception:
            logger.exception("[billing_summary] failed to persist renewal fallback profile_id=%s", billing_profile.id)

    logger.info(
        "[billing_summary] stripe_result profile_id=%s billing_profile_id=%s customer_id=%s subscription_id=%s subscription_retrieve_ok=%s subscription_list_ok=%s invoice_list_ok=%s price_month=%s renewal_date=%s invoice_count=%s payment_method_present=%s",
        profile.id,
        billing_profile.id,
        customer_id,
        subscription_id,
        subscription_retrieve_ok,
        subscription_list_ok,
        invoice_list_ok,
        out.get("price_month"),
        bool(out.get("renewal_date")),
        len(out.get("invoices", []) or []),
        bool(out.get("payment_method")),
    )
    return Response(out)


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def onboarding_local_dev_billing_complete(request: HttpRequest) -> Response:
    """
    Set fake Stripe billing fields so onboarding can complete without Payment Links.

    Enabled when DJANGO_DEBUG is True (local) or when ALLOW_ONBOARDING_BILLING_BYPASS is True
    (explicit staging opt-in). Otherwise returns 404 (production default).
    """
    if not (settings.DEBUG or settings.ALLOW_ONBOARDING_BILLING_BYPASS):
        return Response(status=404)

    raw_profile_id = None
    if isinstance(request.data, dict):
        raw_profile_id = request.data.get("profile_id")
    target_profile = None
    if raw_profile_id is not None and str(raw_profile_id).strip() != "":
        try:
            pk = int(raw_profile_id)
        except (TypeError, ValueError):
            return Response({"error": "Invalid profile_id."}, status=400)
        target_profile = get_business_profile_for_user(request.user, pk)
        if target_profile is None:
            return Response({"error": "Business profile not found."}, status=404)
        profile = target_profile
    else:
        profile = resolve_workspace_business_profile_for_request(request)
        if profile is None:
            return Response({"error": "No active business profile."}, status=404)
    access = viewer_team_access(request.user, profile)
    if not access["viewer_can_access_billing"]:
        return Response({"error": "Billing is not available for this account."}, status=403)

    raw_plan = request.data.get("plan") if isinstance(request.data, dict) else None
    plan = str(raw_plan or "").strip() or BusinessProfile.PLAN_PRO
    if plan not in {
        BusinessProfile.PLAN_STARTER,
        BusinessProfile.PLAN_PRO,
        BusinessProfile.PLAN_ADVANCED,
    }:
        plan = BusinessProfile.PLAN_PRO

    profile.stripe_customer_id = "cus_local_dev"
    profile.stripe_subscription_id = "sub_local_dev"
    profile.stripe_price_id = "price_local_dev"
    profile.stripe_subscription_status = "active"
    profile.plan = plan
    profile.save(
        update_fields=[
            "stripe_customer_id",
            "stripe_subscription_id",
            "stripe_price_id",
            "stripe_subscription_status",
            "plan",
            "updated_at",
        ],
    )
    site = str(getattr(profile, "website_url", "") or "").strip()
    if site:
        data_user = workspace_data_user(profile) or request.user
        try:
            run_full_seo_snapshot_for_profile(
                profile,
                data_user_fallback=data_user,
                abort_on_low_coverage=True,
            )
        except Exception:
            logger.exception(
                "[onboarding_local_dev_billing_complete] full SEO snapshot failed profile_id=%s",
                profile.id,
            )
    return Response({"ok": True, "plan": plan})


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def seo_profile_data(request: HttpRequest) -> Response:
    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        if not should_create_owned_main_business_profile_for_user(request.user):
            return Response({"error": "No active business profile."}, status=404)
        profile = BusinessProfile.objects.create(user=request.user, is_main=True)
        BusinessProfileMembership.objects.get_or_create(
            business_profile=profile,
            user=request.user,
            defaults={
                "role": BusinessProfileMembership.ROLE_ADMIN,
                "is_owner": True,
            },
        )
    profile = (
        BusinessProfile.objects.filter(pk=profile.pk)
        .prefetch_related("tracked_competitors")
        .first()
        or profile
    )
    data_user = workspace_data_user(profile) or request.user
    access = viewer_team_access(request.user, profile)
    serializer = BusinessProfileSEOSerializer(
        profile,
        context={"request": request, "viewer_access": access},
    )
    payload = dict(serializer.data)

    # Expose SEO score trend from historical snapshots for dashboard charting.
    website = str(getattr(profile, "website_url", "") or "").strip()
    parsed_domain = normalize_domain(website) if website else ""

    snapshots_qs = SEOOverviewSnapshot.objects.filter(business_profile=profile)
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
    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response({"points": []})

    data_user = workspace_data_user(profile) or request.user
    website = str(getattr(profile, "website_url", "") or "").strip()
    parsed_domain = normalize_domain(website) if website else ""
    start_current = datetime.now(timezone.utc).date().replace(day=1)

    # Ensure current-period snapshot is present in history (without forcing a refresh).
    if website:
        try:
            get_or_refresh_seo_score_for_user(
                data_user,
                site_url=website,
                force_refresh=False,
                business_profile=profile,
            )
        except Exception:
            logger.exception(
                "[seo_score_history] failed to warm current snapshot user_id=%s website=%s",
                getattr(request.user, "id", None),
                website,
            )

    snapshots_qs = SEOOverviewSnapshot.objects.filter(business_profile=profile)
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
                data_user,
                site_url=website,
                force_refresh=False,
                business_profile=profile,
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
    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        if not should_create_owned_main_business_profile_for_user(request.user):
            return Response({"error": "No active business profile."}, status=404)
        profile = BusinessProfile.objects.create(user=request.user, is_main=True)
        BusinessProfileMembership.objects.get_or_create(
            business_profile=profile,
            user=request.user,
            defaults={
                "role": BusinessProfileMembership.ROLE_ADMIN,
                "is_owner": True,
            },
        )
    access = viewer_team_access(request.user, profile)
    serializer = BusinessProfileAEOSerializer(
        profile,
        context={
            "force_aeo_refresh": False,
            "disable_seo_context_for_aeo": True,
            "request": request,
            "viewer_access": access,
        },
    )
    return Response(serializer.data)


def _competitor_domains_from_references(refs: dict) -> list[str]:
    """Top competitor domains from ``references.competitors`` for prompt-coverage / Actions UI."""
    from urllib.parse import urlparse

    comps = refs.get("competitors")
    if not isinstance(comps, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for c in comps:
        dom = ""
        if isinstance(c, dict):
            dom = str(c.get("domain") or "").strip().lower()
            if not dom:
                url = str(c.get("url") or "").strip()
                if url:
                    try:
                        h = (urlparse(url).netloc or "").strip().lower()
                        if h.startswith("www."):
                            h = h[4:]
                        dom = h
                    except Exception:
                        pass
        elif isinstance(c, str):
            s = c.strip().lower()
            if s and "." in s:
                dom = s.split("/")[0].split("?")[0]
                if dom.startswith("www."):
                    dom = dom[4:]
        if not dom:
            continue
        if dom in seen:
            continue
        seen.add(dom)
        out.append(dom[:200])
        if len(out) >= 10:
            break
    return out


def _improvement_recommendations_for_prompt(
    prompt_key: str,
    response_ids: set[int],
    recs: list,
) -> list:
    """
    Map Phase-5 ``AEORecommendationRun.recommendations_json`` items to a single prompt row.

    Matches, in order: ``applies_to.prompt_examples`` / ``applies_to.response_snapshot_ids`` (v2 multi-angle),
    then ``references.matched_prompt_texts`` / ``matched_response_snapshot_ids``, then legacy ``prompt`` /
    ``references.response_snapshot_id`` for this prompt's response snapshots.

    When recommendations include ``rec_id`` or ``id``, duplicates are deduped by that id; legacy rows without
    ids still dedupe by normalized body text.
    """
    out: list = []
    key = (prompt_key or "").strip()

    def _norm_text(s: str) -> str:
        return " ".join((s or "").split()).strip().lower()

    key_norm = _norm_text(key)
    seen_text: set[str] = set()
    seen_rec_ids: set[str] = set()
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        at = rec.get("action_type")
        matched = False
        refs = rec.get("references") if isinstance(rec.get("references"), dict) else {}
        matched_prompt_texts = refs.get("matched_prompt_texts")
        matched_response_snapshot_ids = refs.get("matched_response_snapshot_ids")
        applies_to = rec.get("applies_to") if isinstance(rec.get("applies_to"), dict) else {}
        applies_examples = applies_to.get("prompt_examples")
        applies_resp = applies_to.get("response_snapshot_ids")

        # Precedence 0: v2 applies_to block (survives multi-angle grouping).
        if not matched and isinstance(applies_examples, list) and applies_examples:
            norm_applies = {_norm_text(str(x)) for x in applies_examples if str(x).strip()}
            if key_norm and key_norm in norm_applies:
                matched = True
        if not matched and isinstance(applies_resp, list) and applies_resp:
            ids_a: set[int] = set()
            for x in applies_resp:
                try:
                    ids_a.add(int(x))
                except (TypeError, ValueError):
                    continue
            if ids_a.intersection(response_ids):
                matched = True

        # Precedence 1: explicit matched prompt membership.
        if not matched and isinstance(matched_prompt_texts, list) and matched_prompt_texts:
            norm_members = {_norm_text(str(x)) for x in matched_prompt_texts if str(x).strip()}
            if key_norm and key_norm in norm_members:
                matched = True
        # Precedence 2: explicit grouped response id membership.
        elif not matched and isinstance(matched_response_snapshot_ids, list) and matched_response_snapshot_ids:
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
        elif not matched and at == "create_content":
            p = (rec.get("prompt") or "").strip()
            if _norm_text(p) == key_norm:
                matched = True
        elif not matched and at == "acquire_citation":
            rid = refs.get("response_snapshot_id")
            try:
                rid_int = int(rid) if rid is not None else None
            except (TypeError, ValueError):
                rid_int = None
            if rid_int is not None and rid_int in response_ids:
                matched = True
        if not matched:
            continue
        summary = (rec.get("summary") or "").strip()
        nl = (rec.get("nl_explanation") or "").strip()
        reason = (rec.get("reason") or "").strip()
        text_out = summary if summary else (nl if nl else reason)
        if not text_out:
            continue
        rec_id_raw = rec.get("rec_id")
        id_raw = rec.get("id")
        rec_id = str(rec_id_raw).strip() if rec_id_raw is not None and str(rec_id_raw).strip() else ""
        if not rec_id and id_raw is not None and str(id_raw).strip():
            rec_id = str(id_raw).strip()
        if rec_id:
            if rec_id in seen_rec_ids:
                continue
            seen_rec_ids.add(rec_id)
        else:
            body_key = _norm_text(text_out)
            if body_key in seen_text:
                continue
            seen_text.add(body_key)
        item = {
            "action_type": at,
            "priority": (rec.get("priority") or "medium"),
            "text": text_out,
        }
        if rec_id:
            item["rec_id"] = rec_id
        if reason and reason != text_out:
            item["reason"] = reason
        if summary and summary != text_out:
            item["summary"] = summary
        ang = rec.get("angle")
        if isinstance(ang, str) and ang.strip():
            item["angle"] = ang.strip()
        acts = rec.get("actions")
        if isinstance(acts, list) and acts:
            item["actions"] = acts
        if isinstance(applies_to, dict) and applies_to:
            item["applies_to"] = {
                k: v
                for k, v in applies_to.items()
                if k in ("prompt_count", "prompt_examples", "response_snapshot_ids", "cluster_summary")
            }
        src = rec.get("source")
        if isinstance(src, str) and src.strip():
            item["source"] = src.strip()
        snap_ids = refs.get("matched_response_snapshot_ids")
        if isinstance(snap_ids, list) and snap_ids:
            norm_ids: list[int] = []
            for x in snap_ids:
                try:
                    norm_ids.append(int(x))
                except (TypeError, ValueError):
                    continue
            if norm_ids:
                item["matched_response_snapshot_ids"] = norm_ids
        comp_domains = _competitor_domains_from_references(refs)
        if comp_domains:
            item["competitor_domains"] = comp_domains[:10]
        out.append(item)
    return out


def _build_aeo_prompt_coverage_payload(profile: BusinessProfile, ready_only: bool = False) -> dict:
    """
    One row per monitored / seen prompt; per-platform cells use **profile-wide** latest snapshots
    (what models show now in the grid).

    Extra keys for clients:
    - ``prompt_scan_total``: len(monitored prompts), same basis as ``monitored_count``.
    - ``prompt_scan_completed``: for the **progress banner**, counts monitored prompts whose
      latest per-platform row is limited to ``AEOResponseSnapshot`` rows for the **newest**
      ``AEOExecutionRun`` only (triple extraction complete on that run).
    - ``visibility_pending`` / ``visibility_pending_reasons``: banner semantics scoped to the
      latest execution run (see ``aeo_banner_visibility_pending_breakdown``); repair / other
      helpers may still use the global breakdown elsewhere.
    - Per-row ``fully_ready`` still uses **global** snapshots + latest aggregate (grid semantics).
    - ``full_phase_*`` (completed count, target, ETA) use the **latest run** only for completion
      detection and ETA start times (``merge_eta_state_after_completions`` with run id).
    Optional ``ready_only`` filters to monitored rows that are ``fully_ready`` (global).
    """
    from .aeo.aeo_extraction_utils import (
        citations_ranking_for_prompt_coverage,
        merge_citations_rankings_across_platform_cells,
        merged_target_url_position,
        root_domain_from_fragment,
        tracked_domain_listed_in_competitors,
        unique_business_count_excluding_target,
    )
    from .models import AEOExecutionRun, AEOResponseSnapshot, AEORecommendationRun

    from .aeo.prompt_storage import (
        custom_prompt_flags_by_text,
        monitored_prompt_keys_in_order,
        prompt_text_from_storage_row,
    )

    monitored_keys = monitored_prompt_keys_in_order(profile.selected_aeo_prompts)
    selected_prompt_count = len(monitored_keys)
    custom_by_cf = custom_prompt_flags_by_text(profile.selected_aeo_prompts)
    profile_site = (getattr(profile, "website_url", None) or "").strip()
    profile_business_name = (getattr(profile, "business_name", None) or "").strip()

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

    latest_run = AEOExecutionRun.objects.filter(profile=profile).order_by("-created_at", "-id").first()
    by_prompt_latest_run: dict[str, list] = {}
    if latest_run is not None:
        for resp in responses:
            if getattr(resp, "execution_run_id", None) != latest_run.id:
                continue
            key_lr = (resp.prompt_text or "").strip()
            if not key_lr:
                continue
            by_prompt_latest_run.setdefault(key_lr, []).append(resp)

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
            comps = latest_ex.competitors_json or []
            competitors_count = len(comps) if isinstance(comps, list) else 0
            ranking, target_url_position = citations_ranking_for_prompt_coverage(
                latest_ex.citations_json,
                latest_ex.competitors_json,
                tracked_website_url_or_domain=profile_site,
                brand_mentioned=bool(latest_ex.brand_mentioned),
                tracked_business_name=profile_business_name,
            )
            tracked_root = root_domain_from_fragment(profile_site) or profile_site.strip().lower().rstrip(".")
            cited_by_citation_domain = target_url_position is not None
            cited_by_competitor_domain = (
                bool(tracked_root)
                and tracked_domain_listed_in_competitors(tracked_root, latest_ex.competitors_json)
            )
            # Prompt-table citation attribution should rely on profile URL/domain evidence only.
            cited = bool(cited_by_citation_domain or cited_by_competitor_domain)
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
    recommendation_strategies: list = []
    if latest_reco is not None:
        raw_st = getattr(latest_reco, "strategies_json", None) or []
        recommendation_strategies = list(raw_st) if isinstance(raw_st, list) else []
    if not recommendation_strategies and reco_items:
        from .aeo.aeo_recommendation_utils import build_recommendation_strategies_from_flat

        recommendation_strategies = build_recommendation_strategies_from_flat(
            reco_items,
            business_profile=profile,
            monitored_prompt_count=selected_prompt_count,
        )
    completed_strategy_ids = _completed_strategy_ids_from_actions_log(
        getattr(latest_reco, "actions_completed_json", None) if latest_reco else None
    )

    ordered_keys: list[str] = []
    seen: set[str] = set()
    for raw in profile.selected_aeo_prompts or []:
        k = prompt_text_from_storage_row(raw)
        if not k or k in seen:
            continue
        seen.add(k)
        ordered_keys.append(k)
    for k in sorted(by_prompt.keys()):
        if k not in seen:
            ordered_keys.append(k)

    from .aeo import prompt_full_ready as aeo_full_ready

    monitored_set = set(monitored_keys)
    recs_settled = aeo_full_ready.recommendations_pipeline_settled_for_visibility(profile)

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

        monitored = key in monitored_set
        fully_ready = monitored and aeo_full_ready.monitored_prompt_fully_ready(
            key, profile, by_prompt, latest_snapshot_per_platform, recs_settled
        )

        prompts.append(
            {
                "prompt": key,
                "prompt_type": prompt_type,
                "is_custom": bool(custom_by_cf.get(key.casefold())),
                "competitors_cited": comp_other_businesses,
                "platforms": platforms,
                "target_url_position": combined_target,
                "citations_ranking": combined_ranking,
                "improvement_recommendations": improvement_recommendations,
                "monitored": monitored,
                "fully_ready": fully_ready,
            }
        )

    tracked_name = profile_business_name
    prompt_scan_total = len(monitored_keys)
    prompt_scan_completed = prompt_scan_completed_count(
        monitored_keys, by_prompt_latest_run, latest_snapshot_per_platform
    )
    from .aeo.visibility_pending import aeo_banner_visibility_pending_breakdown

    _vis_bd = aeo_banner_visibility_pending_breakdown(profile)
    visibility_pending = bool(_vis_bd["visibility_pending"])
    prompt_fill_target = aeo_effective_monitored_target_for_profile(profile)
    recommendations_pending = _aeo_recommendations_pipeline_pending(profile)

    _lr_id = latest_run.id if latest_run is not None else None
    per_key_ready = {
        k: aeo_full_ready.monitored_prompt_fully_ready_for_execution_run(
            k,
            profile,
            by_prompt_latest_run,
            latest_snapshot_per_platform,
            recs_settled,
            _lr_id,
        )
        for k in monitored_keys
    }
    full_phase_completed = sum(1 for k in monitored_keys if per_key_ready.get(k))

    n_mon = len(monitored_keys)
    effective_target = min(n_mon, prompt_fill_target) if n_mon > 0 else prompt_fill_target
    remaining = max(0, int(effective_target) - int(full_phase_completed))

    with transaction.atomic():
        locked = BusinessProfile.objects.select_for_update().get(pk=profile.pk)
        durations = aeo_full_ready.merge_eta_state_after_completions(
            locked,
            monitored_keys,
            per_key_ready,
            execution_run_id=_lr_id,
        )

    full_phase_eta_seconds, full_phase_eta_cold_start = aeo_full_ready.compute_full_phase_eta_seconds(
        durations, remaining, full_phase_completed
    )

    if ready_only:
        prompts = [p for p in prompts if p.get("monitored") and p.get("fully_ready")]

    return {
        "prompts": prompts,
        "monitored_count": selected_prompt_count,
        "tracked_business_name": tracked_name,
        "recommendation_strategies": recommendation_strategies,
        "completed_strategy_ids": completed_strategy_ids,
        "prompt_scan_total": prompt_scan_total,
        "prompt_scan_completed": prompt_scan_completed,
        "visibility_pending": visibility_pending,
        "visibility_pending_reasons": {
            "execution_inflight": bool(_vis_bd["execution_inflight"]),
            "latest_run_extractions_inflight": bool(_vis_bd["latest_run_extractions_inflight"]),
            "snapshots_awaiting_extraction": bool(_vis_bd["snapshots_awaiting_extraction"]),
        },
        "recommendations_pending": recommendations_pending,
        "prompt_fill_completed": selected_prompt_count,
        "prompt_fill_target": prompt_fill_target,
        "full_phase_completed": full_phase_completed,
        "full_phase_target": prompt_fill_target,
        "full_phase_eta_seconds": full_phase_eta_seconds,
        "full_phase_eta_cold_start": full_phase_eta_cold_start,
        "aeo_prompt_expansion_status": getattr(profile, "aeo_prompt_expansion_status", "") or "",
        "aeo_prompt_expansion_last_error": getattr(profile, "aeo_prompt_expansion_last_error", "") or "",
    }


# Dashboard: reuse cached prompt-coverage payload for platform visibility + fast prompt-coverage reads.
_AEO_DASH_BUNDLE_CACHE_STALE_AFTER = timedelta(seconds=90)


def _aeo_platform_rows_from_prompts(prompts: list) -> list:
    """Build ``platforms`` list for ``/api/aeo/platform-visibility/`` from prompt rows."""

    def aggregate(api_key: str) -> tuple[int, int]:
        with_data = 0
        cited = 0
        for row in prompts:
            if not isinstance(row, dict):
                continue
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

    return [
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


def _maybe_enqueue_aeo_dashboard_bundle_refresh(
    profile: BusinessProfile, cache_row: Optional[AEODashboardBundleCache]
) -> None:
    if cache_row is None:
        return
    updated = getattr(cache_row, "updated_at", None)
    if updated is None:
        return
    if django_timezone.now() - updated <= _AEO_DASH_BUNDLE_CACHE_STALE_AFTER:
        return
    try:
        from .tasks import refresh_aeo_dashboard_bundle_cache_task

        refresh_aeo_dashboard_bundle_cache_task.delay(profile.id)
    except Exception:
        logger.exception(
            "[aeo_dashboard_cache] enqueue refresh failed profile_id=%s",
            profile.id,
        )


def _sanitize_prompt_coverage_monitored_flags(profile: BusinessProfile, payload: dict) -> None:
    """
    Recompute ``monitored`` from ``selected_aeo_prompts`` so legacy cached bundles cannot mark
    snapshot-only rows as monitored (which would make DELETE / append UX hit 404).
    """
    from .aeo.prompt_storage import monitored_prompt_keys_in_order, normalize_aeo_prompt_for_match

    monitored_ns = {
        normalize_aeo_prompt_for_match(k)
        for k in monitored_prompt_keys_in_order(profile.selected_aeo_prompts)
    }
    pl = payload.get("prompts")
    if not isinstance(pl, list):
        return
    for row in pl:
        if not isinstance(row, dict):
            continue
        ns = normalize_aeo_prompt_for_match(str(row.get("prompt") or ""))
        row["monitored"] = bool(ns) and ns in monitored_ns


def _aeo_prompt_coverage_payload_for_api(
    profile: BusinessProfile, *, ready_only: bool, force_refresh: bool
) -> dict:
    cache_row = AEODashboardBundleCache.objects.filter(profile=profile).first()
    if (
        cache_row
        and not force_refresh
        and isinstance(cache_row.payload_json, dict)
        and cache_row.payload_json
    ):
        payload = dict(cache_row.payload_json)
        _maybe_enqueue_aeo_dashboard_bundle_refresh(profile, cache_row)
    else:
        payload = _build_aeo_prompt_coverage_payload(profile, ready_only=False)
        AEODashboardBundleCache.objects.update_or_create(
            profile=profile,
            defaults={"payload_json": payload},
        )
    _sanitize_prompt_coverage_monitored_flags(profile, payload)
    if ready_only:
        pl = payload.get("prompts") or []
        payload = {
            **payload,
            "prompts": [
                p
                for p in pl
                if isinstance(p, dict) and p.get("monitored") and p.get("fully_ready")
            ],
        }
    return payload


def _aeo_profile_visibility_pending(profile: BusinessProfile) -> bool:
    """
    True while prompt LLM responses or extraction snapshots are still in flight for monitored
    prompts — dashboard should show a loading state instead of partial visibility %.
    """
    from .aeo.visibility_pending import aeo_visibility_pending_breakdown

    return bool(aeo_visibility_pending_breakdown(profile)["visibility_pending"])


def _patch_prompt_coverage_response_live(profile: BusinessProfile, payload: dict) -> None:
    """
    Refresh visibility flags from DB (cheap) and optionally enqueue repair; mutates ``payload``.
    Cached bundle payloads get up-to-date visibility + repair metadata on read.
    """
    from django.core.cache import cache

    from .aeo.aeo_plan_targets import aeo_should_run_post_payment_expansion, aeo_testing_mode
    from .aeo.visibility_pending import (
        aeo_banner_visibility_pending_breakdown,
        aeo_visibility_pending_breakdown,
    )
    from .tasks import aeo_repair_stalled_visibility_pipeline_task

    # Banner flags should reflect latest-run-only semantics (not profile-wide historic artifacts).
    b = aeo_banner_visibility_pending_breakdown(profile)
    payload["visibility_pending"] = bool(b["visibility_pending"])
    payload["visibility_pending_reasons"] = {
        "execution_inflight": bool(b["execution_inflight"]),
        "latest_run_extractions_inflight": bool(b["latest_run_extractions_inflight"]),
        "snapshots_awaiting_extraction": bool(b["snapshots_awaiting_extraction"]),
    }
    # Keep recommendation pending live even when base payload came from cache.
    payload["recommendations_pending"] = bool(_aeo_recommendations_pipeline_pending(profile))

    # Repair gating remains profile-wide by design (fills historical coverage gaps too).
    b_repair = aeo_visibility_pending_breakdown(profile)
    repair_meta = {"eligible": False, "enqueued": False, "skipped_reason": None}
    if aeo_testing_mode():
        repair_meta["skipped_reason"] = "testing_mode"
    elif not aeo_should_run_post_payment_expansion(profile):
        repair_meta["skipped_reason"] = "plan_not_eligible"
    else:
        repair_meta["eligible"] = True
        ckey = f"aeo_visibility_repair_tick:{profile.id}"
        if cache.get(ckey):
            repair_meta["skipped_reason"] = "throttled"
        else:
            cache.set(ckey, 1, timeout=300)
            try:
                if bool(b_repair["visibility_pending"]):
                    repair_meta["skipped_reason"] = "already_pending"
                else:
                    aeo_repair_stalled_visibility_pipeline_task.delay(profile.id)
                    repair_meta["enqueued"] = True
            except Exception:
                logger.exception(
                    "[aeo_prompt_coverage] repair enqueue failed profile_id=%s",
                    profile.id,
                )
                repair_meta["skipped_reason"] = "enqueue_failed"
                cache.delete(ckey)
    payload["visibility_repair"] = repair_meta


def _aeo_recommendations_pipeline_pending(profile: BusinessProfile) -> bool:
    """
    True while the latest execution run has finished scoring but Phase 5 has not persisted yet.

    Lets the dashboard keep polling prompt-coverage so Actions picks up ``recommendation_strategies``
    after expansion/backfill (visibility can be false while recommendations are still running).
    """
    if not bool(getattr(settings, "AEO_ENABLE_RECOMMENDATION_STAGE", False)):
        return False
    from .models import AEOExecutionRun

    latest_ex = (
        AEOExecutionRun.objects.filter(profile=profile).order_by("-created_at", "-id").first()
    )
    if latest_ex is None:
        return False
    return (
        latest_ex.scoring_status == AEOExecutionRun.STAGE_COMPLETED
        and latest_ex.recommendation_status
        in (AEOExecutionRun.STAGE_PENDING, AEOExecutionRun.STAGE_RUNNING)
    )


def _completed_strategy_ids_from_actions_log(raw: object) -> list[str]:
    """Normalize ``actions_completed_json`` into unique strategy ids."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        strategy_id = ""
        if isinstance(item, dict):
            strategy_id = str(item.get("strategy_id") or "").strip()
        elif isinstance(item, str):
            # Legacy shape: stored as a bare id.
            strategy_id = item.strip()
        if not strategy_id or strategy_id in seen:
            continue
        seen.add(strategy_id)
        out.append(strategy_id)
    return out


@csrf_exempt
@api_view(["GET"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_prompt_coverage_data(request: HttpRequest) -> Response:
    """
    Cached-only prompt coverage read for AI Visibility UI.
    One row per unique prompt text; per-platform (OpenAI / Gemini / Perplexity) citation from latest snapshot each.
    """
    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response(
            {
                "prompts": [],
                "monitored_count": 0,
                "tracked_business_name": "",
                "prompt_scan_total": 0,
                "prompt_scan_completed": 0,
                "visibility_pending": False,
                "visibility_pending_reasons": {
                    "execution_inflight": False,
                    "latest_run_extractions_inflight": False,
                    "snapshots_awaiting_extraction": False,
                },
                "visibility_repair": {
                    "eligible": False,
                    "enqueued": False,
                    "skipped_reason": "no_profile",
                },
                "recommendations_pending": False,
                "prompt_fill_completed": 0,
                "prompt_fill_target": aeo_fallback_global_target_count(),
                "full_phase_completed": 0,
                "full_phase_target": aeo_fallback_global_target_count(),
                "full_phase_eta_seconds": None,
                "full_phase_eta_cold_start": True,
                "aeo_prompt_expansion_status": "",
                "aeo_prompt_expansion_last_error": "",
            }
        )
    ready_only = str(request.GET.get("ready_only", "")).lower() in ("1", "true", "yes")
    force_refresh = str(request.GET.get("refresh", "")).lower() in ("1", "true", "yes")
    payload = _aeo_prompt_coverage_payload_for_api(
        profile, ready_only=ready_only, force_refresh=force_refresh
    )
    _patch_prompt_coverage_response_live(profile, payload)
    return Response(payload)


@csrf_exempt
@api_view(["POST", "DELETE"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_monitored_prompt_append(request: HttpRequest) -> Response:
    """
    Append a user-authored custom monitored prompt to ``BusinessProfile.selected_aeo_prompts``
    and queue Phase 1 (which chains extraction, multi-pass / stability, scoring, recommendations).
    """
    import re

    from .aeo.aeo_plan_targets import (
        aeo_effective_custom_prompt_cap_for_profile,
        aeo_effective_monitored_target_for_profile,
    )
    from .aeo.aeo_prompts import AEOPromptType
    from .aeo.aeo_utils import prompt_record
    from .aeo.prompt_storage import (
        count_custom_prompts_in_selected,
        monitored_prompt_keys_in_order,
        normalize_aeo_prompt_for_match,
        prompt_text_from_storage_row,
        row_is_custom,
    )
    from .models import AEODashboardBundleCache, AEOExecutionRun
    from .tasks import run_aeo_phase1_execution_task

    profile = resolve_workspace_business_profile_for_request(request)
    if profile is None:
        return Response({"error": "No active business profile."}, status=404)
    access = viewer_team_access(request.user, profile)
    if not access.get("viewer_can_edit_company_profile"):
        return Response({"detail": "You do not have permission to edit monitored prompts."}, status=403)

    body = request.data if isinstance(request.data, dict) else {}
    prompt_raw = str(body.get("prompt") or "").strip()

    if request.method == "DELETE":
        if not prompt_raw:
            return Response({"error": "prompt is required."}, status=400)

        removed = False
        removed_custom = False
        with transaction.atomic():
            locked = BusinessProfile.objects.select_for_update().get(pk=profile.pk)
            cur = list(locked.selected_aeo_prompts or [])
            nxt = []
            for raw in cur:
                t = prompt_text_from_storage_row(raw)
                if t and normalize_aeo_prompt_for_match(t) == normalize_aeo_prompt_for_match(prompt_raw):
                    removed = True
                    removed_custom = bool(row_is_custom(raw))
                    continue
                nxt.append(raw)
            if not removed:
                return Response({"error": "Prompt not found on monitored list."}, status=404)
            if removed_custom:
                return Response({"error": "Custom prompts cannot be deleted."}, status=400)
            locked.selected_aeo_prompts = nxt
            locked.aeo_custom_prompt_cap_bonus = int(
                max(0, int(locked.aeo_custom_prompt_cap_bonus or 0)) + 1
            )
            locked.save(
                update_fields=[
                    "selected_aeo_prompts",
                    "aeo_custom_prompt_cap_bonus",
                    "updated_at",
                ]
            )
            profile = locked

        profile.refresh_from_db(
            fields=["selected_aeo_prompts", "aeo_custom_prompt_cap_bonus", "updated_at"]
        )
        AEODashboardBundleCache.objects.filter(profile=profile).delete()
        return Response(
            {
                "ok": True,
                "deleted_prompt": prompt_raw,
                "monitored_count": len(monitored_prompt_keys_in_order(profile.selected_aeo_prompts)),
                "custom_prompt_cap": aeo_effective_custom_prompt_cap_for_profile(profile),
                "custom_prompt_bonus": int(profile.aeo_custom_prompt_cap_bonus or 0),
            },
            status=200,
        )

    if not prompt_raw or len(prompt_raw) > 2000:
        return Response({"error": "A non-empty prompt under 2000 characters is required."}, status=400)

    cap = aeo_effective_monitored_target_for_profile(profile)
    custom_cap = aeo_effective_custom_prompt_cap_for_profile(profile)
    current = list(profile.selected_aeo_prompts or [])
    keys = monitored_prompt_keys_in_order(current)
    custom_n = count_custom_prompts_in_selected(current)
    new_match_key = normalize_aeo_prompt_for_match(prompt_raw)
    if new_match_key in {normalize_aeo_prompt_for_match(k) for k in keys}:
        return Response({"error": "That prompt is already on your monitored list."}, status=400)
    if custom_n >= custom_cap:
        return Response(
            {"error": f"You can add at most {custom_cap} custom prompts on your current plan."},
            status=400,
        )
    if len(keys) >= cap:
        return Response(
            {
                "error": (
                    f"You are tracking the maximum of {cap} prompts for your plan. "
                    "Delete a suggested prompt to free a slot before adding a custom prompt."
                ),
            },
            status=400,
        )

    normalized_text = re.sub(r"\s+", " ", prompt_raw).strip()
    new_row = prompt_record(
        normalized_text,
        prompt_type=AEOPromptType.TRANSACTIONAL,
        weight=1.0,
        dynamic=True,
        is_custom=True,
    )

    with transaction.atomic():
        locked = BusinessProfile.objects.select_for_update().get(pk=profile.pk)
        cur = list(locked.selected_aeo_prompts or [])
        kset = {
            normalize_aeo_prompt_for_match(prompt_text_from_storage_row(x))
            for x in cur
            if prompt_text_from_storage_row(x)
        }
        if normalize_aeo_prompt_for_match(normalized_text) in kset:
            return Response({"error": "That prompt is already on your monitored list."}, status=400)
        cur_keys = monitored_prompt_keys_in_order(cur)
        cur_custom_n = count_custom_prompts_in_selected(cur)
        if cur_custom_n >= custom_cap:
            return Response(
                {"error": f"You can add at most {custom_cap} custom prompts on your current plan."},
                status=400,
            )
        if len(cur_keys) >= cap:
            return Response(
                {
                    "error": (
                        f"You are tracking the maximum of {cap} prompts for your plan. "
                        "Delete a suggested prompt to free a slot before adding a custom prompt."
                    ),
                },
                status=400,
            )
        cur.append(dict(new_row))
        locked.selected_aeo_prompts = cur
        locked.save(update_fields=["selected_aeo_prompts", "updated_at"])

    profile.refresh_from_db(fields=["selected_aeo_prompts", "updated_at"])
    AEODashboardBundleCache.objects.filter(profile=profile).delete()

    inflight = AEOExecutionRun.objects.filter(
        profile=profile,
        status__in=[AEOExecutionRun.STATUS_PENDING, AEOExecutionRun.STATUS_RUNNING],
    ).exists()
    if inflight:
        return Response(
            {
                "ok": True,
                "queued": False,
                "reason": "A pipeline run is already in progress; your prompt was saved and will run when it finishes.",
                "monitored_count": len(monitored_prompt_keys_in_order(profile.selected_aeo_prompts)),
            },
            status=200,
        )

    run = AEOExecutionRun.objects.create(
        profile=profile,
        prompt_count_requested=1,
        status=AEOExecutionRun.STATUS_PENDING,
    )
    transaction.on_commit(
        lambda rid=run.id, payload=[dict(new_row)]: run_aeo_phase1_execution_task.delay(
            rid, prompt_set=payload, force_refresh=True
        )
    )
    return Response(
        {
            "ok": True,
            "queued": True,
            "run_id": run.id,
            "monitored_count": len(monitored_prompt_keys_in_order(profile.selected_aeo_prompts)),
        },
        status=202,
    )


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_mark_recommendation_complete(request: HttpRequest) -> Response:
    """
    Mark a recommendation strategy as completed for the latest recommendation run.
    """
    from .models import AEORecommendationRun

    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response({"error": "Business profile not found."}, status=404)

    strategy_id = str(request.data.get("strategy_id") or "").strip()
    if not strategy_id:
        return Response({"error": "strategy_id is required."}, status=400)

    latest = (
        AEORecommendationRun.objects.filter(profile=profile)
        .order_by("-created_at", "-id")
        .first()
    )
    if latest is None:
        return Response({"error": "No recommendation run found."}, status=404)

    with transaction.atomic():
        locked = AEORecommendationRun.objects.select_for_update().get(pk=latest.pk)
        existing = list(locked.actions_completed_json or [])
        completed_ids = _completed_strategy_ids_from_actions_log(existing)
        if strategy_id not in completed_ids:
            existing.append(
                {
                    "strategy_id": strategy_id,
                    "completed_at": django_timezone.now().isoformat(),
                }
            )
            locked.actions_completed_json = existing
            locked.save(update_fields=["actions_completed_json"])
            completed_ids.append(strategy_id)

    # Ensure prompt-coverage reads pick up latest completion state immediately.
    AEODashboardBundleCache.objects.filter(profile=profile).delete()

    return Response({"ok": True, "completed_strategy_ids": completed_ids})


def _profile_location_line(profile: BusinessProfile) -> str:
    parts = [str(profile.customer_reach_city or "").strip(), str(profile.customer_reach_state or "").strip()]
    parts = [p for p in parts if p]
    if parts:
        return ", ".join(parts)
    return str(profile.business_address or "").strip()


def _profile_service_area_line(profile: BusinessProfile) -> str:
    reach = str(profile.customer_reach or "").strip().lower()
    if reach == "local":
        loc = _profile_location_line(profile)
        return f"{loc} and surrounding area".strip() if loc else "Local service area"
    return "Online / nationwide"


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def actions_generate_page_preview(request: HttpRequest) -> Response:
    """
    Structured JSON landing page preview for an action card.

    Returns a stored snapshot when ``action_key`` + ``content_hash`` match a saved row
    unless ``regenerate`` is true (then OpenAI runs and the snapshot is updated).
    ``content_hash`` must be a SHA-256 hex string of the canonical generation inputs
    (keyword, assets, plan steps, business context lines, etc.) so previews refresh when
    card content changes.
    """
    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response({"error": "Business profile not found."}, status=404)

    action_key = str(request.data.get("action_key") or "").strip()
    if not action_key or len(action_key) > 500:
        return Response({"error": "action_key is required (max 500 characters)."}, status=400)

    keyword = str(request.data.get("keyword") or "").strip()
    if not keyword:
        return Response({"error": "keyword is required."}, status=400)

    regenerate = request.data.get("regenerate") in (True, "true", "1", 1)
    content_hash = str(request.data.get("content_hash") or "").strip()[:64]

    if not regenerate and content_hash:
        snap = (
            ActionsGeneratedPageSnapshot.objects.filter(profile=profile, action_key=action_key)
            .only("page_data", "content_hash")
            .first()
        )
        if (
            snap
            and (snap.content_hash or "") == content_hash
            and isinstance(snap.page_data, dict)
            and isinstance(snap.page_data.get("page"), dict)
        ):
            return Response(snap.page_data)

    business_name = str(request.data.get("business_name") or profile.business_name or "").strip()
    location = str(request.data.get("location") or _profile_location_line(profile) or "").strip()
    service_area = str(request.data.get("service_area") or _profile_service_area_line(profile) or "").strip()
    page_type = str(request.data.get("page_type") or "").strip() or None

    raw_issues = request.data.get("seo_issues")
    seo_issues: list[str] = []
    if isinstance(raw_issues, list):
        for x in raw_issues:
            s = str(x or "").strip()
            if s:
                seo_issues.append(s)
    elif isinstance(raw_issues, str) and raw_issues.strip():
        seo_issues.append(raw_issues.strip())

    extras = openai_utils.parse_actions_landing_page_request_extras(request.data)

    try:
        payload = openai_utils.generate_structured_landing_page_preview(
            keyword=keyword,
            business_name=business_name or "Your business",
            location=location or "—",
            service_area=service_area or "—",
            seo_issues=seo_issues,
            page_type=page_type,
            business_profile=profile,
            **extras,
        )
    except json.JSONDecodeError as exc:
        logger.warning("actions_generate_page_preview json decode: %s", exc)
        return Response({"error": "Model returned invalid JSON. Try again."}, status=502)
    except Exception as exc:
        logger.exception("actions_generate_page_preview failed")
        return Response({"error": str(exc)[:500]}, status=502)

    ActionsGeneratedPageSnapshot.objects.update_or_create(
        profile=profile,
        action_key=action_key,
        defaults={"page_data": payload, "content_hash": content_hash},
    )

    return Response(payload)


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
    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response({"platforms": []})

    force_refresh = str(request.GET.get("refresh", "")).lower() in ("1", "true", "yes")
    cache_row = AEODashboardBundleCache.objects.filter(profile=profile).first()
    if (
        cache_row
        and not force_refresh
        and isinstance(cache_row.payload_json, dict)
        and cache_row.payload_json.get("prompts") is not None
    ):
        prompts = cache_row.payload_json.get("prompts") or []
        platforms_out = _aeo_platform_rows_from_prompts(prompts)
        visibility_pending = _aeo_profile_visibility_pending(profile)
        _maybe_enqueue_aeo_dashboard_bundle_refresh(profile, cache_row)
        return Response({"platforms": platforms_out, "visibility_pending": visibility_pending})

    payload = _build_aeo_prompt_coverage_payload(profile, ready_only=False)
    AEODashboardBundleCache.objects.update_or_create(
        profile=profile,
        defaults={"payload_json": payload},
    )
    prompts = payload.get("prompts") or []
    platforms_out = _aeo_platform_rows_from_prompts(prompts)
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

    profile = resolve_workspace_business_profile_for_request(request)
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

    pid_raw = (request.GET.get("profile_id") or "").strip()
    if pid_raw:
        try:
            pk = int(pid_raw)
        except ValueError:
            return Response({"has_data": False, "total_prompts": 0, "rows": []}, status=400)
        profile = get_business_profile_for_user(request.user, pk)
        if profile is None:
            return Response({"has_data": False, "total_prompts": 0, "rows": []}, status=404)
    else:
        profile = resolve_workspace_business_profile_for_request(request)
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

    profile = resolve_workspace_business_profile_for_request(request)
    if profile is not None:
        profile = (
            BusinessProfile.objects.filter(pk=profile.pk)
            .prefetch_related("tracked_competitors")
            .first()
            or profile
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
    if len(tracked) <= 3:
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

    profile = resolve_workspace_business_profile_for_request(request)
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
                "strategy_count": len(getattr(latest_reco, "strategies_json", None) or []),
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
    profile = resolve_workspace_business_profile_for_request(request)
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
    from .aeo.prompt_storage import monitored_prompt_keys_in_order
    from .models import AEOExecutionRun
    from .tasks import run_aeo_phase1_execution_task

    profile = resolve_workspace_business_profile_for_request(request)
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
    if not isinstance(saved, list) or not monitored_prompt_keys_in_order(saved):
        return Response(
            {"detail": "No saved AEO prompts to refresh. Complete onboarding first."},
            status=400,
        )

    raw_items = plan_items_from_saved_prompt_strings(saved)
    if not raw_items:
        return Response({"detail": "Could not build prompt list from saved prompts."}, status=400)

    target = aeo_effective_monitored_target_for_profile(profile)
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
    from .aeo.prompt_storage import monitored_prompt_keys_in_order
    from .models import AEOExecutionRun
    from .tasks import run_aeo_gemini_refresh_task

    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response({"detail": "No main business profile is configured."}, status=400)

    if not gemini_execution_enabled():
        return Response({"detail": "Gemini is not configured (set GEMINI_API_KEY)."}, status=400)

    saved = profile.selected_aeo_prompts or []
    if not isinstance(saved, list):
        saved = []
    if not monitored_prompt_keys_in_order(saved):
        return Response(
            {"detail": "No saved AEO prompts to refresh. Complete onboarding first."},
            status=400,
        )

    target = aeo_effective_monitored_target_for_profile(profile)
    selected = plan_items_from_saved_prompt_strings(saved, max_items=target)[:target]
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
    from .aeo.prompt_storage import monitored_prompt_keys_in_order
    from .models import AEOExecutionRun
    from .tasks import run_aeo_perplexity_refresh_task

    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response({"detail": "No main business profile is configured."}, status=400)

    if not perplexity_execution_enabled():
        return Response({"detail": "Perplexity is not configured (set PERPLEXITY_API_KEY)."}, status=400)

    saved = profile.selected_aeo_prompts or []
    if not isinstance(saved, list):
        saved = []
    if not monitored_prompt_keys_in_order(saved):
        return Response(
            {"detail": "No saved AEO prompts to refresh. Complete onboarding first."},
            status=400,
        )

    target = aeo_effective_monitored_target_for_profile(profile)
    selected = plan_items_from_saved_prompt_strings(saved, max_items=target)[:target]
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


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def aeo_retry_prompt_expansion(request: HttpRequest) -> Response:
    """
    Enqueue ``schedule_aeo_prompt_plan_expansion`` toward the plan cap (same kwargs pattern as Stripe).
    Pro/Advanced only; rate-limited per user to reduce abuse.
    """
    from .tasks import schedule_aeo_prompt_plan_expansion

    profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        return Response({"detail": "No main business profile is configured."}, status=400)

    if not aeo_should_run_post_payment_expansion(profile):
        return Response(
            {
                "detail": "Finish getting prompts is available on Pro and Advanced plans.",
            },
            status=400,
        )

    rate_key = f"aeo:retry_prompt_expansion:{request.user.id}"
    if not cache.add(rate_key, "1", 60):
        return Response({"detail": "Please wait a minute before requesting again."}, status=429)

    slug = str(getattr(profile, "plan", "") or "")
    cap = int(aeo_monitored_prompt_cap_for_plan_slug(slug))

    transaction.on_commit(
        lambda: schedule_aeo_prompt_plan_expansion.delay(
            profile.id,
            expected_plan_slug=slug,
            expansion_cap=cap,
        )
    )
    logger.info(
        "[AEO retry_prompt_expansion] profile_id=%s plan=%s cap=%s user_id=%s",
        profile.id,
        slug,
        cap,
        getattr(request.user, "id", None),
    )
    return Response({"enqueued": True}, status=200)


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
    q = request.GET.get("skip_execution")
    if q is None:
        return False
    return _truthy_openai_param(q)


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


def _saved_prompt_texts_for_profile_domain(
    profile: BusinessProfile,
    *,
    website_url_from_context: str,
    target_prompt_count: int,
) -> list[str]:
    """
    Resolve a reusable prompt set for onboarding step-2:
    1) current profile saved prompts (same domain),
    2) fallback to another profile under same user with same domain.
    """
    if target_prompt_count < 1:
        return []
    website_url_ctx = str(website_url_from_context or "").strip()
    if not website_url_ctx:
        return []

    def _extract(raw: list) -> list[str]:
        items = plan_items_from_saved_prompt_strings(raw, max_items=target_prompt_count)
        if len(items) != target_prompt_count:
            return []
        return [str(x.get("prompt") or "").strip() for x in items if str(x.get("prompt") or "").strip()]

    if (
        len(list(profile.selected_aeo_prompts or [])) == target_prompt_count
        and _saved_prompts_domain_matches_onboarding_context(profile, website_url_ctx)
    ):
        own = _extract(list(profile.selected_aeo_prompts or []))
        if len(own) == target_prompt_count:
            return own

    ctx_domain = normalize_domain(website_url_ctx) or ""
    if not ctx_domain:
        return []
    sibling_qs = (
        BusinessProfile.objects.filter(user=profile.user)
        .exclude(pk=profile.pk)
        .exclude(website_url="")
        .order_by("-is_main", "-updated_at", "-id")
    )
    for sib in sibling_qs:
        if normalize_domain(sib.website_url or "") != ctx_domain:
            continue
        raw_saved = list(sib.selected_aeo_prompts or [])
        if len(raw_saved) != target_prompt_count:
            continue
        rows = _extract(raw_saved)
        if len(rows) == target_prompt_count:
            return rows
    return []


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
      the plan-derived monitored prompt target, return that list without calling OpenAI or rebuilding templates.
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
    body = request.data if request.method == "POST" else {}
    if not isinstance(body, dict):
        body = {}
    profile_id_raw = body.get("profile_id", request.GET.get("profile_id"))
    profile: BusinessProfile | None = None
    if profile_id_raw not in (None, ""):
        try:
            profile_id = int(profile_id_raw)
        except (TypeError, ValueError):
            return Response({"error": "profile_id must be an integer."}, status=400)
        profile = get_business_profile_for_user(request.user, profile_id)
        if profile is None:
            return Response({"error": "Business profile not found for this account."}, status=404)
    else:
        profile = resolve_workspace_business_profile_for_request(request)
    if not profile:
        if not should_create_owned_main_business_profile_for_user(request.user):
            return Response({"error": "No active business profile."}, status=404)
        profile = BusinessProfile.objects.create(user=request.user, is_main=True)
        BusinessProfileMembership.objects.get_or_create(
            business_profile=profile,
            user=request.user,
            defaults={
                "role": BusinessProfileMembership.ROLE_ADMIN,
                "is_owner": True,
            },
        )
        ensure_organization_for_first_owned_profile(profile)
        profile.refresh_from_db()

    city_override = _first_nonempty_str(body.get("city"), request.GET.get("city"))
    industry_override = _first_nonempty_str(body.get("industry"), request.GET.get("industry"))
    include_openai = _onboarding_plan_include_openai(request, body)
    reuse_saved = _onboarding_reuse_saved_prompts(request, body)
    skip_execution = _onboarding_skip_execution(request, body)

    onboarding_step2 = bool(body.get("onboarding_step2_prompt_plan"))
    selected_topics_raw = body.get("selected_topics")

    # Full monitored cap for profile (plan target after expansion).
    profile_prompt_cap = aeo_effective_monitored_target_for_profile(profile)
    target_prompt_count = profile_prompt_cap
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
        # Step-2 prompt generation should stay fast/interactive (owner selects prompts),
        # so avoid generating the full post-expansion plan cap (75/150) inline.
        onboarding_min = int(aeo_onboarding_complete_min_prompts(profile))
        target_prompt_count = min(profile_prompt_cap, max(onboarding_min, len(selected_topics)))
        oc_early = body.get("onboarding_context") if isinstance(body.get("onboarding_context"), dict) else {}
        website_url_ctx = str(oc_early.get("website_url") or "").strip()
        prompt_texts = _saved_prompt_texts_for_profile_domain(
            profile,
            website_url_from_context=website_url_ctx,
            target_prompt_count=target_prompt_count,
        )
        if len(prompt_texts) == target_prompt_count:
            # Warm this profile cache so follow-up retries stay local to this profile.
            if list(profile.selected_aeo_prompts or []) != prompt_texts:
                try:
                    profile.selected_aeo_prompts = prompt_texts
                    profile.save(update_fields=["selected_aeo_prompts", "updated_at"])
                except Exception:
                    logger.exception("[AEO onboarding] failed to cache reused_saved prompts on profile")
            ser = _serialize_aeo_prompt_items(
                [{"prompt": p, "type": "", "weight": 1.0, "dynamic": True} for p in prompt_texts]
            )
            pbt = _assign_saved_prompts_to_selected_topics(prompt_texts, selected_topics)
            if all(len(pbt.get(t, [])) > 0 for t in selected_topics):
                business_input = aeo_business_input_from_onboarding_payload(
                    business_name=str(oc_early.get("business_name") or ""),
                    website_url=website_url_ctx,
                    location=str(oc_early.get("location") or ""),
                    language=str(oc_early.get("language") or ""),
                    selected_topics=selected_topics,
                    customer_reach=str(oc_early.get("customer_reach") or ""),
                    customer_reach_state=str(oc_early.get("customer_reach_state") or ""),
                    customer_reach_city=str(oc_early.get("customer_reach_city") or ""),
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

    if reuse_saved and isinstance(saved_raw, list):
        raw_items = plan_items_from_saved_prompt_strings(saved_raw, max_items=target_prompt_count)
        if raw_items:
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
                        "combined_shortfall": max(0, target_prompt_count - len(ser)),
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
            customer_reach=str(oc.get("customer_reach") or ""),
            customer_reach_state=str(oc.get("customer_reach_state") or ""),
            customer_reach_city=str(oc.get("customer_reach_city") or ""),
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
            accessible_business_profiles_queryset(request.user)
            .prefetch_related("tracked_competitors")
            .order_by("-is_main", "created_at", "id")
        )
        # Never run live DataForSEO / AEO readiness per row here — it blocks workers and can
        # exceed Gunicorn timeout (see BusinessProfileSerializer._get_aeo_bundle). Same pattern as
        # GET /api/business-profile/?skip_heavy=1 used by the app shell.
        serializer = BusinessProfileSerializer(
            profiles,
            many=True,
            context={
                "request": request,
                "skip_heavy_profile_metrics": True,
                "viewer_access_resolver": lambda p: viewer_team_access(request.user, p),
            },
        )
        return Response(serializer.data)

    # POST: create a new profile under this user
    existing_qs = BusinessProfile.objects.filter(user=request.user)
    is_first = not existing_qs.exists()

    data = request.data.copy()
    # Never allow client to set a different user
    data.pop("user", None)
    # New company profiles start without a plan; billing/webhooks determine paid tier later.
    data.pop("plan", None)

    if is_first:
        # First profile for this user is main by definition
        data["is_main"] = True
    else:
        # For additional profiles, default is_main to False unless explicitly set
        if "is_main" not in data:
            data["is_main"] = False

    # Idempotency guard: if this user already has a profile for the same normalized
    # website domain, return it instead of creating a duplicate profile.
    website_url_raw = str(data.get("website_url") or "").strip()
    requested_domain = normalize_domain(website_url_raw)
    if requested_domain:
        for row in existing_qs.exclude(website_url="").only("id", "website_url"):
            if normalize_domain(row.website_url or "") == requested_domain:
                access = viewer_team_access(request.user, row)
                serializer = BusinessProfileSerializer(
                    row,
                    context={
                        "request": request,
                        "viewer_access": access,
                        "skip_heavy_profile_metrics": True,
                    },
                )
                return Response(serializer.data, status=200)

    if not is_first and not user_may_create_additional_business_profile(request.user):
        return Response(
            {
                "detail": "Additional company profiles are available on the Advanced plan.",
                "error": "advanced_plan_required",
            },
            status=403,
        )

    serializer = BusinessProfileSerializer(data=data)
    serializer.is_valid(raise_exception=True)
    profile = serializer.save(user=request.user)

    if not getattr(profile, "organization_id", None):
        if is_first:
            ensure_organization_for_first_owned_profile(profile)
        else:
            attach_organization_for_additional_profile(profile)
        profile.refresh_from_db()

    # If this profile is being set as main, unset others in the same organization (or same owner legacy).
    if profile.is_main:
        if profile.organization_id:
            BusinessProfile.objects.filter(organization_id=profile.organization_id).exclude(pk=profile.pk).update(
                is_main=False
            )
        else:
            existing_qs.exclude(pk=profile.pk).update(is_main=False)

    # Bootstrap SEO snapshot when the profile already has a site URL (add-company flow sets URL on POST).
    # Without this, PATCH finalize only updates prompts and never hits the "website changed" refresh path.
    # Run async to avoid holding this HTTP request open on DataForSEO network latency.
    site_url = str(getattr(profile, "website_url", "") or "").strip()
    if site_url and normalize_domain(site_url):
        try:
            from .tasks import sync_enrich_seo_snapshot_for_profile_task

            transaction.on_commit(
                lambda pid=int(profile.id): sync_enrich_seo_snapshot_for_profile_task.delay(pid)
            )
        except Exception:
            logger.exception(
                "[business_profile_list] SEO snapshot bootstrap enqueue failed profile_id=%s",
                getattr(profile, "id", None),
            )

    access = viewer_team_access(request.user, profile)
    out_serializer = BusinessProfileSerializer(
        profile,
        context={
            "request": request,
            "viewer_access": access,
            "skip_heavy_profile_metrics": True,
        },
    )
    return Response(out_serializer.data, status=201)


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

    profile = resolve_workspace_business_profile_for_request(request)
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

    data_user = workspace_data_user(profile) or user
    try:
        snapshot_mode, snapshot_location_code = _seo_snapshot_context_for_profile(profile)
        snapshot = SEOOverviewSnapshot.objects.filter(
            business_profile=profile,
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

        get_or_refresh_seo_score_for_user(
            data_user,
            site_url=site_url,
            business_profile=profile,
        )
        snapshot = SEOOverviewSnapshot.objects.filter(
            business_profile=profile,
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
    from .tasks import seo_data_dict_from_seo_overview_snapshot

    seo_data = seo_data_dict_from_seo_overview_snapshot(snapshot)
    seo_data["business_name"] = getattr(profile, "business_name", "") or ""
    seo_data["website_url"] = site_url
    seo_data["business_description"] = getattr(profile, "description", "") or ""

    try:
        steps = generate_seo_next_steps(seo_data, snapshot=snapshot)
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
    access = viewer_team_access(request.user, profile)
    serializer = BusinessProfileSerializer(
        profile,
        context={"request": request, "viewer_access": access},
    )
    return Response(serializer.data)


@csrf_exempt
@api_view(["POST"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def refresh_seo_snapshot(request: HttpRequest) -> Response:
    """
    Force a refresh of the SEO snapshot (keywords, rankings, visibility) for the user's
    main business profile, then return the updated profile.

    Uses ``sync_enrich_current_period_seo_snapshot_for_profile`` (same path as post-payment
    Celery follow-up): bootstraps the current-period row if needed, synchronously enriches
    keywords and recomputes metrics, then enqueues next-steps and keyword-action tasks.
    """
    user = request.user

    profile = resolve_workspace_business_profile_for_request(request)
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
        if not (normalize_domain(site_url) or ""):
            raise ValueError("Could not normalize domain for refresh_seo_snapshot")

        force_domain_intersection = False
        try:
            body = getattr(request, "data", None)
            if isinstance(body, dict):
                force_domain_intersection = bool(body.get("force_domain_intersection"))
        except Exception:
            force_domain_intersection = False

        sync_result = sync_enrich_current_period_seo_snapshot_for_profile(
            profile,
            data_user_fallback=user,
            force_domain_intersection=force_domain_intersection,
            abort_on_low_coverage=True,
        )
        external_api_called = bool(sync_result.get("external_api_called"))
        if sync_result.get("aborted_low_coverage"):
            return Response(
                {
                    "detail": sync_result.get("detail"),
                    "rank_coverage_percent": sync_result.get("rank_coverage_percent"),
                },
                status=409,
            )
    except Exception:
        # Never block the UI on SEO refresh failures; return the profile anyway.
        pass

    logger.info(
        "[refresh_endpoint] module=seo user_id=%s profile_id=%s refresh_completed=true external_api_called=%s",
        getattr(user, "id", None),
        getattr(profile, "id", None),
        bool(external_api_called),
    )
    access = viewer_team_access(request.user, profile)
    serializer = BusinessProfileSEOSerializer(
        profile,
        context={"request": request, "viewer_access": access},
    )
    return Response(serializer.data)

@csrf_exempt
@api_view(["GET", "PATCH", "PUT", "DELETE"])
@authentication_classes([CsrfExemptSessionAuthentication])
@permission_classes([IsAuthenticated])
def business_profile_detail(request: HttpRequest, pk: int) -> Response:
    """
    Retrieve, update, or delete a single business profile the authenticated user may access
    (owned, org-wide team, or site-specific invite).

    Primarily used by the Settings page to update a specific profile or mark it as main.
    """
    profile = get_business_profile_for_user(request.user, pk)
    if profile is None:
        return Response({"detail": "Not found."}, status=404)
    profile = (
        BusinessProfile.objects.prefetch_related("tracked_competitors")
        .filter(pk=profile.pk)
        .first()
        or profile
    )

    if request.method == "GET":
        access = viewer_team_access(request.user, profile)
        force_aeo_refresh = str(request.GET.get("refresh_aeo", "")).strip().lower() in {"1", "true", "yes"}
        skip_heavy = str(request.GET.get("skip_heavy", "")).strip().lower() in {"1", "true", "yes"}
        serializer = BusinessProfileSerializer(
            profile,
            context={
                "request": request,
                "viewer_access": access,
                "force_aeo_refresh": force_aeo_refresh,
                "disable_seo_context_for_aeo": True,
                "skip_heavy_profile_metrics": skip_heavy,
            },
        )
        return Response(serializer.data)

    if request.method in ("PATCH", "PUT"):
        access = viewer_team_access(request.user, profile)
        if not access["viewer_can_edit_company_profile"]:
            return Response(
                {"detail": "You do not have permission to edit company settings."},
                status=403,
            )
        old_site_url = profile.website_url
        serializer = BusinessProfileSerializer(
            profile,
            data=request.data,
            partial=(request.method == "PATCH"),
            context={"request": request, "viewer_access": access},
        )
        serializer.is_valid(raise_exception=True)
        profile = serializer.save()

        # If is_main was set to True on this profile, unset main on all other sites in the org.
        if profile.is_main:
            if profile.organization_id:
                BusinessProfile.objects.filter(organization_id=profile.organization_id).exclude(
                    pk=profile.pk
                ).update(is_main=False)
            else:
                BusinessProfile.objects.filter(user=request.user).exclude(pk=profile.pk).update(is_main=False)
            set_session_active_business_profile_for_user(request, request.user, int(profile.pk))

        # If the website URL changed, refresh SEO snapshot for this site.
        new_site_url = profile.website_url
        if new_site_url and new_site_url != old_site_url:
            try:
                data_user = workspace_data_user(profile) or request.user
                run_full_seo_snapshot_for_profile(
                    profile,
                    data_user_fallback=data_user,
                    abort_on_low_coverage=True,
                )
            except Exception:
                pass

        # Default PATCH/PUT response to lightweight metrics so profile saves never block on
        # live DataForSEO calls (can exceed worker timeout during external crawl requests).
        skip_heavy_raw = str(request.GET.get("skip_heavy", "")).strip().lower()
        skip_heavy = True if skip_heavy_raw == "" else skip_heavy_raw in {"1", "true", "yes"}
        out = BusinessProfileSerializer(
            profile,
            context={
                "request": request,
                "viewer_access": access,
                "disable_seo_context_for_aeo": True,
                "skip_heavy_profile_metrics": skip_heavy,
            },
        )
        return Response(out.data)

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

