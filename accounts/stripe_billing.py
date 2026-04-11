from __future__ import annotations

"""
Stripe → BusinessProfile billing sync.

Payment link → plan mapping uses Django settings STRIPE_PAYMENT_LINK_* (see config.settings.base).
Each value must be the Payment Link ID (e.g. plink_…) or the full buy URL (last path segment is used).
They must match the live Stripe Payment Links used in checkout, or checkout.session.completed may not
resolve a tier when the session omits price IDs.

Cancellation policy: when Stripe reports a terminal subscription status (canceled, unpaid,
incomplete_expired), we set BusinessProfile.plan to PLAN_NONE (empty string). Onboarding and paid
access remain gated on stripe_subscription_status (see accounts.onboarding_completion).
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from urllib.parse import urlparse

import stripe
from django.conf import settings
from django.db import transaction

from .models import BusinessProfile

logger = logging.getLogger(__name__)

ACTIVE_STRIPE_STATUSES = frozenset({"active", "trialing", "past_due"})
# Cleared to PLAN_NONE alongside stripe_subscription_status from Stripe (see module docstring).
TERMINAL_PLAN_CLEAR_STATUSES = frozenset({"canceled", "unpaid", "incomplete_expired"})


@dataclass(frozen=True)
class StripePlanMapping:
    plan_slug: str
    billing_cycle: str


@dataclass(frozen=True)
class StripeSyncResult:
    handled: bool
    did_update: bool
    matched_profile_id: int | None
    matched_by: str | None
    updated_fields: list[str]
    reason_code: str | None = None


def _normalize_link_id(link_or_url: str) -> str:
    s = (link_or_url or "").strip()
    if not s:
        return ""
    if "://" not in s:
        return s
    try:
        p = urlparse(s)
        path = (p.path or "").strip("/")
        return path.split("/")[-1] if path else ""
    except Exception:
        return s


def plan_mapping_by_payment_link_id() -> dict[str, StripePlanMapping]:
    raw = {
        settings.STRIPE_PAYMENT_LINK_STARTER_MONTHLY: StripePlanMapping("starter", "monthly"),
        settings.STRIPE_PAYMENT_LINK_STARTER_YEARLY: StripePlanMapping("starter", "yearly"),
        settings.STRIPE_PAYMENT_LINK_PRO_MONTHLY: StripePlanMapping("pro", "monthly"),
        settings.STRIPE_PAYMENT_LINK_PRO_YEARLY: StripePlanMapping("pro", "yearly"),
        settings.STRIPE_PAYMENT_LINK_ADVANCED_MONTHLY: StripePlanMapping("advanced", "monthly"),
        settings.STRIPE_PAYMENT_LINK_ADVANCED_YEARLY: StripePlanMapping("advanced", "yearly"),
    }
    out: dict[str, StripePlanMapping] = {}
    for k, v in raw.items():
        link_id = _normalize_link_id(k)
        if link_id:
            out[link_id] = v
    return out


def _safe_dt_from_unix(ts: int | None) -> datetime | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except Exception:
        return None


def _as_dict(x) -> dict:
    n = normalize_stripe_payload(x)
    return n if isinstance(n, dict) else {}


def _get_scalar(obj, key: str, default=""):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _get_nested(obj, *path, default=None):
    cur = obj
    for segment in path:
        if isinstance(segment, int):
            if not isinstance(cur, list) or segment < 0 or segment >= len(cur):
                return default
            cur = cur[segment]
            continue
        if not isinstance(cur, dict):
            return default
        cur = cur.get(segment)
        if cur is None:
            return default
    return cur


def _object_payload(payload) -> dict:
    d = _as_dict(payload)
    nested_obj = d.get("object")
    if isinstance(nested_obj, dict):
        return nested_obj
    return d


def _first_price_id_from_lines(obj: dict) -> str:
    return str(_get_nested(obj, "lines", "data", 0, "price", "id", default="") or "").strip()


def _first_price_id_from_items(obj: dict) -> str:
    return str(_get_nested(obj, "items", "data", 0, "price", "id", default="") or "").strip()


def _subscription_id_and_dict(sub_raw) -> tuple[str, dict | None]:
    if isinstance(sub_raw, dict):
        return str(sub_raw.get("id") or "").strip(), sub_raw
    return str(sub_raw or "").strip(), None


def _price_id_from_session_scalar_price(obj: dict) -> str:
    pr = _get_scalar(obj, "price")
    if isinstance(pr, dict):
        return str(pr.get("id") or "").strip()
    s = str(pr or "").strip()
    return s


def _resolve_price_id_for_checkout_session(obj: dict) -> str:
    line_data = _get_nested(obj, "line_items", "data", default=None)
    if isinstance(line_data, list) and line_data:
        pid = str(_get_nested(line_data[0], "price", "id", default="") or "").strip()
        if pid:
            return pid
    pl = _first_price_id_from_lines(obj)
    if pl:
        return pl
    sub_raw = _get_scalar(obj, "subscription")
    if isinstance(sub_raw, dict):
        ip = _first_price_id_from_items(sub_raw)
        if ip:
            return ip
    return _price_id_from_session_scalar_price(obj)


def normalize_stripe_payload(value):
    """
    Recursively normalize Stripe payload objects to plain Python dict/list/scalars.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): normalize_stripe_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_stripe_payload(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_stripe_payload(v) for v in value]
    if hasattr(value, "to_dict_recursive"):
        try:
            return normalize_stripe_payload(value.to_dict_recursive())
        except Exception:
            return {}
    return value


def extract_match_debug_fields(payload) -> dict[str, str]:
    obj = _object_payload(payload)
    details = _as_dict(_get_scalar(obj, "customer_details"))
    return {
        "client_reference_id": str(_get_scalar(obj, "client_reference_id") or "").strip(),
        "customer": str(_get_scalar(obj, "customer") or "").strip(),
        "customer_details_email": str(_get_scalar(details, "email") or "").strip().lower(),
    }


def mask_email(email: str) -> str:
    e = (email or "").strip().lower()
    if "@" not in e:
        return ""
    local, domain = e.split("@", 1)
    if not local:
        return f"***@{domain}"
    return f"{local[:1]}***@{domain}"


def extract_sync_debug_fields(
    payload,
    *,
    event_type: str,
    matched_profile_id: int | None = None,
    did_update: bool | None = None,
) -> dict[str, str]:
    obj = _object_payload(payload)
    details = _as_dict(_get_scalar(obj, "customer_details"))
    email = str(
        _get_scalar(details, "email") or _get_scalar(obj, "customer_email") or _get_scalar(obj, "receipt_email") or ""
    ).strip().lower()
    sub_dbg, _ = _subscription_id_and_dict(_get_scalar(obj, "subscription"))
    sub_dbg = sub_dbg or str(_get_scalar(obj, "id") or "").strip()
    out = {
        "event_type": event_type,
        "client_reference_id": str(_get_scalar(obj, "client_reference_id") or "").strip(),
        "customer": str(_get_scalar(obj, "customer") or "").strip(),
        "subscription": sub_dbg,
        "email": email,
        "matched_profile_id": str(matched_profile_id or ""),
        "did_update": "true" if did_update else "false",
    }
    return out


def infer_sync_failure_reason(event_type: str, payload) -> str:
    obj = _object_payload(payload)
    details = _as_dict(_get_scalar(obj, "customer_details"))
    customer = str(_get_scalar(obj, "customer") or "").strip()
    sub_raw = _get_scalar(obj, "subscription")
    sub_id, sub_dict_inf = _subscription_id_and_dict(sub_raw)
    subscription = sub_id or str(_get_scalar(obj, "id") or "").strip()
    client_ref = str(_get_scalar(obj, "client_reference_id") or "").strip()
    email = str(
        _get_scalar(details, "email") or _get_scalar(obj, "customer_email") or _get_scalar(obj, "receipt_email") or ""
    ).strip().lower()
    payment_link = str(_get_scalar(obj, "payment_link") or "").strip()
    price_id = _first_price_id_from_lines(obj)
    sub_price_id = _first_price_id_from_items(obj)
    if not sub_price_id and sub_dict_inf:
        sub_price_id = _first_price_id_from_items(sub_dict_inf)
    status = str(_get_scalar(obj, "status") or "").strip()

    if not (client_ref or customer or subscription or email):
        return "no_profile_identifiers"
    if event_type == "checkout.session.completed" and not (customer or subscription or payment_link):
        return "no_stripe_ids"
    if event_type == "invoice.paid" and not (customer or subscription or price_id):
        return "no_stripe_ids_or_price"
    if event_type in {"customer.subscription.updated", "customer.subscription.deleted"} and not (
        customer or subscription or sub_price_id or status
    ):
        return "no_price_status_or_ids"
    return "no_profile_match_or_no_updates"


def _plan_from_price(price_id: str) -> str | None:
    pid = (price_id or "").strip()
    if not pid:
        return None
    mapping = {
        "starter": settings.STRIPE_PRICE_ID_STARTER_MONTHLY,
        "pro": settings.STRIPE_PRICE_ID_PRO_MONTHLY,
        "advanced": settings.STRIPE_PRICE_ID_ADVANCED_MONTHLY,
        "starter_yearly": settings.STRIPE_PRICE_ID_STARTER_YEARLY,
        "pro_yearly": settings.STRIPE_PRICE_ID_PRO_YEARLY,
        "advanced_yearly": settings.STRIPE_PRICE_ID_ADVANCED_YEARLY,
    }
    for k, v in mapping.items():
        if v and pid == v:
            return k.split("_")[0]
    return None


def _unique_profile_by_email(email: str) -> BusinessProfile | None:
    qs = BusinessProfile.objects.filter(user__email__iexact=email).order_by("-is_main", "-updated_at")
    rows = list(qs[:2])
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        logger.warning("[stripe] email fallback is ambiguous for email=%s", email)
    return None


def _resolve_profile_for_event(data: dict) -> tuple[BusinessProfile | None, str]:
    obj = _object_payload(data)
    client_ref = str(_get_scalar(obj, "client_reference_id") or "").strip()
    if client_ref.isdigit():
        p = BusinessProfile.objects.filter(id=int(client_ref)).first()
        if p is not None:
            return p, "client_reference_id"

    customer_id = str(_get_scalar(obj, "customer") or "").strip()
    if customer_id:
        profile = (
            BusinessProfile.objects.filter(stripe_customer_id=customer_id)
            .order_by("-is_main", "-updated_at")
            .first()
        )
        if profile is not None:
            return profile, "customer_id"
        # First-time webhook for a customer not yet linked in DB:
        # fetch customer email from Stripe and resolve profile by email.
        try:
            customer = stripe.Customer.retrieve(customer_id)
            customer_dict = _as_dict(customer)
            customer_email = str(customer_dict.get("email") or "").strip().lower()
            if customer_email:
                profile_by_email = _unique_profile_by_email(customer_email)
                if profile_by_email is not None:
                    return profile_by_email, "email"
        except Exception:
            logger.exception("[stripe] failed retrieving customer %s for profile resolution", customer_id)

    details = _as_dict(_get_scalar(obj, "customer_details"))
    email = str(_get_scalar(details, "email") or "").strip().lower()
    if not email:
        email = str(_get_scalar(obj, "customer_email") or _get_scalar(obj, "receipt_email") or "").strip().lower()
    if email:
        p = _unique_profile_by_email(email)
        if p is not None:
            return p, "email"
    return None, "none"


def apply_subscription_payload_to_profile(
    profile: BusinessProfile,
    *,
    customer_id: str = "",
    subscription_id: str = "",
    price_id: str = "",
    status: str = "",
    current_period_end_unix: int | None = None,
    cancel_at_period_end: bool | None = None,
    payment_link_id: str = "",
    subscription_dict: dict | None = None,
) -> tuple[bool, list[str]]:
    effective_price = (price_id or "").strip()
    if not effective_price and subscription_dict:
        effective_price = _first_price_id_from_items(subscription_dict)

    updates: dict[str, object] = {}
    if customer_id:
        updates["stripe_customer_id"] = customer_id
    if subscription_id:
        updates["stripe_subscription_id"] = subscription_id
    if effective_price:
        updates["stripe_price_id"] = effective_price
    if status:
        updates["stripe_subscription_status"] = status
    if current_period_end_unix is not None:
        updates["stripe_current_period_end"] = _safe_dt_from_unix(current_period_end_unix)
    if cancel_at_period_end is not None:
        updates["stripe_cancel_at_period_end"] = bool(cancel_at_period_end)

    st_norm = (status or "").strip().lower()
    resolved_slug = _plan_from_price(effective_price)
    if not resolved_slug and payment_link_id:
        m = plan_mapping_by_payment_link_id().get(_normalize_link_id(payment_link_id))
        if m:
            resolved_slug = m.plan_slug

    if st_norm in TERMINAL_PLAN_CLEAR_STATUSES:
        updates["plan"] = BusinessProfile.PLAN_NONE
    elif resolved_slug in {
        BusinessProfile.PLAN_STARTER,
        BusinessProfile.PLAN_PRO,
        BusinessProfile.PLAN_ADVANCED,
    }:
        updates["plan"] = resolved_slug

    if st_norm in ACTIVE_STRIPE_STATUSES and resolved_slug not in {
        BusinessProfile.PLAN_STARTER,
        BusinessProfile.PLAN_PRO,
        BusinessProfile.PLAN_ADVANCED,
    }:
        logger.warning(
            "[stripe] could not resolve plan slug from Stripe payload profile_id=%s status=%s "
            "effective_price_id=%s payment_link_id=%s (check STRIPE_PRICE_ID_* and STRIPE_PAYMENT_LINK_* settings)",
            profile.id,
            st_norm or "(empty)",
            effective_price or "(empty)",
            payment_link_id or "(empty)",
        )

    if not updates:
        return False, []
    updates["updated_at"] = datetime.now(tz=timezone.utc)
    for k, v in updates.items():
        setattr(profile, k, v)
    profile.save(update_fields=list(updates.keys()))
    plan_after = str(getattr(profile, "plan", "") or "")
    if plan_after in {BusinessProfile.PLAN_PRO, BusinessProfile.PLAN_ADVANCED}:

        def _enqueue_aeo_expansion() -> None:
            try:
                from .tasks import schedule_aeo_prompt_plan_expansion

                schedule_aeo_prompt_plan_expansion.delay(profile.id)
            except Exception:
                logger.exception(
                    "[stripe] enqueue AEO prompt expansion failed profile_id=%s",
                    profile.id,
                )

        transaction.on_commit(_enqueue_aeo_expansion)
    return True, list(updates.keys())


def sync_from_checkout_session(payload: dict, *, event_id: str = "") -> StripeSyncResult:
    payload = normalize_stripe_payload(payload)
    profile, resolver = _resolve_profile_for_event(payload)
    if profile is None:
        dbg = extract_match_debug_fields(payload)
        logger.error(
            "stripe.webhook.skipped event_id=%s event_type=%s reason_code=%s client_reference_id=%s customer=%s customer_details_email=%s",
            event_id,
            "checkout.session.completed",
            "missing_profile_match",
            dbg["client_reference_id"],
            dbg["customer"],
            dbg["customer_details_email"],
        )
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=None,
            matched_by="none",
            updated_fields=[],
            reason_code="missing_profile_match",
        )
    obj = _object_payload(payload)
    client_ref = str(_get_scalar(obj, "client_reference_id") or "").strip()
    details = _as_dict(_get_scalar(obj, "customer_details"))
    checkout_email = str(_get_scalar(details, "email") or _get_scalar(obj, "customer_email") or "").strip().lower()
    if client_ref.isdigit():
        ref_profile = BusinessProfile.objects.filter(id=int(client_ref)).first()
        if ref_profile is not None and checkout_email and ref_profile.user.email:
            if ref_profile.user.email.strip().lower() != checkout_email:
                logger.warning(
                    "[stripe] checkout session email mismatch for client_reference_id=%s profile_email=%s checkout_email=%s",
                    client_ref,
                    ref_profile.user.email.strip().lower(),
                    checkout_email,
                )
    logger.info(
        "[stripe] checkout session matched via %s profile_id=%s client_reference_id=%s",
        resolver,
        profile.id,
        client_ref,
    )
    sub_raw = _get_scalar(obj, "subscription")
    subscription_id, subscription_dict = _subscription_id_and_dict(sub_raw)
    customer = str(_get_scalar(obj, "customer") or "").strip()
    payment_link = str(_get_scalar(obj, "payment_link") or "").strip()
    resolved_price = _resolve_price_id_for_checkout_session(obj)
    if not (customer or subscription_id or payment_link):
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=profile.id,
            matched_by=resolver,
            updated_fields=[],
            reason_code="empty_update_payload",
        )
    email = str(_get_scalar(details, "email") or "").strip()
    if not email:
        email = str(_get_scalar(obj, "customer_email") or "").strip()
    if email and not profile.user.email:
        profile.user.email = email
        profile.user.save(update_fields=["email"])
    checkout_status = ""
    if subscription_id:
        if isinstance(sub_raw, dict):
            checkout_status = str(sub_raw.get("status") or "").strip().lower()
        if checkout_status not in ACTIVE_STRIPE_STATUSES:
            checkout_status = "active"
    did_update, updated_fields = apply_subscription_payload_to_profile(
        profile,
        customer_id=customer,
        subscription_id=subscription_id,
        price_id=resolved_price,
        status=checkout_status,
        payment_link_id=payment_link,
        subscription_dict=subscription_dict,
    )
    if not did_update:
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=profile.id,
            matched_by=resolver,
            updated_fields=[],
            reason_code="empty_update_payload",
        )
    return StripeSyncResult(
        handled=True,
        did_update=True,
        matched_profile_id=profile.id,
        matched_by=resolver,
        updated_fields=updated_fields,
        reason_code=None,
    )


def sync_from_invoice_paid(payload: dict, *, event_id: str = "") -> StripeSyncResult:
    payload = normalize_stripe_payload(payload)
    profile, resolver = _resolve_profile_for_event(payload)
    if profile is None:
        dbg = extract_match_debug_fields(payload)
        logger.error(
            "stripe.webhook.skipped event_id=%s event_type=%s reason_code=%s client_reference_id=%s customer=%s customer_details_email=%s",
            event_id,
            "invoice.paid",
            "missing_profile_match",
            dbg["client_reference_id"],
            dbg["customer"],
            dbg["customer_details_email"],
        )
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=None,
            matched_by="none",
            updated_fields=[],
            reason_code="missing_profile_match",
        )
    obj = _object_payload(payload)
    customer = str(_get_scalar(obj, "customer") or "").strip()
    sub_raw = _get_scalar(obj, "subscription")
    subscription_id, subscription_dict = _subscription_id_and_dict(sub_raw)
    price_id = _first_price_id_from_lines(obj)
    if not price_id and subscription_dict:
        price_id = _first_price_id_from_items(subscription_dict)
    if not (customer or subscription_id or price_id):
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=profile.id,
            matched_by=resolver,
            updated_fields=[],
            reason_code="empty_update_payload",
        )
    did_update, updated_fields = apply_subscription_payload_to_profile(
        profile,
        customer_id=customer,
        subscription_id=subscription_id,
        price_id=price_id,
        status="active",
        subscription_dict=subscription_dict,
    )
    if not did_update:
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=profile.id,
            matched_by=resolver,
            updated_fields=[],
            reason_code="empty_update_payload",
        )
    return StripeSyncResult(
        handled=True,
        did_update=True,
        matched_profile_id=profile.id,
        matched_by=resolver,
        updated_fields=updated_fields,
        reason_code=None,
    )


def sync_from_subscription(payload: dict, *, event_id: str = "") -> StripeSyncResult:
    payload = normalize_stripe_payload(payload)
    obj = _object_payload(payload)
    customer = str(_get_scalar(obj, "customer") or "").strip()
    subscription = str(_get_scalar(obj, "id") or "").strip()
    profile = None
    if customer:
        profile = (
            BusinessProfile.objects.filter(stripe_customer_id=customer)
            .order_by("-is_main", "-updated_at")
            .first()
        )
    if profile is None and subscription:
        profile = (
            BusinessProfile.objects.filter(stripe_subscription_id=subscription)
            .order_by("-is_main", "-updated_at")
            .first()
        )
    if profile is None:
        profile, resolver = _resolve_profile_for_event(payload)
    else:
        resolver = "customer/subscription"
    if profile is None:
        dbg = extract_match_debug_fields(payload)
        logger.error(
            "stripe.webhook.skipped event_id=%s event_type=%s reason_code=%s client_reference_id=%s customer=%s customer_details_email=%s",
            event_id,
            "customer.subscription.updated",
            "missing_profile_match",
            dbg["client_reference_id"],
            dbg["customer"],
            dbg["customer_details_email"],
        )
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=None,
            matched_by="none",
            updated_fields=[],
            reason_code="missing_profile_match",
        )

    status = str(_get_scalar(obj, "status") or "").strip()
    cancel_at_period_end = bool(_get_scalar(obj, "cancel_at_period_end"))
    current_period_end = _get_scalar(obj, "current_period_end", None)
    price_id = _first_price_id_from_items(obj)
    if not (customer or subscription or price_id or status):
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=profile.id,
            matched_by=resolver,
            updated_fields=[],
            reason_code="empty_update_payload",
        )
    did_update, updated_fields = apply_subscription_payload_to_profile(
        profile,
        customer_id=customer,
        subscription_id=subscription,
        price_id=price_id,
        status=status,
        current_period_end_unix=int(current_period_end) if isinstance(current_period_end, int) else None,
        cancel_at_period_end=cancel_at_period_end,
    )
    if not did_update:
        return StripeSyncResult(
            handled=False,
            did_update=False,
            matched_profile_id=profile.id,
            matched_by=resolver,
            updated_fields=[],
            reason_code="empty_update_payload",
        )
    return StripeSyncResult(
        handled=True,
        did_update=True,
        matched_profile_id=profile.id,
        matched_by=resolver,
        updated_fields=updated_fields,
        reason_code=None,
    )
