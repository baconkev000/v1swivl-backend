from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from urllib.parse import urlparse

import stripe
from django.conf import settings

from .models import BusinessProfile

logger = logging.getLogger(__name__)

ACTIVE_STRIPE_STATUSES = frozenset({"active", "trialing", "past_due"})


@dataclass(frozen=True)
class StripePlanMapping:
    plan_slug: str
    billing_cycle: str


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
    d = _as_dict(payload)
    obj = _as_dict(d.get("object"))
    details = _as_dict(obj.get("customer_details"))
    return {
        "client_reference_id": str(obj.get("client_reference_id") or "").strip(),
        "customer": str(obj.get("customer") or "").strip(),
        "customer_details_email": str(details.get("email") or "").strip().lower(),
    }


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


def _resolve_profile_for_event(data: dict) -> BusinessProfile | None:
    d = _as_dict(data)
    obj = _as_dict(d.get("object"))
    customer_id = str(obj.get("customer") or "").strip()
    if customer_id:
        profile = (
            BusinessProfile.objects.filter(stripe_customer_id=customer_id)
            .order_by("-is_main", "-updated_at")
            .first()
        )
        if profile is not None:
            return profile
        # First-time webhook for a customer not yet linked in DB:
        # fetch customer email from Stripe and resolve profile by email.
        try:
            customer = stripe.Customer.retrieve(customer_id)
            customer_dict = _as_dict(customer)
            customer_email = str(customer_dict.get("email") or "").strip().lower()
            if customer_email:
                profile_by_email = (
                    BusinessProfile.objects.filter(user__email__iexact=customer_email)
                    .order_by("-is_main", "-updated_at")
                    .first()
                )
                if profile_by_email is not None:
                    return profile_by_email
        except Exception:
            logger.exception("[stripe] failed retrieving customer %s for profile resolution", customer_id)

    client_ref = str(obj.get("client_reference_id") or "").strip()
    if client_ref.isdigit():
        return BusinessProfile.objects.filter(id=int(client_ref)).first()

    details = _as_dict(obj.get("customer_details"))
    email = str(details.get("email") or "").strip().lower()
    if not email:
        email = str(obj.get("customer_email") or obj.get("receipt_email") or "").strip().lower()
    if email:
        return (
            BusinessProfile.objects.filter(user__email__iexact=email)
            .order_by("-is_main", "-updated_at")
            .first()
        )
    return None


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
) -> None:
    updates: dict[str, object] = {}
    if customer_id:
        updates["stripe_customer_id"] = customer_id
    if subscription_id:
        updates["stripe_subscription_id"] = subscription_id
    if price_id:
        updates["stripe_price_id"] = price_id
    if status:
        updates["stripe_subscription_status"] = status
    if current_period_end_unix is not None:
        updates["stripe_current_period_end"] = _safe_dt_from_unix(current_period_end_unix)
    if cancel_at_period_end is not None:
        updates["stripe_cancel_at_period_end"] = bool(cancel_at_period_end)

    plan = _plan_from_price(price_id)
    if not plan and payment_link_id:
        m = plan_mapping_by_payment_link_id().get(_normalize_link_id(payment_link_id))
        if m:
            plan = m.plan_slug
    if plan in {BusinessProfile.PLAN_STARTER, BusinessProfile.PLAN_PRO, BusinessProfile.PLAN_ADVANCED}:
        updates["plan"] = plan

    if not updates:
        return
    updates["updated_at"] = datetime.now(tz=timezone.utc)
    for k, v in updates.items():
        setattr(profile, k, v)
    profile.save(update_fields=list(updates.keys()))


def sync_from_checkout_session(payload: dict) -> bool:
    profile = _resolve_profile_for_event(payload)
    if profile is None:
        dbg = extract_match_debug_fields(payload)
        logger.error(
            "[stripe] checkout session: no matching profile client_reference_id=%s customer=%s customer_details.email=%s",
            dbg["client_reference_id"],
            dbg["customer"],
            dbg["customer_details_email"],
        )
        return False
    obj = _as_dict(_as_dict(payload).get("object"))
    subscription = str(obj.get("subscription") or "").strip()
    customer = str(obj.get("customer") or "").strip()
    payment_link = str(obj.get("payment_link") or "").strip()
    details = _as_dict(obj.get("customer_details"))
    email = str(details.get("email") or "").strip()
    if not email:
        email = str(obj.get("customer_email") or "").strip()
    if email and not profile.user.email:
        profile.user.email = email
        profile.user.save(update_fields=["email"])
    apply_subscription_payload_to_profile(
        profile,
        customer_id=customer,
        subscription_id=subscription,
        payment_link_id=payment_link,
    )
    return True


def sync_from_invoice_paid(payload: dict) -> bool:
    profile = _resolve_profile_for_event(payload)
    if profile is None:
        dbg = extract_match_debug_fields(payload)
        logger.error(
            "[stripe] invoice paid: no matching profile client_reference_id=%s customer=%s customer_details.email=%s",
            dbg["client_reference_id"],
            dbg["customer"],
            dbg["customer_details_email"],
        )
        return False
    obj = _as_dict(_as_dict(payload).get("object"))
    customer = str(obj.get("customer") or "").strip()
    subscription = str(obj.get("subscription") or "").strip()
    lines = _as_dict(obj.get("lines"))
    first = None
    if isinstance(lines.get("data"), list) and lines.get("data"):
        first = lines.get("data")[0]
    first_d = _as_dict(first)
    price = _as_dict(first_d.get("price"))
    price_id = str(price.get("id") or "").strip()
    apply_subscription_payload_to_profile(
        profile,
        customer_id=customer,
        subscription_id=subscription,
        price_id=price_id,
        status="active",
    )
    return True


def sync_from_subscription(payload: dict) -> bool:
    obj = _as_dict(_as_dict(payload).get("object"))
    customer = str(obj.get("customer") or "").strip()
    subscription = str(obj.get("id") or "").strip()
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
        profile = _resolve_profile_for_event(payload)
    if profile is None:
        dbg = extract_match_debug_fields(payload)
        logger.error(
            "[stripe] subscription event: no matching profile client_reference_id=%s customer=%s customer_details.email=%s",
            dbg["client_reference_id"],
            dbg["customer"],
            dbg["customer_details_email"],
        )
        return False

    status = str(obj.get("status") or "").strip()
    cancel_at_period_end = bool(obj.get("cancel_at_period_end"))
    current_period_end = obj.get("current_period_end")
    items = obj.get("items") if isinstance(obj.get("items"), dict) else {}
    first = None
    if isinstance(items.get("data"), list) and items.get("data"):
        first = items.get("data")[0]
    price = first.get("price") if isinstance(first, dict) and isinstance(first.get("price"), dict) else {}
    price_id = str(price.get("id") or "").strip()
    apply_subscription_payload_to_profile(
        profile,
        customer_id=customer,
        subscription_id=subscription,
        price_id=price_id,
        status=status,
        current_period_end_unix=int(current_period_end) if isinstance(current_period_end, int) else None,
        cancel_at_period_end=cancel_at_period_end,
    )
    return True
