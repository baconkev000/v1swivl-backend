"""
Plan-derived monitored AEO prompt targets (production).

Total monitored prompts (generated + onboarding seed) per plan:
  Starter: 25 | Pro: 75 | Advanced: 150

User-authored custom monitored prompt caps match those totals:
  Starter: 25 | Pro: 75 | Advanced: 150

Onboarding still delivers ``AEO_ONBOARDING_BASELINE_MONITORED_PROMPT_COUNT`` (10) prompts;
completion gates use that baseline, not the full plan cap.

PLAN_NONE and starter both use the starter caps.

Testing: AEO_TESTING_MODE + AEO_TEST_PROMPT_COUNT override plan (keeps tests small).
"""
from __future__ import annotations

from typing import Any

from django.conf import settings

from accounts.models import BusinessProfile

_ACTIVE_SUBSCRIPTION_STATUSES = frozenset({"active", "trialing", "past_due"})

AEO_PLAN_CAP_STARTER: int = 25
AEO_PLAN_CAP_PRO: int = 75
AEO_PLAN_CAP_ADVANCED: int = 150

AEO_CUSTOM_PROMPT_CAP_STARTER: int = 25
AEO_CUSTOM_PROMPT_CAP_PRO: int = 75
AEO_CUSTOM_PROMPT_CAP_ADVANCED: int = 150

# Delivered in onboarding; used for ``business_profile_fully_onboarded`` / serializer min, not plan cap.
AEO_ONBOARDING_BASELINE_MONITORED_PROMPT_COUNT: int = 10


def aeo_monitored_prompt_cap_for_plan_slug(plan: str) -> int:
    p = (plan or "").strip().lower()
    if p == BusinessProfile.PLAN_PRO:
        return AEO_PLAN_CAP_PRO
    if p == BusinessProfile.PLAN_ADVANCED:
        return AEO_PLAN_CAP_ADVANCED
    return AEO_PLAN_CAP_STARTER


def _normalized_plan_slug(raw: str) -> str:
    p = (raw or "").strip().lower()
    if p == BusinessProfile.PLAN_PRO:
        return BusinessProfile.PLAN_PRO
    if p in {BusinessProfile.PLAN_ADVANCED, "enterprise", "scale"}:
        return BusinessProfile.PLAN_ADVANCED
    if p == BusinessProfile.PLAN_STARTER:
        return BusinessProfile.PLAN_STARTER
    return BusinessProfile.PLAN_NONE


def _effective_plan_slug_for_profile(profile: BusinessProfile | None) -> str:
    """
    Resolve plan for feature gating and target sizing.

    For add-company flows, newly created profiles may intentionally keep ``plan=""`` until
    onboarding/payment completes. In that case, inherit the owning account's effective paid
    tier from the user's primary profile so caps/expansion still align with account billing.
    """
    if profile is None:
        return BusinessProfile.PLAN_NONE
    own = _normalized_plan_slug(str(getattr(profile, "plan", "") or ""))
    if own != BusinessProfile.PLAN_NONE:
        return own

    owner_id = getattr(profile, "user_id", None)
    if not owner_id:
        return BusinessProfile.PLAN_NONE
    anchor = (
        BusinessProfile.objects.filter(user_id=owner_id)
        .order_by("-is_main", "id")
        .only("plan", "stripe_subscription_status")
        .first()
    )
    if anchor is None:
        return BusinessProfile.PLAN_NONE
    anchor_plan = _normalized_plan_slug(str(getattr(anchor, "plan", "") or ""))
    if anchor_plan != BusinessProfile.PLAN_NONE:
        return anchor_plan
    status = str(getattr(anchor, "stripe_subscription_status", "") or "").strip().lower()
    if status in _ACTIVE_SUBSCRIPTION_STATUSES:
        return BusinessProfile.PLAN_STARTER
    return BusinessProfile.PLAN_NONE


def aeo_custom_monitored_prompt_cap_for_plan_slug(plan: str) -> int:
    p = (plan or "").strip().lower()
    if p == BusinessProfile.PLAN_PRO:
        return AEO_CUSTOM_PROMPT_CAP_PRO
    if p == BusinessProfile.PLAN_ADVANCED:
        return AEO_CUSTOM_PROMPT_CAP_ADVANCED
    return AEO_CUSTOM_PROMPT_CAP_STARTER


def aeo_testing_mode() -> bool:
    return bool(getattr(settings, "AEO_TESTING_MODE", False))


def aeo_testing_target_count() -> int:
    try:
        return max(1, int(getattr(settings, "AEO_TEST_PROMPT_COUNT", 10)))
    except (TypeError, ValueError):
        return 10


def aeo_fallback_global_target_count() -> int:
    """When no profile is available (legacy call sites)."""
    if aeo_testing_mode():
        return aeo_testing_target_count()
    return AEO_PLAN_CAP_STARTER


def aeo_effective_monitored_target_for_profile(profile: BusinessProfile | None) -> int:
    if profile is None:
        return aeo_fallback_global_target_count()
    if aeo_testing_mode():
        return aeo_testing_target_count()
    return aeo_monitored_prompt_cap_for_plan_slug(_effective_plan_slug_for_profile(profile))


def aeo_effective_custom_prompt_cap_for_profile(profile: BusinessProfile | None) -> int:
    """
    Max user-authored (``is_custom``) monitored prompts for the plan.

    In testing mode, caps at the effective monitored target so append flows stay small.
    """
    base = aeo_custom_monitored_prompt_cap_for_plan_slug(
        _effective_plan_slug_for_profile(profile) if profile is not None else ""
    )
    if profile is None:
        return base
    if aeo_testing_mode():
        return min(base, aeo_effective_monitored_target_for_profile(profile))
    return base


def aeo_onboarding_complete_min_prompts(profile: BusinessProfile) -> int:
    """
    Minimum prompts required for business_profile_fully_onboarded.
    Production: onboarding baseline (10) or plan cap if lower. Testing: full test target.
    """
    cap = aeo_effective_monitored_target_for_profile(profile)
    if aeo_testing_mode():
        return cap
    return min(AEO_ONBOARDING_BASELINE_MONITORED_PROMPT_COUNT, cap)


def aeo_effective_cap_for_validation(instance: BusinessProfile | None, attrs: dict[str, Any]) -> int:
    """Cap for serializer validation (honors plan change in same PATCH when not in testing mode)."""
    if aeo_testing_mode():
        return aeo_testing_target_count()
    plan = BusinessProfile.PLAN_NONE
    if instance is not None:
        plan = str(attrs.get("plan", instance.plan) or instance.plan or "")
    else:
        plan = str(attrs.get("plan") or BusinessProfile.PLAN_NONE)
    return aeo_monitored_prompt_cap_for_plan_slug(plan)


def aeo_onboarding_min_for_validation(instance: BusinessProfile | None, attrs: dict[str, Any]) -> int:
    if aeo_testing_mode():
        return aeo_effective_cap_for_validation(instance, attrs)
    cap = aeo_effective_cap_for_validation(instance, attrs)
    return min(AEO_ONBOARDING_BASELINE_MONITORED_PROMPT_COUNT, cap)


def aeo_should_run_post_payment_expansion(profile: BusinessProfile) -> bool:
    if aeo_testing_mode():
        return False
    status = str(getattr(profile, "stripe_subscription_status", "") or "").strip().lower()
    if status not in _ACTIVE_SUBSCRIPTION_STATUSES:
        return False
    return _effective_plan_slug_for_profile(profile) in {
        BusinessProfile.PLAN_PRO,
        BusinessProfile.PLAN_ADVANCED,
    }


def aeo_http_call_bounds_for_monitoring(
    prompt_count: int,
    *,
    providers: int = 2,
    min_passes: int = 2,
    max_passes: int = 3,
) -> dict[str, int]:
    """
    Rough provider HTTP call bounds for observability (2 passes typical; 3 if unstable).
    Marketing-facing headlines can diverge; logs should record these numbers.
    """
    return {
        "monitored_prompts": int(prompt_count),
        "providers": int(providers),
        "passes_min": int(min_passes),
        "passes_max": int(max_passes),
        "http_calls_approx_lo": int(prompt_count * providers * min_passes),
        "http_calls_approx_hi": int(prompt_count * providers * max_passes),
    }
