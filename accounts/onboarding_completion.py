"""
Decide if a user has finished onboarding (profile + keywords + AEO prompts + AEO scan pipeline).

Used by auth/status, post-login redirects, and should stay aligned with onboarding hydration.
"""
from __future__ import annotations

from .dataforseo_utils import normalize_domain
from .business_profile_access import (
    resolve_main_business_profile_for_user,
    user_has_external_workspace_membership,
)
from .models import (
    AEOExecutionRun,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    BusinessProfile,
    OnboardingOnPageCrawl,
    SEOOverviewSnapshot,
)
from .aeo.aeo_plan_targets import (
    aeo_effective_monitored_target_for_profile,
    aeo_onboarding_complete_min_prompts,
)

ACTIVE_SUBSCRIPTION_STATUSES = frozenset({"active", "trialing", "past_due"})


def profile_has_active_subscription(profile: BusinessProfile) -> bool:
    """
    Stripe is source of truth for billing state (not BusinessProfile.plan).
    Plan may be PLAN_NONE while status is still active during edge cases; prefer this for access gates.
    """
    status = str(profile.stripe_subscription_status or "").strip().lower()
    return status in ACTIVE_SUBSCRIPTION_STATUSES


def effective_dashboard_plan_slug_for_owned_business_profiles(user) -> str:
    """
    Effective paid tier for the account holder's owned profiles.

    Mirrors the frontend ``dashboardPlanSlug(profile, subscription)`` rule: prefer
    ``is_main``, then oldest by id; treat active Stripe with empty plan as starter.
    """
    if user is None or not getattr(user, "is_authenticated", False):
        return "none"
    qs = BusinessProfile.objects.filter(user=user)
    bp = qs.filter(is_main=True).first() or qs.order_by("id").first()
    if bp is None:
        return "none"
    raw = str(bp.plan or "").strip().lower()
    sub_active = profile_has_active_subscription(bp)
    if raw in ("pro", "professional"):
        return "pro"
    if raw in ("advanced", "enterprise", "scale"):
        return "advanced"
    if raw == "starter":
        return "starter"
    if sub_active:
        return "starter"
    return "none"


def user_may_create_additional_business_profile(user) -> bool:
    """Second and subsequent *owned* company profiles require an Advanced plan."""
    return effective_dashboard_plan_slug_for_owned_business_profiles(user) == "advanced"


def profile_has_ranked_keywords(profile: BusinessProfile) -> bool:
    """
    True if we have a non-empty keyword list from onboarding Labs crawl or SEO snapshot cache.
    """
    domain = normalize_domain(profile.website_url or "")
    if domain:
        crawl = (
            OnboardingOnPageCrawl.objects.filter(
                user=profile.user,
                domain=domain,
                status=OnboardingOnPageCrawl.STATUS_COMPLETED,
            )
            .order_by("-created_at")
            .first()
        )
        if crawl is not None:
            rk = crawl.ranked_keywords
            if isinstance(rk, list) and len(rk) > 0:
                return True
            rt = crawl.review_topics
            if isinstance(rt, list) and len(rt) > 0:
                return True
    snap = (
        SEOOverviewSnapshot.objects.filter(business_profile=profile)
        .order_by("-last_fetched_at", "-id")
        .only("top_keywords")
        .first()
    )
    if snap is not None:
        tk = getattr(snap, "top_keywords", None) or []
        if isinstance(tk, list) and len(tk) > 0:
            return True
    return False


def profile_has_aeo_response_snapshots(profile: BusinessProfile) -> bool:
    return AEOResponseSnapshot.objects.filter(profile=profile).exists()


def profile_has_aeo_extraction_snapshots(profile: BusinessProfile) -> bool:
    return AEOExtractionSnapshot.objects.filter(response_snapshot__profile=profile).exists()


def profile_has_valid_aeo_run_artifacts(profile: BusinessProfile) -> bool:
    """
    Redirect/readiness guard: require persisted response + extraction artifacts tied to a completed run.
    Aggregate rows alone are not sufficient.
    """
    run = (
        AEOExecutionRun.objects.filter(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
        .order_by("-created_at")
        .first()
    )
    if run is None:
        return False
    response_count = AEOResponseSnapshot.objects.filter(profile=profile, execution_run=run).count()
    extraction_count = AEOExtractionSnapshot.objects.filter(
        response_snapshot__profile=profile,
        response_snapshot__execution_run=run,
    ).count()
    minimum = max(1, int(run.prompt_count_executed or 0))
    if response_count < minimum or extraction_count < minimum:
        return False
    return True


def business_profile_fully_onboarded(profile: BusinessProfile | None) -> bool:
    """
    Full onboarding: active subscription, profile fields, keywords, and AEO pipeline artifacts.

    Monitored prompts: must be at least the onboarding baseline (10 in production) and at most
    the plan cap. Pro/Advanced may still be expanding toward their plan totals in the background;
    users with 10 prompts from onboarding and an active paid plan still qualify once the rest of
    the gates pass.
    """
    if profile is None:
        return False
    if not (
        (profile.business_name or "").strip()
        and (profile.website_url or "").strip()
        and (profile.business_address or "").strip()
    ):
        return False

    cap = aeo_effective_monitored_target_for_profile(profile)
    need_min = aeo_onboarding_complete_min_prompts(profile)
    prompts = profile.selected_aeo_prompts or []
    prompts = [str(x).strip() for x in prompts if str(x).strip()]
    n = len(prompts)
    if n < need_min or n > cap:
        return False
    if not profile_has_active_subscription(profile):
        return False

    domain = normalize_domain(profile.website_url or "")
    if not domain:
        return False

    if not profile_has_ranked_keywords(profile):
        return False

    if not profile_has_valid_aeo_run_artifacts(profile):
        return False

    return True


def user_has_completed_full_onboarding(user) -> bool:
    if user is None or not user.is_authenticated:
        return False
    if user_has_external_workspace_membership(user):
        return True
    profile = resolve_main_business_profile_for_user(user)
    if business_profile_fully_onboarded(profile):
        return True
    # Multi-workspace: switching `is_main` to a newer profile can leave the new main
    # still running the AEO pipeline while another owned profile is already complete.
    # Middleware and login must still allow /app so the user is not forced back through
    # first-time onboarding for an established account.
    owned_qs = BusinessProfile.objects.filter(user=user).order_by("-id")
    main_id = getattr(profile, "pk", None)
    if main_id is not None:
        owned_qs = owned_qs.exclude(pk=main_id)
    for bp in owned_qs:
        if business_profile_fully_onboarded(bp):
            return True
    return False
