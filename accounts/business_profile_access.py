"""
Resolve which BusinessProfile a logged-in user operates on (owned vs team workspace)
and derive UI / permission flags for viewers.

Organizations group all sites (BusinessProfile rows) under one billing workspace.
OrganizationMembership grants access to every site in the org (typically created when
someone is invited on the main company profile). ``hidden_from_team_ui`` only affects
customer-facing team lists, not API/workspace access (internal admins stay fully scoped).
Site-only BusinessProfileMembership on a sub-site does not imply org-wide access.
"""
from __future__ import annotations

from typing import Any

from django.contrib.auth.models import AnonymousUser
from django.db.models import Q
from django.http import HttpRequest

from .models import BusinessProfile, BusinessProfileMembership, OrganizationMembership

# Persisted per Django session: which BusinessProfile the shell should open.
ACTIVE_BUSINESS_PROFILE_SESSION_KEY = "active_business_profile_id"


def should_create_owned_main_business_profile_for_user(user: Any) -> bool:
    """
    True when we should bootstrap an owned main BusinessProfile for this user.

    Team-only accounts (already have a ``BusinessProfileMembership`` on another user's
    workspace, and no owned ``BusinessProfile``) must not get a personal profile row.
    """
    if user is None or isinstance(user, AnonymousUser) or not user.is_authenticated:
        return False
    if BusinessProfile.objects.filter(user=user).exists():
        return False
    if BusinessProfileMembership.objects.filter(user=user).exists():
        return False
    return True


def user_has_external_workspace_membership(user: Any) -> bool:
    if user is None or isinstance(user, AnonymousUser) or not user.is_authenticated:
        return False
    return BusinessProfileMembership.objects.filter(user=user).exclude(business_profile__user=user).exists()


def accessible_business_profiles_queryset(user: Any):
    """
    All BusinessProfile rows the user may operate on:

    - profiles they own (``user`` FK),
    - every profile in an organization they belong to (``OrganizationMembership``),
    - profiles they were invited to explicitly (``BusinessProfileMembership``).
    """
    if user is None or isinstance(user, AnonymousUser) or not user.is_authenticated:
        return BusinessProfile.objects.none()

    org_ids = list(
        OrganizationMembership.objects.filter(user=user).values_list("organization_id", flat=True)
    )
    q = Q(user=user)
    if org_ids:
        q |= Q(organization_id__in=org_ids)
    site_ids = BusinessProfileMembership.objects.filter(user=user).values_list(
        "business_profile_id",
        flat=True,
    )
    if site_ids:
        q |= Q(pk__in=site_ids)
    return BusinessProfile.objects.filter(q).distinct().order_by("-is_main", "created_at", "id")


def get_business_profile_for_user(user: Any, pk: int) -> BusinessProfile | None:
    return accessible_business_profiles_queryset(user).filter(pk=int(pk)).first()


def resolve_main_business_profile_for_user(user: Any) -> BusinessProfile | None:
    """
    Prefer a team workspace the user was invited to (membership on another user's profile),
    then fall back to profiles they own.
    """
    profile, _source = resolve_main_business_profile_for_user_with_source(user)
    return profile


def set_session_active_business_profile_for_user(request: HttpRequest, user: Any, profile_id: int | None) -> bool:
    """
    Store (or clear) the UI workspace selection in the session.

    ``profile_id`` must be a profile the user may access, or None to clear the override.
    Returns False if a non-None id is not accessible.
    """
    if not hasattr(request, "session"):
        return False
    if profile_id is None:
        request.session.pop(ACTIVE_BUSINESS_PROFILE_SESSION_KEY, None)
        request.session.modified = True
        return True
    prof = get_business_profile_for_user(user, int(profile_id))
    if prof is None:
        return False
    request.session[ACTIVE_BUSINESS_PROFILE_SESSION_KEY] = int(prof.pk)
    request.session.modified = True
    return True


def resolve_workspace_business_profile_for_request_with_source(request: HttpRequest) -> tuple[BusinessProfile | None, str]:
    """
    Active workspace for API handlers that have a request (session-aware).

    If the session holds a valid accessible profile id, use it; otherwise fall back to
    ``resolve_main_business_profile_for_user_with_source`` (team vs owned rules).
    """
    user = getattr(request, "user", None)
    if user is None or isinstance(user, AnonymousUser) or not user.is_authenticated:
        return None, "unauthenticated"

    raw = request.session.get(ACTIVE_BUSINESS_PROFILE_SESSION_KEY)
    if raw is not None:
        try:
            pk = int(raw)
        except (TypeError, ValueError):
            request.session.pop(ACTIVE_BUSINESS_PROFILE_SESSION_KEY, None)
            request.session.modified = True
        else:
            prof = get_business_profile_for_user(user, pk)
            if prof is not None:
                return prof, "session_active"
            request.session.pop(ACTIVE_BUSINESS_PROFILE_SESSION_KEY, None)
            request.session.modified = True

    return resolve_main_business_profile_for_user_with_source(user)


def resolve_workspace_business_profile_for_request(request: HttpRequest) -> BusinessProfile | None:
    profile, _src = resolve_workspace_business_profile_for_request_with_source(request)
    return profile


def resolve_main_business_profile_for_user_with_source(user: Any) -> tuple[BusinessProfile | None, str]:
    if user is None or isinstance(user, AnonymousUser) or not user.is_authenticated:
        return None, "unauthenticated"

    qs = BusinessProfileMembership.objects.filter(user=user).select_related("business_profile")
    visible_qs = qs.filter(hidden_from_team_ui=False)
    external = visible_qs.exclude(business_profile__user=user).order_by("pk").first()
    if external is not None:
        return external.business_profile, "external_membership"

    owned = BusinessProfile.objects.filter(user=user)
    owned_main = owned.filter(is_main=True).first() or owned.first()
    if owned_main is not None:
        return owned_main, "owned_main"

    any_visible = visible_qs.order_by("-is_owner", "pk").first()
    if any_visible is not None:
        return any_visible.business_profile, "membership_fallback_visible"

    any_m = qs.order_by("-is_owner", "pk").first()
    if any_m is not None:
        return any_m.business_profile, "membership_fallback_hidden"
    return None, "none"


def get_membership(user: Any, profile: BusinessProfile | None) -> BusinessProfileMembership | None:
    if profile is None or user is None or not getattr(user, "is_authenticated", False):
        return None
    return BusinessProfileMembership.objects.filter(business_profile=profile, user=user).first()


def get_organization_membership(user: Any, profile: BusinessProfile | None) -> OrganizationMembership | None:
    if profile is None or user is None or not getattr(user, "is_authenticated", False):
        return None
    oid = getattr(profile, "organization_id", None)
    if not oid:
        return None
    return OrganizationMembership.objects.filter(organization_id=int(oid), user=user).first()


def workspace_data_user(profile: BusinessProfile | None) -> Any:
    """User row used for SEO snapshots and other per-workspace caches keyed by profile owner."""
    if profile is None:
        return None
    return profile.user


def _viewer_access_from_org_membership(om: OrganizationMembership) -> dict[str, Any]:
    if om.is_owner:
        return {
            "viewer_team_role": "owner",
            "viewer_is_main_account_owner": True,
            "viewer_can_edit_company_profile": True,
            "viewer_can_access_billing": True,
            "viewer_can_manage_team": True,
        }
    if om.role == OrganizationMembership.ROLE_ADMIN:
        return {
            "viewer_team_role": "admin",
            "viewer_is_main_account_owner": False,
            "viewer_can_edit_company_profile": True,
            "viewer_can_access_billing": True,
            "viewer_can_manage_team": True,
        }
    return {
        "viewer_team_role": "member",
        "viewer_is_main_account_owner": False,
        "viewer_can_edit_company_profile": False,
        "viewer_can_access_billing": False,
        "viewer_can_manage_team": False,
    }


def viewer_team_access(user: Any, profile: BusinessProfile | None) -> dict[str, Any]:
    """
    Flags for the authenticated viewer against the resolved BusinessProfile.

    Precedence:
    1. Account holder for this site row (``profile.user``).
    2. Organization-level membership (all sites in the org).
    3. Per-profile team membership.
    """
    out = {
        "viewer_team_role": "owner",
        "viewer_is_main_account_owner": True,
        "viewer_can_edit_company_profile": True,
        "viewer_can_access_billing": True,
        "viewer_can_manage_team": True,
    }
    if profile is None or user is None or not getattr(user, "is_authenticated", False):
        return out

    is_row_owner = int(profile.user_id) == int(user.id)
    if is_row_owner:
        return out

    om = get_organization_membership(user, profile)
    if om is not None:
        return _viewer_access_from_org_membership(om)

    m = get_membership(user, profile)
    if m is None:
        return {
            "viewer_team_role": "member",
            "viewer_is_main_account_owner": False,
            "viewer_can_edit_company_profile": False,
            "viewer_can_access_billing": False,
            "viewer_can_manage_team": False,
        }

    is_main_account_owner = bool(m.is_owner)
    is_admin = m.role == BusinessProfileMembership.ROLE_ADMIN
    is_member = m.role == BusinessProfileMembership.ROLE_MEMBER

    if is_main_account_owner:
        return {
            "viewer_team_role": "owner",
            "viewer_is_main_account_owner": True,
            "viewer_can_edit_company_profile": True,
            "viewer_can_access_billing": True,
            "viewer_can_manage_team": True,
        }

    if is_admin:
        return {
            "viewer_team_role": "admin",
            "viewer_is_main_account_owner": False,
            "viewer_can_edit_company_profile": True,
            "viewer_can_access_billing": True,
            "viewer_can_manage_team": True,
        }

    if is_member:
        return {
            "viewer_team_role": "member",
            "viewer_is_main_account_owner": False,
            "viewer_can_edit_company_profile": False,
            "viewer_can_access_billing": False,
            "viewer_can_manage_team": False,
        }

    return out


def ensure_organization_for_first_owned_profile(profile: BusinessProfile) -> None:
    """Create organization + owner membership when the first owned profile is saved."""
    from .models import Organization

    if getattr(profile, "organization_id", None):
        return
    uid = int(profile.user_id)
    org = Organization.objects.create(
        owner_user_id=uid,
        name=str(getattr(profile, "business_name", "") or "")[:255],
    )
    BusinessProfile.objects.filter(pk=profile.pk).update(organization_id=org.id)
    profile.organization_id = org.id
    OrganizationMembership.objects.get_or_create(
        organization_id=org.id,
        user_id=uid,
        defaults={
            "role": OrganizationMembership.ROLE_ADMIN,
            "is_owner": True,
            "hidden_from_team_ui": False,
        },
    )


def attach_organization_for_additional_profile(profile: BusinessProfile) -> None:
    """Attach a newly created site to the account holder's existing organization."""
    from .models import Organization

    if getattr(profile, "organization_id", None):
        return
    uid = int(profile.user_id)
    org = Organization.objects.filter(owner_user_id=uid).order_by("id").first()
    if org is None:
        ensure_organization_for_first_owned_profile(profile)
        return
    BusinessProfile.objects.filter(pk=profile.pk).update(organization_id=org.id)
    profile.organization_id = org.id


def sync_organization_membership_for_main_team_invite(
    profile: BusinessProfile,
    invited_user: Any,
    role_raw: str,
) -> None:
    """When someone is invited on the *main* company profile, mirror access at org scope."""
    oid = getattr(profile, "organization_id", None)
    if not oid or not profile.is_main:
        return
    role_store = (
        OrganizationMembership.ROLE_ADMIN
        if role_raw == BusinessProfileMembership.ROLE_ADMIN
        else OrganizationMembership.ROLE_MEMBER
    )
    OrganizationMembership.objects.update_or_create(
        organization_id=int(oid),
        user=invited_user,
        defaults={
            "role": role_store,
            "is_owner": False,
            "hidden_from_team_ui": False,
        },
    )


def remove_organization_membership_for_main_team_leave(profile: BusinessProfile, removed_user_id: int) -> None:
    """Removing a user from the main company team revokes org-wide access for that user."""
    oid = getattr(profile, "organization_id", None)
    if not oid or not profile.is_main:
        return
    if int(removed_user_id) == int(profile.user_id):
        return
    OrganizationMembership.objects.filter(
        organization_id=int(oid),
        user_id=int(removed_user_id),
    ).delete()
