"""
Resolve which BusinessProfile a logged-in user operates on (owned vs team workspace)
and derive UI / permission flags for viewers.
"""
from __future__ import annotations

from typing import Any

from django.contrib.auth.models import AnonymousUser

from .models import BusinessProfile, BusinessProfileMembership


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


def resolve_main_business_profile_for_user(user: Any) -> BusinessProfile | None:
    """
    Prefer a team workspace the user was invited to (membership on another user's profile),
    then fall back to profiles they own.
    """
    if user is None or isinstance(user, AnonymousUser) or not user.is_authenticated:
        return None

    qs = BusinessProfileMembership.objects.filter(user=user).select_related("business_profile")
    external = qs.exclude(business_profile__user=user).order_by("pk").first()
    if external is not None:
        return external.business_profile

    owned = BusinessProfile.objects.filter(user=user)
    owned_main = owned.filter(is_main=True).first() or owned.first()
    if owned_main is not None:
        return owned_main

    any_m = qs.order_by("-is_owner", "pk").first()
    if any_m is not None:
        return any_m.business_profile
    return None


def get_membership(user: Any, profile: BusinessProfile | None) -> BusinessProfileMembership | None:
    if profile is None or user is None or not getattr(user, "is_authenticated", False):
        return None
    return BusinessProfileMembership.objects.filter(business_profile=profile, user=user).first()


def workspace_data_user(profile: BusinessProfile | None) -> Any:
    """User row used for SEO snapshots and other per-workspace caches keyed by profile owner."""
    if profile is None:
        return None
    return profile.user


def viewer_team_access(user: Any, profile: BusinessProfile | None) -> dict[str, Any]:
    """
    Flags for the authenticated viewer against the resolved BusinessProfile.

    - owner: primary account holder (``is_owner`` on membership or legacy profile owner).
    - admin: same product access as owner except we still expose ``viewer_is_main_account_owner``.
    - member: read-only company settings, no billing.
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

    m = get_membership(user, profile)
    is_row_owner = int(profile.user_id) == int(user.id)

    if m is None:
        if is_row_owner:
            return out
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

    if is_main_account_owner or is_row_owner:
        return {
            "viewer_team_role": "owner",
            "viewer_is_main_account_owner": is_main_account_owner or is_row_owner,
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
