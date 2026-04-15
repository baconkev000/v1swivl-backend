import pytest
from allauth.socialaccount.models import SocialAccount
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.business_profile_access import should_create_owned_main_business_profile_for_user
from accounts.models import BusinessProfile, BusinessProfileMembership
from accounts.onboarding_completion import user_has_completed_full_onboarding
from accounts.user_identity_reconciliation import reconcile_user_identity_for_email

User = get_user_model()


@pytest.mark.django_db
def test_reconcile_duplicate_email_prefers_membership_user_and_moves_social_account():
    owner = User.objects.create_user(username="owner_ws", email="owner_ws@example.com", password="pw")
    membership_user = User(username="team@example.com", email="team@example.com")
    membership_user.set_unusable_password()
    membership_user.save()
    BusinessProfile.objects.create(user=owner, is_main=True, business_name="Owner Workspace")
    BusinessProfileMembership.objects.create(
        business_profile=BusinessProfile.objects.get(user=owner),
        user=membership_user,
        role=BusinessProfileMembership.ROLE_MEMBER,
        is_owner=False,
    )

    dup_social = User.objects.create_user(username="team-social", email="team@example.com", password="pw")
    SocialAccount.objects.create(user=dup_social, provider="google", uid="google-team-1")

    result = reconcile_user_identity_for_email(
        "team@example.com",
        preferred_user=dup_social,
        reason="test_duplicate_split",
    )
    assert result.user is not None
    assert result.user.id == membership_user.id
    assert dup_social.id in result.merged_user_ids
    assert SocialAccount.objects.filter(user=membership_user, provider="google", uid="google-team-1").exists()
    assert user_has_completed_full_onboarding(membership_user) is True
    assert should_create_owned_main_business_profile_for_user(membership_user) is False


@pytest.mark.django_db
def test_business_profile_api_reconciles_duplicate_session_user_to_team_workspace():
    owner = User.objects.create_user(username="owner2", email="owner2@example.com", password="pw")
    owner_profile = BusinessProfile.objects.create(user=owner, is_main=True, business_name="Owner Workspace")

    invited = User(username="split@example.com", email="split@example.com")
    invited.set_unusable_password()
    invited.save()
    BusinessProfileMembership.objects.create(
        business_profile=owner_profile,
        user=invited,
        role=BusinessProfileMembership.ROLE_ADMIN,
        is_owner=False,
    )

    # Duplicate user can authenticate and would normally resolve to a personal workspace.
    dup = User.objects.create_user(username="split-login", email="split@example.com", password="pw")
    BusinessProfile.objects.create(user=dup, is_main=True, business_name="Personal Empty")

    client = APIClient()
    assert client.login(username="split-login", password="pw")
    res = client.get("/api/business-profile/?skip_heavy=1")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == owner_profile.id
    assert body["business_name"] == "Owner Workspace"


@pytest.mark.django_db
def test_owner_and_solo_users_keep_owned_workspace_behavior():
    owner = User.objects.create_user(username="solo-owner", email="solo-owner@example.com", password="pw")
    owned = BusinessProfile.objects.create(user=owner, is_main=True, business_name="Owned Workspace")
    result = reconcile_user_identity_for_email(owner.email, preferred_user=owner, reason="owner_solo")
    assert result.user is not None
    assert result.user.id == owner.id
    client = APIClient()
    client.force_authenticate(user=owner)
    res = client.get("/api/business-profile/?skip_heavy=1")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == owned.id
    assert body["business_name"] == "Owned Workspace"
