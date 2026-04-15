import pytest
from django.contrib.auth import get_user_model

from accounts.business_profile_access import (
    resolve_main_business_profile_for_user,
    resolve_main_business_profile_for_user_with_source,
    should_create_owned_main_business_profile_for_user,
    user_has_external_workspace_membership,
)
from accounts.models import BusinessProfile, BusinessProfileMembership

User = get_user_model()


@pytest.mark.django_db
def test_should_create_false_when_user_has_team_membership_only():
    owner = User.objects.create_user(username="o1", email="o1@example.com", password="pw")
    member = User.objects.create_user(username="m1", email="m1@example.com", password="pw")
    bp = BusinessProfile.objects.create(user=owner, is_main=True, business_name="Acme")
    BusinessProfileMembership.objects.create(
        business_profile=bp,
        user=member,
        role=BusinessProfileMembership.ROLE_MEMBER,
        is_owner=False,
    )
    assert not BusinessProfile.objects.filter(user=member).exists()
    assert should_create_owned_main_business_profile_for_user(member) is False
    assert user_has_external_workspace_membership(member) is True
    assert resolve_main_business_profile_for_user(member).pk == bp.pk


@pytest.mark.django_db
def test_should_create_true_for_new_user_with_no_profile_or_membership():
    solo = User.objects.create_user(username="s1", email="s1@example.com", password="pw")
    assert should_create_owned_main_business_profile_for_user(solo) is True
    assert user_has_external_workspace_membership(solo) is False


@pytest.mark.django_db
def test_resolve_profile_source_prefers_external_workspace():
    owner = User.objects.create_user(username="o2", email="o2@example.com", password="pw")
    member = User.objects.create_user(username="m2", email="m2@example.com", password="pw")
    shared = BusinessProfile.objects.create(user=owner, is_main=True, business_name="Shared")
    BusinessProfileMembership.objects.create(
        business_profile=shared,
        user=member,
        role=BusinessProfileMembership.ROLE_MEMBER,
        is_owner=False,
    )
    profile, source = resolve_main_business_profile_for_user_with_source(member)
    assert profile is not None
    assert profile.id == shared.id
    assert source == "external_membership"
