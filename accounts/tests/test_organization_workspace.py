"""Organization-scoped multi-site access (org admins see every BusinessProfile in the org)."""

import pytest
from django.contrib.auth import get_user_model

from accounts.business_profile_access import accessible_business_profiles_queryset
from accounts.models import (
    BusinessProfile,
    BusinessProfileMembership,
    Organization,
    OrganizationMembership,
)

User = get_user_model()


@pytest.mark.django_db
def test_accessible_profiles_org_admin_sees_all_sites():
    owner = User.objects.create_user(username="orgown1", email="orgown1@example.com", password="pw")
    admin = User.objects.create_user(username="orgadm1", email="orgadm1@example.com", password="pw")
    org = Organization.objects.create(owner_user=owner, name="Acme Org")
    BusinessProfile.objects.create(
        user=owner,
        is_main=True,
        organization=org,
        business_name="Main",
        website_url="https://main.example",
        business_address="US",
    )
    BusinessProfile.objects.create(
        user=owner,
        is_main=False,
        organization=org,
        business_name="Sub",
        website_url="https://sub.example",
        business_address="US",
    )
    OrganizationMembership.objects.create(
        organization=org,
        user=admin,
        role=OrganizationMembership.ROLE_ADMIN,
        is_owner=False,
    )
    qs = accessible_business_profiles_queryset(admin)
    assert qs.count() == 2


@pytest.mark.django_db
def test_accessible_profiles_site_only_admin_sees_one_site():
    owner = User.objects.create_user(username="orgown2", email="orgown2@example.com", password="pw")
    admin = User.objects.create_user(username="orgadm2", email="orgadm2@example.com", password="pw")
    org = Organization.objects.create(owner_user=owner, name="Beta Org")
    main = BusinessProfile.objects.create(
        user=owner,
        is_main=True,
        organization=org,
        business_name="Main",
        website_url="https://main2.example",
        business_address="US",
    )
    sub = BusinessProfile.objects.create(
        user=owner,
        is_main=False,
        organization=org,
        business_name="Sub",
        website_url="https://sub2.example",
        business_address="US",
    )
    BusinessProfileMembership.objects.create(
        business_profile=sub,
        user=admin,
        role=BusinessProfileMembership.ROLE_ADMIN,
        is_owner=False,
    )
    qs = accessible_business_profiles_queryset(admin)
    assert set(qs.values_list("id", flat=True)) == {sub.id}
    assert main.id not in set(qs.values_list("id", flat=True))
