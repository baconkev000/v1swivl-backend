import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import BusinessProfile

User = get_user_model()


@pytest.mark.django_db
def test_post_second_business_profile_forbidden_without_advanced_plan():
    user = User.objects.create_user(username="multi1", email="multi1@example.com", password="pw")
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="First",
        website_url="https://first.example",
        business_address="US",
        plan=BusinessProfile.PLAN_PRO,
        stripe_subscription_status="active",
    )
    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/business-profiles/",
        {
            "business_name": "Second",
            "website_url": "https://second.example",
            "business_address": "US",
        },
        format="json",
    )
    assert resp.status_code == 403
    assert resp.data.get("error") == "advanced_plan_required"
    assert BusinessProfile.objects.filter(user=user).count() == 1


@pytest.mark.django_db
def test_post_second_business_profile_allowed_with_advanced_plan(monkeypatch):
    user = User.objects.create_user(username="multi2", email="multi2@example.com", password="pw")
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="First",
        website_url="https://first.example",
        business_address="US",
        plan=BusinessProfile.PLAN_ADVANCED,
        stripe_subscription_status="active",
    )
    full_calls: list[int] = []

    def _capture_full(profile, **kwargs):
        full_calls.append(int(profile.pk))
        return {"ok": True, "persisted": True}

    monkeypatch.setattr("accounts.views.run_full_seo_snapshot_for_profile", _capture_full)
    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/business-profiles/",
        {
            "business_name": "Second",
            "website_url": "https://second.example",
            "business_address": "US",
        },
        format="json",
    )
    assert resp.status_code == 201
    assert BusinessProfile.objects.filter(user=user).count() == 2
    assert full_calls == [int(resp.data["id"])]
