import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import BusinessProfile

User = get_user_model()


@pytest.mark.django_db
def test_local_dev_billing_complete_returns_404_when_debug_off(settings):
    settings.DEBUG = False
    settings.ALLOW_ONBOARDING_BILLING_BYPASS = False
    user = User.objects.create_user(username="ldb1@example.com", email="ldb1@example.com", password="x")
    BusinessProfile.objects.create(user=user, is_main=True, business_name="A")
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/onboarding/local-dev-billing-complete/", {"plan": "pro"}, format="json")
    assert res.status_code == 404


@pytest.mark.django_db
def test_local_dev_billing_complete_sets_fake_stripe_when_bypass_flag_on_debug_off(settings):
    settings.DEBUG = False
    settings.ALLOW_ONBOARDING_BILLING_BYPASS = True
    user = User.objects.create_user(username="ldb3@example.com", email="ldb3@example.com", password="x")
    BusinessProfile.objects.create(user=user, is_main=True, business_name="C")
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/onboarding/local-dev-billing-complete/", {"plan": "advanced"}, format="json")
    assert res.status_code == 200
    assert res.json().get("ok") is True
    p = BusinessProfile.objects.get(user=user, is_main=True)
    assert p.stripe_customer_id == "cus_local_dev"
    assert p.stripe_subscription_id == "sub_local_dev"
    assert p.stripe_subscription_status == "active"
    assert p.plan == BusinessProfile.PLAN_ADVANCED


@pytest.mark.django_db
def test_local_dev_billing_complete_sets_fake_stripe_when_debug_on(settings):
    settings.DEBUG = True
    user = User.objects.create_user(username="ldb2@example.com", email="ldb2@example.com", password="x")
    BusinessProfile.objects.create(user=user, is_main=True, business_name="B")
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/onboarding/local-dev-billing-complete/", {"plan": "advanced"}, format="json")
    assert res.status_code == 200
    assert res.json().get("ok") is True
    p = BusinessProfile.objects.get(user=user, is_main=True)
    assert p.stripe_customer_id == "cus_local_dev"
    assert p.stripe_subscription_id == "sub_local_dev"
    assert p.stripe_subscription_status == "active"
    assert p.plan == BusinessProfile.PLAN_ADVANCED


@pytest.mark.django_db
def test_local_dev_billing_complete_runs_full_seo_snapshot_when_website_present(
    settings,
    monkeypatch,
):
    settings.DEBUG = True
    full_calls: list[int] = []

    def _fake_run_full(profile, **kwargs):
        full_calls.append(int(profile.pk))
        return {"ok": True, "persisted": True}

    monkeypatch.setattr(
        "accounts.views.run_full_seo_snapshot_for_profile",
        _fake_run_full,
    )
    user = User.objects.create_user(username="ldbseo@example.com", email="ldbseo@example.com", password="x")
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="WithSite",
        website_url="https://ldbseo.example.com",
    )
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/onboarding/local-dev-billing-complete/", {"plan": "pro"}, format="json")
    assert res.status_code == 200
    assert full_calls == [BusinessProfile.objects.get(user=user, is_main=True).id]


@pytest.mark.django_db
def test_local_dev_billing_complete_with_profile_id_updates_that_profile(settings):
    settings.DEBUG = True
    user = User.objects.create_user(username="ldb4@example.com", email="ldb4@example.com", password="x")
    main_p = BusinessProfile.objects.create(user=user, is_main=True, business_name="Main")
    extra = BusinessProfile.objects.create(user=user, is_main=False, business_name="Extra")
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post(
        "/api/onboarding/local-dev-billing-complete/",
        {"plan": "starter", "profile_id": extra.id},
        format="json",
    )
    assert res.status_code == 200
    extra.refresh_from_db()
    main_p.refresh_from_db()
    assert extra.stripe_subscription_status == "active"
    assert extra.plan == BusinessProfile.PLAN_STARTER
    main_p.refresh_from_db()
    assert (main_p.stripe_customer_id or "") == ""
