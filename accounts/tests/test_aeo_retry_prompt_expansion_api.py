import pytest

try:
    import stripe  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("stripe is required for accounts.views URL tests", allow_module_level=True)

from django.contrib.auth import get_user_model
from django.core.cache import cache
from rest_framework.test import APIClient

from accounts.models import BusinessProfile

User = get_user_model()


@pytest.mark.django_db
def test_retry_prompt_expansion_pro_enqueues(monkeypatch):
    cache.clear()

    user = User.objects.create_user(username="rpe1", email="rpe1@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        plan=BusinessProfile.PLAN_PRO,
        selected_aeo_prompts=["a", "b"],
    )

    captured: dict = {}

    def fake_delay(pid, expected_plan_slug=None, expansion_cap=None):
        captured["profile_id"] = pid
        captured["expected_plan_slug"] = expected_plan_slug
        captured["expansion_cap"] = expansion_cap

    monkeypatch.setattr("accounts.tasks.schedule_aeo_prompt_plan_expansion.delay", fake_delay)
    monkeypatch.setattr("accounts.views.transaction.on_commit", lambda fn: fn())

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/aeo/retry-prompt-expansion/", {}, format="json")
    assert res.status_code == 200
    assert res.json()["enqueued"] is True
    assert captured["profile_id"] == profile.id
    assert captured["expected_plan_slug"] == BusinessProfile.PLAN_PRO
    assert captured["expansion_cap"] == 75


@pytest.mark.django_db
def test_retry_prompt_expansion_starter_enqueues(monkeypatch):
    cache.clear()

    user = User.objects.create_user(username="rpe2", email="rpe2@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        plan=BusinessProfile.PLAN_STARTER,
        stripe_subscription_status="active",
        selected_aeo_prompts=["a"],
    )
    captured: dict = {}

    def fake_delay(pid, expected_plan_slug=None, expansion_cap=None):
        captured["profile_id"] = pid
        captured["expected_plan_slug"] = expected_plan_slug
        captured["expansion_cap"] = expansion_cap

    monkeypatch.setattr("accounts.tasks.schedule_aeo_prompt_plan_expansion.delay", fake_delay)
    monkeypatch.setattr("accounts.views.transaction.on_commit", lambda fn: fn())

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/aeo/retry-prompt-expansion/", {}, format="json")
    assert res.status_code == 200
    assert res.json()["enqueued"] is True
    assert captured["profile_id"] == profile.id
    assert captured["expected_plan_slug"] == BusinessProfile.PLAN_STARTER
    assert captured["expansion_cap"] == 25


@pytest.mark.django_db
def test_retry_prompt_expansion_rate_limited(monkeypatch):
    cache.clear()

    user = User.objects.create_user(username="rpe3", email="rpe3@example.com", password="pw")
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        plan=BusinessProfile.PLAN_ADVANCED,
        selected_aeo_prompts=["a"],
    )

    monkeypatch.setattr("accounts.tasks.schedule_aeo_prompt_plan_expansion.delay", lambda *a, **k: None)
    monkeypatch.setattr("accounts.views.transaction.on_commit", lambda fn: fn())

    client = APIClient()
    client.force_authenticate(user=user)
    assert client.post("/api/aeo/retry-prompt-expansion/", {}, format="json").status_code == 200
    assert client.post("/api/aeo/retry-prompt-expansion/", {}, format="json").status_code == 429
