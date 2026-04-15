import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import BusinessProfile

User = get_user_model()


@pytest.mark.django_db
def test_pro_plan_allows_custom_append_beyond_50_total(monkeypatch):
    user = User.objects.create_user(username="procap@example.com", email="procap@example.com", password="pw")
    selected = [f"Prompt {i}" for i in range(50)]
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        plan=BusinessProfile.PLAN_PRO,
        selected_aeo_prompts=selected,
    )
    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", lambda *a, **k: None)

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post(
        "/api/aeo/monitored-prompts/",
        {"prompt": "What is a good resume writing tool that does all the work for me?"},
        format="json",
    )
    assert res.status_code == 202
    body = res.json()
    assert body["ok"] is True
    assert body["monitored_count"] == 51
    profile.refresh_from_db()
    assert len(profile.selected_aeo_prompts or []) == 51


@pytest.mark.django_db
def test_advanced_custom_cap_is_50(monkeypatch):
    user = User.objects.create_user(username="advcap@example.com", email="advcap@example.com", password="pw")
    base = [f"Base Prompt {i}" for i in range(99)]
    custom = [
        {
            "prompt": f"Custom Prompt {i}",
            "type": "transactional",
            "weight": 1.0,
            "dynamic": True,
            "is_custom": True,
        }
        for i in range(50)
    ]
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        plan=BusinessProfile.PLAN_ADVANCED,
        selected_aeo_prompts=base + custom,
    )
    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", lambda *a, **k: None)

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post(
        "/api/aeo/monitored-prompts/",
        {"prompt": "One more custom prompt"},
        format="json",
    )
    assert res.status_code == 400
    assert "at most 50 custom prompts" in res.json().get("error", "").lower()
