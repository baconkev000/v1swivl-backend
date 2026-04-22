import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.aeo.aeo_prompts import AEOPromptType
from accounts.aeo.aeo_utils import prompt_record
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
def test_advanced_plan_blocks_when_at_total_prompt_limit(monkeypatch):
    """Advanced: 150 total monitored = 150 custom cap; 151st append hits total limit."""
    user = User.objects.create_user(username="advcap@example.com", email="advcap@example.com", password="pw")
    custom_rows = [
        prompt_record(
            f"Custom prompt {i} for advanced cap test",
            prompt_type=AEOPromptType.TRANSACTIONAL,
            weight=1.0,
            dynamic=True,
            is_custom=True,
        )
        for i in range(149)
    ]
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        plan=BusinessProfile.PLAN_ADVANCED,
        selected_aeo_prompts=custom_rows,
    )
    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", lambda *a, **k: None)

    client = APIClient()
    client.force_authenticate(user=user)
    res_ok = client.post(
        "/api/aeo/monitored-prompts/",
        {"prompt": "One hundred fiftieth unique custom prompt for limit test"},
        format="json",
    )
    assert res_ok.status_code == 202

    res = client.post(
        "/api/aeo/monitored-prompts/",
        {"prompt": "One hundred fifty first should be rejected"},
        format="json",
    )
    assert res.status_code == 400
    err = res.json().get("error", "").lower()
    assert "at most 150" in err
