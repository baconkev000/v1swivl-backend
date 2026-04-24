import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.aeo.aeo_prompts import AEOPromptType
from accounts.aeo.aeo_plan_targets import aeo_effective_custom_prompt_cap_for_profile
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


@pytest.mark.django_db
def test_delete_suggested_prompt_increments_custom_cap_bonus():
    user = User.objects.create_user(username="del-suggested@example.com", email="del-suggested@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        plan=BusinessProfile.PLAN_STARTER,
        selected_aeo_prompts=[
            prompt_record(
                "Best accounting software for agencies",
                prompt_type=AEOPromptType.TRANSACTIONAL,
                dynamic=True,
                is_custom=False,
            ),
            prompt_record(
                "What is the best payroll software for startups?",
                prompt_type=AEOPromptType.TRANSACTIONAL,
                dynamic=True,
                is_custom=True,
            ),
        ],
    )
    before_cap = aeo_effective_custom_prompt_cap_for_profile(profile)
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.delete(
        "/api/aeo/monitored-prompts/",
        {"prompt": "Best accounting software for agencies"},
        format="json",
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["custom_prompt_bonus"] == 1
    profile.refresh_from_db()
    assert int(profile.aeo_custom_prompt_cap_bonus or 0) == 1
    assert aeo_effective_custom_prompt_cap_for_profile(profile) == before_cap + 1
    saved = list(profile.selected_aeo_prompts or [])
    assert all("Best accounting software for agencies" != str((x or {}).get("prompt") if isinstance(x, dict) else x) for x in saved)


@pytest.mark.django_db
def test_delete_suggested_matches_whitespace_normalized():
    """DELETE body collapses whitespace to match legacy string rows (not only dict records)."""
    user = User.objects.create_user(username="del-ws@example.com", email="del-ws@example.com", password="pw")
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        plan=BusinessProfile.PLAN_STARTER,
        selected_aeo_prompts=["Best  accounting\nsoftware\tfor agencies"],
    )
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.delete(
        "/api/aeo/monitored-prompts/",
        {"prompt": "Best accounting software for agencies"},
        format="json",
    )
    assert res.status_code == 200
    assert res.json().get("ok") is True


@pytest.mark.django_db
def test_sanitize_prompt_coverage_monitored_flags_clears_false_monitored():
    from accounts.views import _sanitize_prompt_coverage_monitored_flags

    user = User.objects.create_user(username="san-mon@example.com", email="san-mon@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        plan=BusinessProfile.PLAN_STARTER,
        selected_aeo_prompts=["Stored prompt alpha"],
    )
    payload = {
        "prompts": [
            {"prompt": "Stored prompt alpha", "monitored": True},
            {"prompt": "Snapshot only beta", "monitored": True},
        ]
    }
    _sanitize_prompt_coverage_monitored_flags(profile, payload)
    rows = payload["prompts"]
    assert rows[0]["monitored"] is True
    assert rows[1]["monitored"] is False


@pytest.mark.django_db
def test_delete_custom_prompt_is_rejected():
    user = User.objects.create_user(username="del-custom@example.com", email="del-custom@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        plan=BusinessProfile.PLAN_STARTER,
        selected_aeo_prompts=[
            prompt_record(
                "Custom prompt to keep",
                prompt_type=AEOPromptType.TRANSACTIONAL,
                dynamic=True,
                is_custom=True,
            ),
        ],
    )
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.delete(
        "/api/aeo/monitored-prompts/",
        {"prompt": "Custom prompt to keep"},
        format="json",
    )
    assert res.status_code == 400
    assert "cannot be deleted" in (res.json().get("error") or "").lower()
    profile.refresh_from_db()
    assert int(profile.aeo_custom_prompt_cap_bonus or 0) == 0
