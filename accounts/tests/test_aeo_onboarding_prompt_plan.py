import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import AEOExecutionRun, BusinessProfile


User = get_user_model()


@pytest.mark.django_db
def test_aeo_onboarding_prompt_plan_requires_auth():
    client = APIClient()
    resp = client.get("/api/aeo/onboarding-prompt-plan/")
    assert resp.status_code == 403


@pytest.mark.django_db
def test_aeo_onboarding_prompt_plan_returns_groups_and_combined(monkeypatch):
    user = User.objects.create_user(
        username="u-aeo-onb",
        email="u-aeo-onb@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Test Dental",
        business_address="123 Main St, Salt Lake City, UT",
        industry="dental",
        website_url="https://example.com",
    )

    client = APIClient()
    client.force_authenticate(user=user)
    prompts = [
        {
            "prompt": f"openai prompt {i}",
            "type": "transactional" if i % 2 == 0 else "trust",
            "weight": 1.0,
            "dynamic": True,
        }
        for i in range(20)
    ]
    monkeypatch.setattr(
        "accounts.aeo.aeo_utils.run_prompt_batch_via_openai",
        lambda *args, **kwargs: prompts[: kwargs.get("max_additional", 0)],
    )
    resp = client.get("/api/aeo/onboarding-prompt-plan/?include_openai=0")
    assert resp.status_code == 200
    body = resp.json()
    assert "groups" in body
    assert "combined" in body
    assert "business" in body
    assert "meta" in body
    assert body["meta"].get("openai_status") in {"ok", "partial"}
    assert "fixed" in body["groups"]
    assert "dynamic" in body["groups"]
    assert "openai_generated" in body["groups"]
    assert isinstance(body["groups"]["fixed"], list)
    assert isinstance(body["combined"], list)
    assert body["groups"]["fixed"] == []
    assert body["groups"]["dynamic"] == []
    assert body["groups"]["openai_generated"] == body["combined"]
    if body["combined"]:
        row = body["combined"][0]
        assert "prompt" in row
        assert "type" in row
        assert "weight" in row
        assert "dynamic" in row


@pytest.mark.django_db
def test_aeo_onboarding_prompt_plan_reuse_saved_enqueues_on_commit(monkeypatch, settings):
    user = User.objects.create_user(
        username="u-aeo-onb-reuse",
        email="u-aeo-onb-reuse@example.com",
        password="pw",
    )
    saved_prompts = [f"saved prompt {i}" for i in range(10)]
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Test Business",
        business_address="123 Main St, Salt Lake City, UT",
        industry="dental",
        website_url="https://example.com",
        selected_aeo_prompts=saved_prompts,
    )

    callbacks = []
    delayed_calls = []

    def _capture_on_commit(cb):
        callbacks.append(cb)

    def _capture_delay(run_id, prompt_payload):
        delayed_calls.append((run_id, prompt_payload))

    monkeypatch.setattr("accounts.views.transaction.on_commit", _capture_on_commit)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", _capture_delay)

    client = APIClient()
    client.force_authenticate(user=user)
    settings.AEO_TESTING_MODE = True
    settings.AEO_TEST_PROMPT_COUNT = 10
    resp = client.get("/api/aeo/onboarding-prompt-plan/?include_openai=0&reuse_saved=1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["openai_status"] == "reused_saved"
    assert len(callbacks) == 1
    assert delayed_calls == []

    callbacks[0]()
    assert len(delayed_calls) == 1
    run_id, payload = delayed_calls[0]
    assert isinstance(run_id, int)
    run = AEOExecutionRun.objects.get(id=run_id)
    assert run.prompt_count_requested == len(body["combined"])
    assert payload == body["combined"]


@pytest.mark.django_db
def test_aeo_onboarding_prompt_plan_generated_enqueues_on_commit(monkeypatch, settings):
    user = User.objects.create_user(
        username="u-aeo-onb-generated",
        email="u-aeo-onb-generated@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Test Business",
        business_address="123 Main St, Salt Lake City, UT",
        industry="dental",
        website_url="https://example.com",
    )

    callbacks = []
    delayed_calls = []

    def _capture_on_commit(cb):
        callbacks.append(cb)

    def _capture_delay(run_id, prompt_payload):
        delayed_calls.append((run_id, prompt_payload))

    monkeypatch.setattr("accounts.views.transaction.on_commit", _capture_on_commit)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", _capture_delay)

    client = APIClient()
    client.force_authenticate(user=user)
    settings.AEO_TESTING_MODE = True
    settings.AEO_TEST_PROMPT_COUNT = 10

    generated = [
        {
            "prompt": f"generated prompt {i}",
            "type": "transactional",
            "weight": 1.0,
            "dynamic": True,
        }
        for i in range(20)
    ]

    def _mock_openai(*args, **kwargs):
        take = int(kwargs.get("max_additional", 0))
        return generated[:take]

    monkeypatch.setattr("accounts.aeo.aeo_utils.run_prompt_batch_via_openai", _mock_openai)
    resp = client.get("/api/aeo/onboarding-prompt-plan/?include_openai=0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["openai_status"] == "ok"
    assert body["groups"]["fixed"] == []
    assert body["groups"]["dynamic"] == []
    assert body["groups"]["openai_generated"] == body["combined"]
    assert len(callbacks) == 1
    assert delayed_calls == []

    callbacks[0]()
    assert len(delayed_calls) == 1
    run_id, payload = delayed_calls[0]
    assert isinstance(run_id, int)
    run = AEOExecutionRun.objects.get(id=run_id)
    assert run.prompt_count_requested == min(len(body["combined"]), int(body["meta"]["combined_target"]))
    assert payload == body["combined"]


@pytest.mark.django_db
def test_aeo_onboarding_prompt_plan_openai_only_hits_exact_target(monkeypatch, settings):
    user = User.objects.create_user(
        username="u-aeo-onb-openai-exact",
        email="u-aeo-onb-openai-exact@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Exact Target Biz",
        business_address="123 Main St, Salt Lake City, UT",
        industry="home services",
        website_url="https://example.com",
    )
    settings.AEO_TESTING_MODE = True
    settings.AEO_TEST_PROMPT_COUNT = 10

    call_types = []

    def _mock_openai(*args, **kwargs):
        call_types.append(kwargs.get("system_prompt", ""))
        n = int(kwargs.get("max_additional", 0))
        ptype = "transactional"
        sp = kwargs.get("system_prompt", "")
        if "type\" to \"trust" in sp or 'type" to "trust"' in sp:
            ptype = "trust"
        elif "type\" to \"comparison" in sp or 'type" to "comparison"' in sp:
            ptype = "comparison"
        elif "type\" to \"authority" in sp or 'type" to "authority"' in sp:
            ptype = "authority"
        return [
            {
                "prompt": f"{ptype} prompt {i}",
                "type": ptype,
                "weight": 1.0,
                "dynamic": True,
            }
            for i in range(n)
        ]

    monkeypatch.setattr("accounts.aeo.aeo_utils.run_prompt_batch_via_openai", _mock_openai)

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.get("/api/aeo/onboarding-prompt-plan/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["combined_target"] == 10
    assert len(body["combined"]) == 10
    assert body["meta"]["combined_shortfall"] == 0
    assert body["meta"]["openai_status"] == "ok"
    assert body["groups"]["fixed"] == []
    assert body["groups"]["dynamic"] == []
    assert body["groups"]["openai_generated"] == body["combined"]
    assert call_types
    allowed = {"transactional", "trust", "comparison", "authority"}
    assert all(item["type"] in allowed for item in body["combined"])


@pytest.mark.django_db
def test_aeo_onboarding_prompt_plan_openai_shortfall_meta(monkeypatch, settings):
    user = User.objects.create_user(
        username="u-aeo-onb-openai-shortfall",
        email="u-aeo-onb-openai-shortfall@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Shortfall Biz",
        business_address="123 Main St, Salt Lake City, UT",
        industry="dental",
        website_url="https://example.com",
    )
    settings.AEO_TESTING_MODE = True
    settings.AEO_TEST_PROMPT_COUNT = 10

    def _mock_openai(*args, **kwargs):
        return [
            {
                "prompt": "duplicate prompt",
                "type": "transactional",
                "weight": 1.0,
                "dynamic": True,
            }
        ]

    monkeypatch.setattr("accounts.aeo.aeo_utils.run_prompt_batch_via_openai", _mock_openai)

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.get("/api/aeo/onboarding-prompt-plan/")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["combined"]) == 1
    assert body["meta"]["combined_target"] == 10
    assert body["meta"]["combined_shortfall"] == 9
    assert body["meta"]["openai_status"] == "partial"
    assert body["meta"]["openai_prompt_count"] == 1
    assert body["groups"]["fixed"] == []
    assert body["groups"]["dynamic"] == []
