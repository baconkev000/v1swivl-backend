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


@pytest.mark.django_db
def test_aeo_onboarding_step2_reuses_db_prompts_same_domain_no_openai(monkeypatch, settings):
    """Step-2 POST must return saved prompts when domain matches; no build_full_aeo_prompt_plan."""
    settings.AEO_TESTING_MODE = True
    user = User.objects.create_user(
        username="u-aeo-step2-cache",
        email="u-aeo-step2-cache@example.com",
        password="pw",
    )
    saved_prompts = [f"cached prompt {i}" for i in range(10)]
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Cached Co",
        business_address="US",
        website_url="https://example.com",
        selected_aeo_prompts=saved_prompts,
    )

    def boom(*args, **kwargs):
        raise AssertionError("build_full_aeo_prompt_plan must not be called when DB prompts match domain")

    monkeypatch.setattr("accounts.views.build_full_aeo_prompt_plan", boom)

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/aeo/onboarding-prompt-plan/",
        {
            "onboarding_step2_prompt_plan": True,
            "selected_topics": ["Alpha", "Beta"],
            "onboarding_context": {
                "business_name": "Cached Co",
                "website_url": "https://www.example.com/about",
                "location": "US",
                "language": "English",
            },
        },
        format="json",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["openai_status"] == "reused_saved"
    pbt = body["prompts_by_topic"]
    assert set(pbt.keys()) == {"Alpha", "Beta"}
    assert len(pbt["Alpha"]) + len(pbt["Beta"]) == 10
    assert all(len(pbt[k]) > 0 for k in pbt)


@pytest.mark.django_db
def test_aeo_onboarding_step2_reuses_sibling_profile_prompts_same_domain_no_openai(monkeypatch, settings):
    settings.AEO_TESTING_MODE = True
    user = User.objects.create_user(
        username="u-aeo-step2-sibling-cache",
        email="u-aeo-step2-sibling-cache@example.com",
        password="pw",
    )
    source_saved = [f"sibling cached prompt {i}" for i in range(10)]
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Source Co",
        website_url="https://example.com",
        selected_aeo_prompts=source_saved,
    )
    target = BusinessProfile.objects.create(
        user=user,
        is_main=False,
        business_name="Target Co",
        website_url="https://example.com",
        selected_aeo_prompts=[],
    )

    def boom(*args, **kwargs):
        raise AssertionError("build_full_aeo_prompt_plan must not be called when sibling cached prompts are reusable")

    monkeypatch.setattr("accounts.views.build_full_aeo_prompt_plan", boom)

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/aeo/onboarding-prompt-plan/",
        {
            "profile_id": target.id,
            "onboarding_step2_prompt_plan": True,
            "selected_topics": ["Alpha", "Beta"],
            "onboarding_context": {
                "business_name": "Target Co",
                "website_url": "https://www.example.com/services",
                "location": "US",
                "language": "English",
            },
        },
        format="json",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["openai_status"] == "reused_saved"
    pbt = body["prompts_by_topic"]
    assert set(pbt.keys()) == {"Alpha", "Beta"}
    assert len(pbt["Alpha"]) + len(pbt["Beta"]) == 10


@pytest.mark.django_db
def test_aeo_onboarding_step2_supersedes_inflight_and_enqueues_fresh_run(monkeypatch, settings):
    settings.AEO_TESTING_MODE = True
    settings.AEO_TEST_PROMPT_COUNT = 10
    user = User.objects.create_user(
        username="u-aeo-step2-supersede",
        email="u-aeo-step2-supersede@example.com",
        password="pw",
    )
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Switch Co",
        business_address="US",
        website_url="https://old-site.example",
    )
    stale = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_RUNNING,
        extraction_status=AEOExecutionRun.STAGE_RUNNING,
        scoring_status=AEOExecutionRun.STAGE_PENDING,
        recommendation_status=AEOExecutionRun.STAGE_PENDING,
        background_status=AEOExecutionRun.STAGE_PENDING,
    )

    callbacks = []
    delayed_calls = []

    monkeypatch.setattr("accounts.views.transaction.on_commit", lambda cb: callbacks.append(cb))
    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase1_execution_task.delay",
        lambda run_id, prompt_payload: delayed_calls.append((run_id, prompt_payload)),
    )
    monkeypatch.setattr(
        "accounts.views.build_full_aeo_prompt_plan",
        lambda *args, **kwargs: {
            "combined": [
                {"prompt": f"fresh prompt {i}", "type": "transactional", "weight": 1.0, "dynamic": True}
                for i in range(10)
            ],
            "prompts_by_topic": {"Alpha": [f"fresh prompt {i}" for i in range(5)], "Beta": [f"fresh prompt {i}" for i in range(5, 10)]},
            "meta": {"openai_status": "ok", "combined_target": 10, "combined_shortfall": 0},
            "business": {},
        },
    )

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/aeo/onboarding-prompt-plan/",
        {
            "onboarding_step2_prompt_plan": True,
            "selected_topics": ["Alpha", "Beta"],
            "onboarding_context": {
                "business_name": "Switch Co",
                "website_url": "https://new-site.example",
                "location": "US",
                "language": "English",
            },
        },
        format="json",
    )

    assert resp.status_code == 200
    stale.refresh_from_db()
    assert stale.status == AEOExecutionRun.STATUS_FAILED
    assert stale.error_message == "superseded_onboarding_step2_refresh"
    assert len(callbacks) == 1
    assert delayed_calls == []

    callbacks[0]()
    assert len(delayed_calls) == 1
    run_id, payload = delayed_calls[0]
    assert isinstance(run_id, int)
    assert run_id != stale.id
    assert len(payload) == 10
