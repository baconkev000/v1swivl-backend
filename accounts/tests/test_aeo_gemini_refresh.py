import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import AEOExecutionRun, AEOResponseSnapshot, BusinessProfile
from accounts.tasks import run_aeo_gemini_refresh_task, run_aeo_phase1_execution_task, run_aeo_phase3_extraction_task

User = get_user_model()


@pytest.mark.django_db
def test_refresh_gemini_endpoint_enqueues_gemini_only_task(monkeypatch, settings):
    settings.GEMINI_API_KEY = "test-key"
    user = User.objects.create_user(username="rg1", email="rg1@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        selected_aeo_prompts=["prompt one", "prompt two", "prompt three"],
    )
    captured: dict = {}

    def fake_delay(run_id, prompt_set=None):
        captured["run_id"] = run_id
        captured["prompt_set"] = prompt_set or []

    monkeypatch.setattr("accounts.tasks.run_aeo_gemini_refresh_task.delay", fake_delay)
    monkeypatch.setattr("accounts.views.transaction.on_commit", lambda fn: fn())

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/aeo/refresh-gemini/", {}, format="json")
    assert res.status_code == 202
    body = res.json()
    assert body["provider"] == "gemini"
    assert body["status"] == "pending"
    assert body["prompt_count_requested"] >= 1
    assert isinstance(body["run_id"], int)
    assert captured["run_id"] == body["run_id"]
    prompts = captured["prompt_set"]
    assert len(prompts) >= 1
    assert all(isinstance(p.get("prompt"), str) and p.get("prompt").strip() for p in prompts)
    assert BusinessProfile.objects.get(id=profile.id).selected_aeo_prompts == [
        "prompt one",
        "prompt two",
        "prompt three",
    ]


@pytest.mark.django_db
def test_refresh_gemini_endpoint_prevents_duplicate_inflight(monkeypatch, settings):
    settings.GEMINI_API_KEY = "test-key"
    user = User.objects.create_user(username="rg2", email="rg2@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        selected_aeo_prompts=["prompt one"],
    )
    AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_PENDING,
        error_message="refresh_provider=gemini",
    )
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/aeo/refresh-gemini/", {}, format="json")
    assert res.status_code == 409


@pytest.mark.django_db
def test_run_aeo_gemini_refresh_task_uses_selected_prompts_and_gemini_provider(monkeypatch):
    user = User.objects.create_user(username="rg3", email="rg3@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        selected_aeo_prompts=["one", "two"],
    )
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)
    captured: dict = {}

    def fake_phase1(run_id, prompt_set=None, providers=None, force_refresh=False):
        captured["run_id"] = run_id
        captured["prompt_set"] = prompt_set or []
        captured["providers"] = providers
        captured["force_refresh"] = force_refresh

    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task", fake_phase1)
    run_aeo_gemini_refresh_task(run.id)
    assert captured["run_id"] == run.id
    assert captured["providers"] == ["gemini"]
    assert captured["force_refresh"] is True
    assert [p["prompt"] for p in captured["prompt_set"]] == ["one", "two"]


@pytest.mark.django_db
def test_gemini_phase1_and_phase3_extract_only_new_gemini_rows(monkeypatch):
    user = User.objects.create_user(username="rg4", email="rg4@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)
    existing_openai = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="openai old",
        prompt_hash="h_old",
        raw_response="old",
        platform="openai",
    )
    created_ids: list[int] = []

    def fake_batch(prompt_set, business_profile, **kwargs):
        assert kwargs.get("providers") == ["gemini"]
        # Guardrail: OpenAI path must not be used for this refresh.
        assert kwargs.get("providers") != ["openai"]
        for i, p in enumerate(prompt_set):
            row = AEOResponseSnapshot.objects.create(
                profile=business_profile,
                execution_run=run,
                prompt_text=str(p.get("prompt") or ""),
                prompt_hash=f"h_g_{i}",
                raw_response=f"gemini {i}",
                platform="gemini",
            )
            created_ids.append(row.id)
        return {
            "executed": len(created_ids),
            "failed": 0,
            "results": [
                {"success": True, "snapshot_id": sid, "platform": "gemini"}
                for sid in created_ids
            ],
        }

    extraction_calls: list[int] = []

    def fake_extraction(snapshot, save=True, competitor_hints=None):
        extraction_calls.append(snapshot.id)
        return {"extraction_snapshot_id": snapshot.id, "save_error": None}

    queued_phase3: list[tuple[int, list[int], str | None]] = []
    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch", fake_batch)
    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase3_extraction_task.delay",
        lambda run_id, snapshot_ids=None, response_platform=None: queued_phase3.append(
            (run_id, snapshot_ids or [], response_platform)
        ),
    )
    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_extraction)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda run_id: None)
    monkeypatch.setattr("accounts.tasks._enqueue_seo_after_aeo", lambda run_id: None)

    prompts = [{"prompt": "g1"}, {"prompt": "g2"}]
    run_aeo_phase1_execution_task(run.id, prompt_set=prompts, providers=["gemini"], force_refresh=True)
    assert queued_phase3 == [(run.id, created_ids, "gemini")]
    assert AEOResponseSnapshot.objects.filter(profile=profile, platform="gemini", id__in=created_ids).count() == 2

    run_aeo_phase3_extraction_task(run.id, created_ids, "gemini")
    assert set(extraction_calls) == set(created_ids)
    assert existing_openai.id not in extraction_calls

