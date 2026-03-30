import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from django.contrib.auth import get_user_model

from accounts.aeo.aeo_execution_utils import PLATFORM_GEMINI, PLATFORM_OPENAI, run_aeo_prompt_batch
from accounts.aeo.aeo_scoring_utils import calculate_aeo_scores_for_business, latest_extraction_per_response
from accounts.aeo.gemini_execution_utils import run_single_aeo_prompt_gemini
from accounts.models import (
    AEOExecutionRun,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    BusinessProfile,
)

User = get_user_model()


@pytest.mark.django_db
def test_dual_provider_batch_returns_two_results(settings, monkeypatch):
    settings.GEMINI_API_KEY = "test-key"
    user = User.objects.create_user(username="d1", email="d1@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)
    pair = uuid.uuid4()

    o = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="h",
        raw_response="ro",
        platform=PLATFORM_OPENAI,
        execution_run=run,
        execution_pair_id=pair,
    )
    g = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="h",
        raw_response="rg",
        platform=PLATFORM_GEMINI,
        execution_run=run,
        execution_pair_id=pair,
    )

    def openai_ok(prompt_obj, business_profile, **kwargs):
        return {
            "success": True,
            "snapshot_id": o.id,
            "platform": PLATFORM_OPENAI,
            "error": None,
            "prompt": "q",
            "prompt_hash": "h",
        }

    def gemini_ok(prompt_obj, business_profile, **kwargs):
        return {
            "success": True,
            "snapshot_id": g.id,
            "platform": PLATFORM_GEMINI,
            "error": None,
            "prompt": "q",
            "prompt_hash": "h",
        }

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_single_aeo_prompt", openai_ok)
    monkeypatch.setattr("accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini", gemini_ok)

    batch = run_aeo_prompt_batch([{"prompt": "q"}], profile, save=True, execution_run=run)
    assert batch["executed"] == 2
    assert batch["failed"] == 0
    assert len(batch["results"]) == 2
    assert {batch["results"][0]["snapshot_id"], batch["results"][1]["snapshot_id"]} == {o.id, g.id}


@pytest.mark.django_db
def test_dual_provider_uses_thread_pool(settings, monkeypatch):
    settings.GEMINI_API_KEY = "test-key"
    user = User.objects.create_user(username="d1b", email="d1b@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    original_init = ThreadPoolExecutor.__init__

    def tracked_init(self, *args, **kwargs):
        assert kwargs.get("max_workers") == 2
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", tracked_init)

    monkeypatch.setattr(
        "accounts.aeo.aeo_execution_utils.run_single_aeo_prompt",
        lambda *a, **k: {
            "success": True,
            "snapshot_id": 1,
            "platform": PLATFORM_OPENAI,
            "error": None,
            "prompt": "q",
            "prompt_hash": "h",
        },
    )
    monkeypatch.setattr(
        "accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini",
        lambda *a, **k: {
            "success": True,
            "snapshot_id": 2,
            "platform": PLATFORM_GEMINI,
            "error": None,
            "prompt": "q",
            "prompt_hash": "h",
        },
    )

    run_aeo_prompt_batch([{"prompt": "q"}], profile, save=False, execution_run=run)


@pytest.mark.django_db
def test_gemini_failure_openai_row_saved(settings, monkeypatch):
    settings.GEMINI_API_KEY = "test-key"
    user = User.objects.create_user(username="d2", email="d2@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    o = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="hx",
        raw_response="ro",
        platform=PLATFORM_OPENAI,
        execution_run=run,
    )

    def openai_ok(prompt_obj, business_profile, **kwargs):
        return {
            "success": True,
            "snapshot_id": o.id,
            "platform": PLATFORM_OPENAI,
            "error": None,
            "prompt": "q",
            "prompt_hash": "hx",
        }

    def gemini_fail(prompt_obj, business_profile, **kwargs):
        return {
            "success": False,
            "snapshot_id": None,
            "platform": PLATFORM_GEMINI,
            "error": "APIError: boom",
            "prompt": "q",
            "prompt_hash": "hx",
        }

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_single_aeo_prompt", openai_ok)
    monkeypatch.setattr("accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini", gemini_fail)

    batch = run_aeo_prompt_batch([{"prompt": "q"}], profile, save=True, execution_run=run)
    assert batch["executed"] == 1
    assert batch["failed"] == 1
    assert {r["snapshot_id"] for r in batch["results"] if r["success"]} == {o.id}


@pytest.mark.django_db
def test_missing_gemini_key_skips_gemini_path(settings, monkeypatch):
    settings.GEMINI_API_KEY = ""
    monkeypatch.delenv("GOOGLE_GEMINI_API_KEY", raising=False)

    class _DummyClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "accounts.aeo.aeo_execution_utils._execution_openai_client",
        lambda *_a, **_k: _DummyClient(),
    )

    user = User.objects.create_user(username="d3", email="d3@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    o = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="hy",
        raw_response="ro",
        platform=PLATFORM_OPENAI,
        execution_run=run,
    )

    def openai_ok(prompt_obj, business_profile, **kwargs):
        return {
            "success": True,
            "snapshot_id": o.id,
            "platform": PLATFORM_OPENAI,
            "error": None,
            "prompt": "q",
            "prompt_hash": "hy",
        }

    gemini_called = {"n": 0}

    def gemini_should_not_run(*args, **kwargs):
        gemini_called["n"] += 1
        raise AssertionError("gemini should not run without API key")

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_single_aeo_prompt", openai_ok)
    monkeypatch.setattr("accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini", gemini_should_not_run)

    batch = run_aeo_prompt_batch([{"prompt": "q"}], profile, save=True, execution_run=run)
    assert gemini_called["n"] == 0
    assert batch["executed"] == 1
    assert len(batch["results"]) == 1


@pytest.mark.django_db
def test_run_single_aeo_prompt_gemini_skipped_without_key(settings, monkeypatch):
    settings.GEMINI_API_KEY = ""
    monkeypatch.delenv("GOOGLE_GEMINI_API_KEY", raising=False)
    user = User.objects.create_user(username="d4", email="d4@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")

    out = run_single_aeo_prompt_gemini({"prompt": "hello"}, profile, save=False)
    assert out["success"] is False
    assert out["error"] == "skipped_no_api_key"
    assert out["platform"] == PLATFORM_GEMINI


@pytest.mark.django_db
def test_scoring_counts_openai_only_when_both_platforms_exist():
    user = User.objects.create_user(username="d5", email="d5@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Acme",
        website_url="https://acme.com",
    )
    h = "samehash"
    ro = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash=h,
        raw_response="Acme is great",
        platform=PLATFORM_OPENAI,
    )
    rg = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash=h,
        raw_response="Acme is great",
        platform=PLATFORM_GEMINI,
    )
    AEOExtractionSnapshot.objects.create(
        response_snapshot=ro,
        brand_mentioned=True,
        mention_position="top",
        mention_count=1,
        competitors_json=[],
        citations_json=[],
        sentiment="neutral",
        confidence_score=0.9,
        extraction_model="x",
        extraction_parse_failed=False,
    )
    AEOExtractionSnapshot.objects.create(
        response_snapshot=rg,
        brand_mentioned=False,
        mention_position="none",
        mention_count=0,
        competitors_json=[],
        citations_json=[],
        sentiment="neutral",
        confidence_score=0.9,
        extraction_model="x",
        extraction_parse_failed=False,
    )

    openai_only = latest_extraction_per_response(profile, response_platform=PLATFORM_OPENAI)
    assert len(openai_only) == 1
    assert openai_only[0].response_snapshot_id == ro.id

    all_plat = latest_extraction_per_response(profile, response_platform=None)
    assert len(all_plat) == 2

    scores = calculate_aeo_scores_for_business(profile, save=False)
    assert int(scores["total_prompts"]) == 1
    assert scores["visibility_score"] == 100.0


@pytest.mark.django_db
def test_generate_gemini_execution_text_mocked(monkeypatch):
    monkeypatch.setattr(
        "accounts.gemini_utils.get_effective_gemini_api_key",
        lambda: "k",
    )

    class FakeResponse:
        text = "  hello  "

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("google.generativeai.configure", lambda **kw: None)
    monkeypatch.setattr("google.generativeai.GenerativeModel", FakeModel)

    from accounts.gemini_utils import generate_gemini_execution_text

    raw, err = generate_gemini_execution_text(
        system_instruction="sys",
        user_text="user",
        temperature=0.2,
        max_output_tokens=100,
    )
    assert err is None
    assert raw == "hello"
