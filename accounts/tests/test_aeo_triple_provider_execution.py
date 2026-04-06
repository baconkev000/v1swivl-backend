"""
Triple-provider (OpenAI + Gemini + Perplexity) AEO Phase 1 batch behavior.
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from django.contrib.auth import get_user_model

from accounts.aeo.aeo_execution_utils import (
    PLATFORM_GEMINI,
    PLATFORM_OPENAI,
    PLATFORM_PERPLEXITY,
    _execution_max_workers,
    run_aeo_prompt_batch,
)
from accounts.models import AEOExecutionRun, AEOResponseSnapshot, BusinessProfile
from accounts.tasks import run_aeo_phase3_extraction_task

User = get_user_model()


@pytest.mark.django_db
def test_triple_provider_batch_when_all_keys_set(settings, monkeypatch):
    settings.GEMINI_API_KEY = "g-key"
    settings.PERPLEXITY_API_KEY = "p-key"
    user = User.objects.create_user(username="t1", email="t1@example.com", password="pw")
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
    p = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="h",
        raw_response="rp",
        platform=PLATFORM_PERPLEXITY,
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

    def perplexity_ok(prompt_obj, business_profile, **kwargs):
        return {
            "success": True,
            "snapshot_id": p.id,
            "platform": PLATFORM_PERPLEXITY,
            "error": None,
            "prompt": "q",
            "prompt_hash": "h",
        }

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_single_aeo_prompt", openai_ok)
    monkeypatch.setattr("accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini", gemini_ok)
    monkeypatch.setattr(
        "accounts.aeo.perplexity_execution_utils.run_single_aeo_prompt_perplexity",
        perplexity_ok,
    )

    batch = run_aeo_prompt_batch([{"prompt": "q"}], profile, save=True, execution_run=run)
    assert batch["executed"] == 3
    assert batch["failed"] == 0
    assert len(batch["results"]) == 3
    assert {batch["results"][i]["snapshot_id"] for i in range(3)} == {o.id, g.id, p.id}


@pytest.mark.django_db
def test_triple_batch_ten_prompts_max_workers(settings, monkeypatch):
    """10 prompts × 3 providers should use one pool (default 20 workers)."""
    settings.AEO_EXECUTION_MAX_WORKERS = 20
    settings.GEMINI_API_KEY = "g"
    settings.PERPLEXITY_API_KEY = "p"
    user = User.objects.create_user(username="t1b", email="t1b@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    original_init = ThreadPoolExecutor.__init__

    def tracked_init(self, *args, **kwargs):
        assert kwargs.get("max_workers") == _execution_max_workers()
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", tracked_init)

    def stub(**kwargs):
        return {
            "success": True,
            "snapshot_id": 1,
            "platform": PLATFORM_OPENAI,
            "error": None,
            "prompt": "q",
            "prompt_hash": "h",
        }

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_single_aeo_prompt", stub)
    monkeypatch.setattr("accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini", stub)
    monkeypatch.setattr("accounts.aeo.perplexity_execution_utils.run_single_aeo_prompt_perplexity", stub)

    prompts = [{"prompt": f"q{i}"} for i in range(10)]
    run_aeo_prompt_batch(prompts, profile, save=False, execution_run=run)


@pytest.mark.django_db
def test_perplexity_only_filter_runs_one_provider(settings, monkeypatch):
    settings.PERPLEXITY_API_KEY = "p-key"
    user = User.objects.create_user(username="t1c", email="t1c@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    called = {"openai": 0, "gemini": 0, "perplexity": 0}

    def openai_fn(*a, **k):
        called["openai"] += 1
        return {"success": True, "snapshot_id": 1, "platform": PLATFORM_OPENAI, "error": None, "prompt": "q", "prompt_hash": "h"}

    def gemini_fn(*a, **k):
        called["gemini"] += 1
        return {"success": True, "snapshot_id": 2, "platform": PLATFORM_GEMINI, "error": None, "prompt": "q", "prompt_hash": "h"}

    def perplexity_fn(*a, **k):
        called["perplexity"] += 1
        return {
            "success": True,
            "snapshot_id": 3,
            "platform": PLATFORM_PERPLEXITY,
            "error": None,
            "prompt": "q",
            "prompt_hash": "h",
        }

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_single_aeo_prompt", openai_fn)
    monkeypatch.setattr("accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini", gemini_fn)
    monkeypatch.setattr("accounts.aeo.perplexity_execution_utils.run_single_aeo_prompt_perplexity", perplexity_fn)

    run_aeo_prompt_batch([{"prompt": "q"}], profile, save=False, execution_run=run, providers=["perplexity"])
    assert called == {"openai": 0, "gemini": 0, "perplexity": 1}


@pytest.mark.django_db
def test_triple_provider_same_execution_pair_id_per_prompt(settings, monkeypatch):
    """One logical prompt → one UUID passed to OpenAI, Gemini, and Perplexity jobs."""
    settings.GEMINI_API_KEY = "g-key"
    settings.PERPLEXITY_API_KEY = "p-key"
    user = User.objects.create_user(username="t1d", email="t1d@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)
    seen_pairs: list[tuple[str, uuid.UUID | None]] = []

    def capture_pair(platform: str):
        def _fn(prompt_obj, business_profile, **kwargs):
            pid = kwargs.get("execution_pair_id")
            seen_pairs.append((platform, pid))
            return {
                "success": True,
                "snapshot_id": 1,
                "platform": platform,
                "error": None,
                "prompt": "q",
                "prompt_hash": "h",
            }

        return _fn

    monkeypatch.setattr(
        "accounts.aeo.aeo_execution_utils.run_single_aeo_prompt",
        capture_pair(PLATFORM_OPENAI),
    )
    monkeypatch.setattr(
        "accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini",
        capture_pair(PLATFORM_GEMINI),
    )
    monkeypatch.setattr(
        "accounts.aeo.perplexity_execution_utils.run_single_aeo_prompt_perplexity",
        capture_pair(PLATFORM_PERPLEXITY),
    )

    run_aeo_prompt_batch([{"prompt": "q"}], profile, save=False, execution_run=run)
    assert len(seen_pairs) == 3
    uuids = {p for _, p in seen_pairs}
    assert len(uuids) == 1
    assert uuids.pop() is not None


@pytest.mark.django_db
def test_triple_batch_ten_prompts_max_in_flight(settings, monkeypatch):
    """10 prompts × 3 providers with pool size 20 → at most 20 concurrent calls."""
    settings.AEO_EXECUTION_MAX_WORKERS = 20
    settings.GEMINI_API_KEY = "g"
    settings.PERPLEXITY_API_KEY = "p"
    user = User.objects.create_user(username="t1e", email="t1e@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    lock = threading.Lock()
    in_flight = [0]
    max_in_flight = [0]

    def track_enter():
        with lock:
            in_flight[0] += 1
            max_in_flight[0] = max(max_in_flight[0], in_flight[0])

    def track_exit():
        with lock:
            in_flight[0] -= 1

    def stub(prompt_obj, business_profile, **kwargs):
        track_enter()
        try:
            time.sleep(0.2)
            return {
                "success": True,
                "snapshot_id": 1,
                "platform": PLATFORM_OPENAI,
                "error": None,
                "prompt": prompt_obj.get("prompt", ""),
                "prompt_hash": "h",
            }
        finally:
            track_exit()

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_single_aeo_prompt", stub)
    monkeypatch.setattr("accounts.aeo.gemini_execution_utils.run_single_aeo_prompt_gemini", stub)
    monkeypatch.setattr("accounts.aeo.perplexity_execution_utils.run_single_aeo_prompt_perplexity", stub)

    prompts = [{"prompt": f"q{i}"} for i in range(10)]
    run_aeo_prompt_batch(prompts, profile, save=False, execution_run=run)
    assert max_in_flight[0] == 20


@pytest.mark.django_db
def test_phase3_extracts_perplexity_snapshot_like_others(monkeypatch):
    user = User.objects.create_user(username="t1f", email="t1f@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    o = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="h_o",
        raw_response="openai text",
        platform=PLATFORM_OPENAI,
        execution_run=run,
    )
    g = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="h_g",
        raw_response="gemini text",
        platform=PLATFORM_GEMINI,
        execution_run=run,
    )
    p = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="h_p",
        raw_response="perplexity text",
        platform=PLATFORM_PERPLEXITY,
        execution_run=run,
    )

    extraction_calls: list[int] = []

    def fake_extraction(snapshot, save=True, competitor_hints=None):
        extraction_calls.append(snapshot.id)
        return {"extraction_snapshot_id": snapshot.id, "save_error": None}

    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_extraction)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda run_id: None)

    run_aeo_phase3_extraction_task(run.id, [o.id, g.id, p.id], None)
    assert set(extraction_calls) == {o.id, g.id, p.id}


@pytest.mark.django_db
def test_phase3_perplexity_platform_filter_scopes_to_perplexity_rows_only(monkeypatch):
    user = User.objects.create_user(username="t1g", email="t1g@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    o = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="h_o",
        raw_response="openai",
        platform=PLATFORM_OPENAI,
        execution_run=run,
    )
    p = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q",
        prompt_hash="h_p",
        raw_response="perplexity",
        platform=PLATFORM_PERPLEXITY,
        execution_run=run,
    )

    extraction_calls: list[int] = []

    def fake_extraction(snapshot, save=True, competitor_hints=None):
        extraction_calls.append(snapshot.id)
        return {"extraction_snapshot_id": snapshot.id, "save_error": None}

    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_extraction)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda run_id: None)

    run_aeo_phase3_extraction_task(run.id, [o.id, p.id], PLATFORM_PERPLEXITY)
    assert extraction_calls == [p.id]


@pytest.mark.django_db
def test_run_single_aeo_prompt_perplexity_parses_chat_completions_response(settings, monkeypatch):
    """Perplexity OpenAI-shaped JSON → raw_response and optional response model name."""
    settings.PERPLEXITY_API_KEY = "pk-test"
    settings.PERPLEXITY_AEO_MODEL = "sonar"
    user = User.objects.create_user(username="t1h", email="t1h@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")

    class FakeResp:
        status_code = 200
        content = b"{}"
        text = ""

        def json(self):
            return {
                "model": "sonar-pro",
                "choices": [{"message": {"content": "  Answer text  "}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    monkeypatch.setattr("accounts.aeo.perplexity_execution_utils.requests.post", lambda **kw: FakeResp())
    monkeypatch.setattr("accounts.aeo.perplexity_execution_utils.record_perplexity_request", lambda **kw: None)

    from accounts.aeo.perplexity_execution_utils import run_single_aeo_prompt_perplexity

    out = run_single_aeo_prompt_perplexity({"prompt": "What is AEO?"}, profile, save=False)
    assert out["success"] is True
    assert out["platform"] == PLATFORM_PERPLEXITY
    assert out["raw_response"] == "Answer text"
    assert out["model_name"] == "sonar-pro"
    assert out["error"] is None
