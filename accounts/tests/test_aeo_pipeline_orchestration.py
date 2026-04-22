from concurrent.futures import Future
from typing import Any

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.models import (
    AEOExecutionRun,
    AEOExtractionSnapshot,
    AEOPromptExecutionAggregate,
    AEOResponseSnapshot,
    AEORecommendationRun,
    AEOScoreSnapshot,
    BusinessProfile,
)
from accounts.aeo.worker_limits import aeo_execution_max_workers
from accounts.tasks import (
    run_aeo_phase1_execution_task,
    run_aeo_phase2_confidence_task,
    run_aeo_phase3_extraction_task,
    run_aeo_phase4_scoring_task,
    run_aeo_phase5_recommendation_task,
    trigger_seo_warmup_after_aeo_task,
)
from accounts.aeo.aeo_recommendation_utils import generate_aeo_recommendations
from accounts.aeo.aeo_scoring_utils import calculate_aeo_scores_for_business


User = get_user_model()


class _ImmediateThreadPoolExecutor:
    """Runs work inline so pytest monkeypatches on ``run_single_extraction`` apply (worker threads can miss them)."""

    def __init__(self, max_workers: int = 1) -> None:
        self._max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:
            fut.set_exception(exc)
        return fut


@pytest.fixture(autouse=True)
def _stub_aeo_celery_delays(monkeypatch):
    """Tests run without Redis/broker; individual tests override .delay when asserting enqueue order."""
    noop = lambda *args, **kwargs: None
    monkeypatch.setattr("accounts.tasks.refresh_competitor_snapshot_for_profile_task.delay", noop)
    monkeypatch.setattr("accounts.tasks.refresh_aeo_dashboard_bundle_cache_task.delay", noop)
    monkeypatch.setattr("accounts.tasks.aeo_repair_stalled_visibility_pipeline_task.delay", noop)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase2_confidence_task.delay", noop)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase3_extraction_task.delay", noop)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", noop)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase5_recommendation_task.delay", noop)
    monkeypatch.setattr("accounts.tasks.trigger_seo_warmup_after_aeo_task.delay", noop)


@pytest.mark.django_db
def test_phase1_completion_enqueues_extraction_on_partial_failures(monkeypatch, settings):
    settings.AEO_TESTING_MODE = False
    user = User.objects.create_user(username="p1", email="p1@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        plan=BusinessProfile.PLAN_ADVANCED,
    )
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    snap = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q1",
        prompt_hash="h1",
        raw_response="r1",
    )

    def fake_batch(*args, **kwargs):
        one_ok = {"success": True, "snapshot_id": snap.id}
        one_fail = {"success": False, "snapshot_id": None}
        cb = kwargs.get("on_result")
        if cb is not None:
            cb(one_ok, {"prompt_obj": {"prompt": "q1", "type": "transactional", "dynamic": True, "weight": 1.0}})
            cb(one_fail, {"prompt_obj": {"prompt": "q1"}})
        return {
            "executed": 1,
            "failed": 1,
            "results": [one_ok, one_fail],
        }

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch", fake_batch)
    row = AEOExtractionSnapshot.objects.create(
        response_snapshot=snap,
        brand_mentioned=True,
        mention_position="top",
        mention_count=1,
        competitors_json=[],
        citations_json=[],
        sentiment="neutral",
        confidence_score=0.9,
        extraction_model="fake",
        extraction_parse_failed=False,
    )
    monkeypatch.setattr(
        "accounts.aeo.aeo_extraction_utils.run_single_extraction",
        lambda snapshot, save=True, competitor_hints=None: {"extraction_snapshot_id": row.id},
    )

    queued = []
    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase3_extraction_task.delay",
        lambda run_id, snapshot_ids=None, response_platform=None: queued.append(
            (run_id, snapshot_ids or [], response_platform)
        ),
    )
    monkeypatch.setattr("accounts.tasks.run_aeo_phase2_confidence_task.delay", lambda *args, **kwargs: None)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda *args, **kwargs: None)

    run_aeo_phase1_execution_task(run.id, [{"prompt": "q1"}])
    run.refresh_from_db()
    assert run.status == AEOExecutionRun.STATUS_COMPLETED
    assert run.prompt_count_executed == 1
    assert run.prompt_count_failed == 1
    assert queued == [(run.id, [snap.id], None)]


@pytest.mark.django_db
def test_phase1_missing_snapshot_id_marks_run_failed(monkeypatch):
    user = User.objects.create_user(username="p1m", email="p1m@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    monkeypatch.setattr(
        "accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch",
        lambda *args, **kwargs: {
            "executed": 1,
            "failed": 0,
            "results": [{"success": True, "snapshot_id": None}],
        },
    )
    queued = []
    monkeypatch.setattr("accounts.tasks.trigger_seo_warmup_after_aeo_task.delay", lambda rid: queued.append(rid))
    run_aeo_phase1_execution_task(run.id, [{"prompt": "q1"}])
    run.refresh_from_db()
    assert run.status == AEOExecutionRun.STATUS_FAILED
    assert run.extraction_status == AEOExecutionRun.STAGE_FAILED
    assert "artifact_mismatch" in (run.error_message or "")
    assert queued == [run.id]


@pytest.mark.django_db
def test_phase1_extraction_failure_skips_aggregate_and_fails_readiness(monkeypatch):
    user = User.objects.create_user(username="p1e", email="p1e@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)
    snap = AEOResponseSnapshot.objects.create(
        profile=profile,
        execution_run=run,
        prompt_text="q1",
        prompt_hash="h1e",
        raw_response="r1",
        platform="openai",
    )
    monkeypatch.setattr(
        "accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch",
        lambda *args, **kwargs: {"executed": 1, "failed": 0, "results": [{"success": True, "snapshot_id": snap.id}]},
    )
    monkeypatch.setattr(
        "accounts.aeo.aeo_extraction_utils.run_single_extraction",
        lambda snapshot, save=True, competitor_hints=None: {"extraction_snapshot_id": None},
    )
    monkeypatch.setattr("accounts.tasks.trigger_seo_warmup_after_aeo_task.delay", lambda *args, **kwargs: None)
    run_aeo_phase1_execution_task(run.id, [{"prompt": "q1"}])
    run.refresh_from_db()
    assert run.status == AEOExecutionRun.STATUS_FAILED
    assert AEOPromptExecutionAggregate.objects.filter(profile=profile, execution_run=run).count() == 0


@pytest.mark.django_db
def test_phase3_creates_extraction_rows_and_enqueues_scoring(monkeypatch):
    user = User.objects.create_user(username="p3", email="p3@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        started_at=timezone.now(),
    )
    r1 = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="a", prompt_hash="ha", raw_response="ra")
    r2 = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="b", prompt_hash="hb", raw_response="rb")

    def fake_single_extraction(snapshot, save=True, competitor_hints=None):
        row = AEOExtractionSnapshot.objects.create(
            response_snapshot=snapshot,
            brand_mentioned=True,
            mention_position="top",
            mention_count=1,
            competitors_json=[],
            citations_json=["example.com"],
            sentiment="positive",
            confidence_score=0.9,
            extraction_model="fake",
            extraction_parse_failed=False,
        )
        return {"extraction_snapshot_id": row.id, "save_error": None}

    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_single_extraction)
    monkeypatch.setattr("django.db.close_old_connections", lambda: None)
    monkeypatch.setattr("accounts.tasks.ThreadPoolExecutor", _ImmediateThreadPoolExecutor)
    queued = []
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda run_id: queued.append(run_id))

    run_aeo_phase3_extraction_task(run.id, [r1.id, r2.id])
    run.refresh_from_db()
    assert run.extraction_status == AEOExecutionRun.STAGE_COMPLETED
    assert run.extraction_count == 2
    assert AEOExtractionSnapshot.objects.filter(response_snapshot__profile=profile).count() == 2
    assert queued == [run.id]


@pytest.mark.django_db
def test_phase3_thread_pool_uses_aeo_execution_max_workers(monkeypatch, settings):
    settings.AEO_EXECUTION_MAX_WORKERS = 11
    user = User.objects.create_user(username="p3w", email="p3w@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        started_at=timezone.now(),
    )
    r1 = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="a", prompt_hash="ha", raw_response="ra")

    seen_workers: list[int] = []

    class _RecordingImmediatePool(_ImmediateThreadPoolExecutor):
        def __init__(self, max_workers: int = 1) -> None:
            seen_workers.append(int(max_workers))
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr("django.db.close_old_connections", lambda: None)
    monkeypatch.setattr("accounts.tasks.ThreadPoolExecutor", _RecordingImmediatePool)

    def fake_single_extraction(snapshot, save=True, competitor_hints=None):
        row = AEOExtractionSnapshot.objects.create(
            response_snapshot=snapshot,
            brand_mentioned=True,
            mention_position="top",
            mention_count=1,
            competitors_json=[],
            citations_json=[],
            sentiment="neutral",
            confidence_score=0.5,
            extraction_model="fake",
            extraction_parse_failed=False,
        )
        return {"extraction_snapshot_id": row.id, "save_error": None}

    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_single_extraction)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda _rid: None)

    run_aeo_phase3_extraction_task(run.id, [r1.id])
    assert seen_workers == [aeo_execution_max_workers()]


@pytest.mark.django_db
def test_phase3_skips_idempotent_snapshots_but_extracts_others(monkeypatch):
    user = User.objects.create_user(username="p3i", email="p3i@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    t0 = timezone.now()
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        started_at=t0,
    )
    r1 = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="a", prompt_hash="ha", raw_response="ra")
    r2 = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="b", prompt_hash="hb", raw_response="rb")
    AEOExtractionSnapshot.objects.create(
        response_snapshot=r1,
        brand_mentioned=True,
        mention_position="top",
        mention_count=1,
        competitors_json=[],
        citations_json=[],
        sentiment="neutral",
        confidence_score=0.5,
        extraction_model="prev",
        extraction_parse_failed=False,
    )

    called_ids: list[int] = []

    def fake_single_extraction(snapshot, save=True, competitor_hints=None):
        called_ids.append(snapshot.id)
        row = AEOExtractionSnapshot.objects.create(
            response_snapshot=snapshot,
            brand_mentioned=True,
            mention_position="top",
            mention_count=1,
            competitors_json=[],
            citations_json=[],
            sentiment="neutral",
            confidence_score=0.5,
            extraction_model="fake",
            extraction_parse_failed=False,
        )
        return {"extraction_snapshot_id": row.id, "save_error": None}

    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_single_extraction)
    monkeypatch.setattr("django.db.close_old_connections", lambda: None)
    monkeypatch.setattr("accounts.tasks.ThreadPoolExecutor", _ImmediateThreadPoolExecutor)
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda _rid: None)

    run_aeo_phase3_extraction_task(run.id, [r1.id, r2.id])
    run.refresh_from_db()
    assert called_ids == [r2.id]
    assert run.extraction_count == 1


@pytest.mark.django_db
def test_phase4_creates_score_snapshot_with_expected_fields(monkeypatch, settings):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = False
    settings.AEO_TESTING_MODE = False
    user = User.objects.create_user(username="p4", email="p4@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        plan=BusinessProfile.PLAN_ADVANCED,
    )
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        extraction_status=AEOExecutionRun.STAGE_COMPLETED,
    )
    rsp = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="a", prompt_hash="ha", raw_response="ra")
    AEOExtractionSnapshot.objects.create(
        response_snapshot=rsp,
        brand_mentioned=True,
        mention_position="top",
        mention_count=2,
        competitors_json=["comp1"],
        citations_json=["source.com"],
        sentiment="positive",
        confidence_score=0.8,
        extraction_model="fake",
        extraction_parse_failed=False,
    )

    run_aeo_phase4_scoring_task(run.id)
    run.refresh_from_db()
    assert run.scoring_status == AEOExecutionRun.STAGE_COMPLETED
    assert run.score_snapshot_id is not None
    score = AEOScoreSnapshot.objects.get(id=run.score_snapshot_id)
    assert score.total_prompts >= 1
    assert score.total_mentions >= 1
    assert score.execution_run_id == run.id


@pytest.mark.django_db
def test_phase1_sample_profile_enqueues_phase2_only_not_phase4_from_phase1(monkeypatch, settings):
    """Onboarding-sized profiles: Phase 1 must not schedule Phase 4 (Phase 2 owns that enqueue)."""
    settings.AEO_TESTING_MODE = False
    user = User.objects.create_user(username="p1smp", email="p1smp@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        plan=BusinessProfile.PLAN_STARTER,
    )
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)
    prompt_item = {"prompt": "q1", "type": "transactional", "dynamic": True, "weight": 1.0}

    def fake_batch(selected, profile, save=True, execution_run=None, providers=None, on_result=None, **kwargs):
        results = []
        for item in selected:
            prov = (providers or ["openai"])[0]
            snap = AEOResponseSnapshot.objects.create(
                profile=profile,
                execution_run=execution_run,
                prompt_text=str(item.get("prompt") or ""),
                prompt_hash=f"h-{prov}",
                raw_response="r",
                platform=prov,
            )
            one = {"success": True, "snapshot_id": snap.id}
            results.append(one)
            if on_result is not None:
                cat_item = dict(item)
                from accounts.aeo.progressive_onboarding import classify_prompt_category

                cat_item["_aeo_category"] = classify_prompt_category(item)
                on_result(one, {"prompt_obj": cat_item})
        return {"executed": len(results), "failed": 0, "results": results}

    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch", fake_batch)

    def fake_single_extraction(snapshot, save=True, competitor_hints=None):
        row = AEOExtractionSnapshot.objects.create(
            response_snapshot=snapshot,
            brand_mentioned=True,
            mention_position="top",
            mention_count=1,
            competitors_json=[],
            citations_json=[],
            sentiment="neutral",
            confidence_score=0.9,
            extraction_model="fake",
            extraction_parse_failed=False,
        )
        return {"extraction_snapshot_id": row.id}

    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_single_extraction)

    phase2_calls: list[tuple] = []
    phase4_calls: list[int] = []
    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase2_confidence_task.delay",
        lambda rid, ps=None: phase2_calls.append((rid, ps)),
    )
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda rid: phase4_calls.append(rid))

    run_aeo_phase1_execution_task(run.id, [prompt_item])
    run.refresh_from_db()
    assert run.status == AEOExecutionRun.STATUS_COMPLETED
    assert len(phase2_calls) == 1
    assert phase2_calls[0][0] == run.id
    assert phase4_calls == []


@pytest.mark.django_db
def test_phase2_no_aggregates_enqueues_phase4_once(monkeypatch, settings):
    settings.AEO_TESTING_MODE = False
    user = User.objects.create_user(username="p2na", email="p2na@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    phase4_calls: list[int] = []
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda rid: phase4_calls.append(rid))
    monkeypatch.setattr("accounts.aeo.perplexity_execution_utils.perplexity_execution_enabled", lambda: False)

    run_aeo_phase2_confidence_task(
        run.id,
        [{"prompt": "orphan", "type": "transactional", "dynamic": True, "weight": 1.0}],
    )
    run.refresh_from_db()
    assert run.background_status == AEOExecutionRun.STAGE_SKIPPED
    assert phase4_calls == [run.id]


@pytest.mark.django_db
def test_calculate_aeo_scores_for_business_links_execution_run_when_saving(settings):
    settings.AEO_TESTING_MODE = False
    user = User.objects.create_user(username="p4fk", email="p4fk@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        website_url="https://example.com",
    )
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    rsp = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="a", prompt_hash="ha", raw_response="ra")
    AEOExtractionSnapshot.objects.create(
        response_snapshot=rsp,
        brand_mentioned=True,
        mention_position="top",
        mention_count=1,
        competitors_json=[],
        citations_json=["example.com"],
        sentiment="positive",
        confidence_score=0.8,
        extraction_model="fake",
        extraction_parse_failed=False,
    )
    calculate_aeo_scores_for_business(profile, save=True, execution_run_id=run.id)
    snap = AEOScoreSnapshot.objects.filter(profile=profile).order_by("-id").first()
    assert snap is not None
    assert snap.execution_run_id == run.id


@pytest.mark.django_db
def test_generate_aeo_recommendations_uses_execution_run_score_not_latest(monkeypatch, settings):
    settings.AEO_RECOMMENDATION_GROUP_GAPS = False
    user = User.objects.create_user(username="p5sc", email="p5sc@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz", website_url="https://x.com")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    older = AEOScoreSnapshot.objects.create(
        profile=profile,
        visibility_score=11.0,
        weighted_position_score=11.0,
        citation_share=11.0,
        competitor_dominance_json={},
        total_prompts=1,
        total_mentions=1,
    )
    AEOScoreSnapshot.objects.create(
        profile=profile,
        visibility_score=99.0,
        weighted_position_score=99.0,
        citation_share=99.0,
        competitor_dominance_json={},
        total_prompts=1,
        total_mentions=1,
    )
    run.score_snapshot_id = older.id
    run.save(update_fields=["score_snapshot_id", "updated_at"])

    monkeypatch.setattr(
        "accounts.aeo.aeo_recommendation_utils.latest_extraction_per_response",
        lambda _bp, **kwargs: [],
    )

    out = generate_aeo_recommendations(profile, save=False, execution_run_id=run.id)
    assert out["score_snapshot_id"] == older.id
    assert out["visibility_score"] == 11.0


@pytest.mark.django_db
def test_aeo_pipeline_status_endpoint_uses_cached_snapshots_only(monkeypatch):
    user = User.objects.create_user(username="pstat", email="pstat@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        scoring_status=AEOExecutionRun.STAGE_COMPLETED,
    )
    score = AEOScoreSnapshot.objects.create(
        profile=profile,
        visibility_score=50.0,
        weighted_position_score=25.0,
        citation_share=10.0,
        competitor_dominance_json={},
        total_prompts=3,
        total_mentions=5,
    )
    run.score_snapshot_id = score.id
    run.save(update_fields=["score_snapshot_id", "updated_at"])

    monkeypatch.setattr(
        "accounts.views.get_or_refresh_seo_score_for_user",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.get("/api/aeo/pipeline-status/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run"]["id"] == run.id
    assert body["score_snapshot"]["id"] == score.id
    assert body["score_snapshot"]["aeo_score"] == 28  # round((50 + 25 + 10) / 3)
    assert "freshness" in body


@pytest.mark.django_db
def test_aeo_completed_triggers_seo_once(monkeypatch, settings):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = False
    user = User.objects.create_user(username="seo1", email="seo1@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz", website_url="https://example.com")
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        extraction_status=AEOExecutionRun.STAGE_COMPLETED,
    )
    rsp = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="a", prompt_hash="h", raw_response="r")
    AEOExtractionSnapshot.objects.create(
        response_snapshot=rsp,
        brand_mentioned=True,
        mention_position="top",
        mention_count=1,
        competitors_json=[],
        citations_json=[],
        sentiment="neutral",
        confidence_score=0.8,
        extraction_model="m",
        extraction_parse_failed=False,
    )
    queued = []
    monkeypatch.setattr("accounts.tasks.trigger_seo_warmup_after_aeo_task.delay", lambda rid: queued.append(rid))
    run_aeo_phase4_scoring_task(run.id)
    assert queued == [run.id]


@pytest.mark.django_db
def test_aeo_skipped_cached_still_attempts_seo_trigger(monkeypatch):
    user = User.objects.create_user(username="seo2", email="seo2@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz", website_url="https://example.com")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)
    previous = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        fetch_mode=AEOExecutionRun.FETCH_MODE_FRESH_FETCH,
        finished_at=timezone.now(),
    )
    assert previous.id is not None
    queued = []
    monkeypatch.setattr("accounts.tasks.trigger_seo_warmup_after_aeo_task.delay", lambda rid: queued.append(rid))
    run_aeo_phase1_execution_task(run.id, [{"prompt": "a"}])
    run.refresh_from_db()
    assert run.status == AEOExecutionRun.STATUS_SKIPPED_CACHED
    assert queued == [run.id]


@pytest.mark.django_db
def test_aeo_phase1_failure_still_attempts_seo_trigger(monkeypatch):
    user = User.objects.create_user(username="seo3", email="seo3@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz", website_url="https://example.com")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)
    monkeypatch.setattr(
        "accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    queued = []
    monkeypatch.setattr("accounts.tasks.trigger_seo_warmup_after_aeo_task.delay", lambda rid: queued.append(rid))
    run_aeo_phase1_execution_task(run.id, [{"prompt": "a"}])
    run.refresh_from_db()
    assert run.status == AEOExecutionRun.STATUS_FAILED
    assert queued == [run.id]


@pytest.mark.django_db
def test_seo_trigger_task_skips_when_missing_website_url():
    user = User.objects.create_user(username="seo4", email="seo4@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz", website_url="")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    trigger_seo_warmup_after_aeo_task(run.id)
    run.refresh_from_db()
    assert run.seo_trigger_status == "skipped_no_website"
    assert run.seo_triggered_at is not None


@pytest.mark.django_db
def test_seo_trigger_task_is_idempotent_on_replay(monkeypatch):
    user = User.objects.create_user(username="seo5", email="seo5@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz", website_url="https://example.com")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    calls = {"count": 0}

    def fake_refresh(*args, **kwargs):
        calls["count"] += 1
        return {"seo_score": 42}

    monkeypatch.setattr("accounts.dataforseo_utils.get_or_refresh_seo_score_for_user", fake_refresh)
    monkeypatch.setattr(
        "accounts.seo_snapshot_refresh.sync_enrich_current_period_seo_snapshot_for_profile",
        lambda *a, **k: {"ok": True, "persisted": True},
    )
    trigger_seo_warmup_after_aeo_task(run.id)
    trigger_seo_warmup_after_aeo_task(run.id)
    run.refresh_from_db()
    assert calls["count"] == 1
    assert run.seo_trigger_status == "success"


@pytest.mark.django_db
def test_phase3_advanced_enqueues_phase2_not_direct_phase4(settings, monkeypatch):
    settings.AEO_TESTING_MODE = False
    user = User.objects.create_user(username="p3adv", email="p3adv@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        plan=BusinessProfile.PLAN_ADVANCED,
    )
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        started_at=timezone.now(),
    )
    AEOPromptExecutionAggregate.objects.create(
        profile=profile,
        execution_run=run,
        prompt_hash="ha",
        prompt_text="topic a",
        openai_pass_count=1,
        gemini_pass_count=1,
    )
    r1 = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="a", prompt_hash="ha", raw_response="ra")

    def fake_single_extraction(snapshot, save=True, competitor_hints=None):
        row = AEOExtractionSnapshot.objects.create(
            response_snapshot=snapshot,
            brand_mentioned=True,
            mention_position="top",
            mention_count=1,
            competitors_json=[],
            citations_json=["example.com"],
            sentiment="positive",
            confidence_score=0.9,
            extraction_model="fake",
            extraction_parse_failed=False,
        )
        return {"extraction_snapshot_id": row.id, "save_error": None}

    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_single_extraction)
    monkeypatch.setattr("django.db.close_old_connections", lambda: None)
    monkeypatch.setattr("accounts.tasks.ThreadPoolExecutor", _ImmediateThreadPoolExecutor)
    phase2_calls: list[tuple] = []
    phase4_calls: list[int] = []
    monkeypatch.setattr("accounts.tasks.refresh_competitor_snapshot_for_profile_task.delay", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase2_confidence_task.delay",
        lambda *args, **kwargs: phase2_calls.append((args, kwargs)),
    )
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda rid: phase4_calls.append(rid))

    run_aeo_phase3_extraction_task(run.id, [r1.id])
    assert len(phase2_calls) == 1
    assert phase2_calls[0][0][0] == run.id
    assert phase4_calls == []


@pytest.mark.django_db
def test_phase3_starter_enqueues_phase4_not_phase2(settings, monkeypatch):
    settings.AEO_TESTING_MODE = False
    user = User.objects.create_user(username="p3st", email="p3st@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        started_at=timezone.now(),
    )
    r1 = AEOResponseSnapshot.objects.create(profile=profile, prompt_text="a", prompt_hash="ha", raw_response="ra")

    def fake_single_extraction(snapshot, save=True, competitor_hints=None):
        row = AEOExtractionSnapshot.objects.create(
            response_snapshot=snapshot,
            brand_mentioned=True,
            mention_position="top",
            mention_count=1,
            competitors_json=[],
            citations_json=["example.com"],
            sentiment="positive",
            confidence_score=0.9,
            extraction_model="fake",
            extraction_parse_failed=False,
        )
        return {"extraction_snapshot_id": row.id, "save_error": None}

    monkeypatch.setattr("accounts.aeo.aeo_extraction_utils.run_single_extraction", fake_single_extraction)
    monkeypatch.setattr("django.db.close_old_connections", lambda: None)
    monkeypatch.setattr("accounts.tasks.ThreadPoolExecutor", _ImmediateThreadPoolExecutor)
    phase2_calls: list[tuple] = []
    phase4_calls: list[int] = []
    monkeypatch.setattr("accounts.tasks.refresh_competitor_snapshot_for_profile_task.delay", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase2_confidence_task.delay",
        lambda *args, **kwargs: phase2_calls.append(1),
    )
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda rid: phase4_calls.append(rid))

    run_aeo_phase3_extraction_task(run.id, [r1.id])
    assert phase2_calls == []
    assert phase4_calls == [run.id]


@pytest.mark.django_db
def test_phase5_save_recommendation_uses_score_from_execution_run(settings, monkeypatch):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = True
    settings.AEO_RECOMMENDATION_GROUP_GAPS = False
    settings.AEO_RECOMMENDATION_USE_OPENAI = False
    user = User.objects.create_user(username="p5lnk", email="p5lnk@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        website_url="https://example.com",
    )
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    snap_run = AEOScoreSnapshot.objects.create(
        profile=profile,
        visibility_score=22.0,
        weighted_position_score=22.0,
        citation_share=22.0,
        competitor_dominance_json={},
        total_prompts=2,
        total_mentions=2,
    )
    AEOScoreSnapshot.objects.create(
        profile=profile,
        visibility_score=77.0,
        weighted_position_score=77.0,
        citation_share=77.0,
        competitor_dominance_json={},
        total_prompts=2,
        total_mentions=2,
    )
    run.score_snapshot_id = snap_run.id
    run.save(update_fields=["score_snapshot_id", "updated_at"])

    monkeypatch.setattr(
        "accounts.aeo.aeo_recommendation_utils.latest_extraction_per_response",
        lambda _bp, **kwargs: [],
    )

    captured: dict[str, Any] = {}

    def capture_save(bp, **kw):
        captured["score_snapshot"] = kw.get("score_snapshot")
        return AEORecommendationRun.objects.create(
            profile=bp,
            score_snapshot=kw.get("score_snapshot"),
            recommendations_json=[],
            strategies_json=[],
            visibility_score_at_run=float(kw.get("visibility_score") or 0),
            weighted_position_score_at_run=float(kw.get("weighted_position_score") or 0),
            citation_share_at_run=float(kw.get("citation_share") or 0),
        )

    monkeypatch.setattr("accounts.aeo.aeo_recommendation_utils.save_recommendation_run", capture_save)

    run_aeo_phase5_recommendation_task(run.id)
    run.refresh_from_db()
    assert run.recommendation_status == AEOExecutionRun.STAGE_COMPLETED
    assert captured.get("score_snapshot") is not None
    assert captured["score_snapshot"].id == snap_run.id
    assert run.recommendation_run_id is not None
    rec = AEORecommendationRun.objects.get(id=run.recommendation_run_id)
    assert rec.score_snapshot_id == snap_run.id


@pytest.mark.django_db
def test_phase5_reinvocation_calls_generate_twice(settings, monkeypatch):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = True
    user = User.objects.create_user(username="p5r", email="p5r@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    calls: list[int] = []

    def fake_gen(profile, save=True, **kwargs):
        assert kwargs.get("execution_run_id") == run.id
        calls.append(1)
        row = AEORecommendationRun.objects.create(
            profile=profile,
            recommendations_json=[{"rec_id": "x"}],
            strategies_json=[],
        )
        return {"recommendation_run_id": row.id, "recommendations": [{"rec_id": "x"}]}

    monkeypatch.setattr("accounts.aeo.aeo_recommendation_utils.generate_aeo_recommendations", fake_gen)
    monkeypatch.setattr("accounts.tasks._enqueue_seo_after_aeo", lambda _rid: None)

    run_aeo_phase5_recommendation_task(run.id)
    run_aeo_phase5_recommendation_task(run.id)
    assert len(calls) == 2
