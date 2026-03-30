import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.models import (
    AEOExecutionRun,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    AEOScoreSnapshot,
    BusinessProfile,
)
from accounts.tasks import (
    run_aeo_phase1_execution_task,
    run_aeo_phase3_extraction_task,
    run_aeo_phase4_scoring_task,
    trigger_seo_warmup_after_aeo_task,
)


User = get_user_model()


@pytest.mark.django_db
def test_phase1_completion_enqueues_extraction_on_partial_failures(monkeypatch):
    user = User.objects.create_user(username="p1", email="p1@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_PENDING)

    snap = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="q1",
        prompt_hash="h1",
        raw_response="r1",
    )

    monkeypatch.setattr(
        "accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch",
        lambda *args, **kwargs: {
            "executed": 1,
            "failed": 1,
            "results": [
                {"success": True, "snapshot_id": snap.id},
                {"success": False, "snapshot_id": None},
            ],
        },
    )

    queued = []
    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase3_extraction_task.delay",
        lambda run_id, snapshot_ids=None: queued.append((run_id, snapshot_ids or [])),
    )

    run_aeo_phase1_execution_task(run.id, [{"prompt": "q1"}])
    run.refresh_from_db()
    assert run.status == AEOExecutionRun.STATUS_COMPLETED
    assert run.prompt_count_executed == 1
    assert run.prompt_count_failed == 1
    assert queued == [(run.id, [snap.id])]


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
    queued = []
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda run_id: queued.append(run_id))

    run_aeo_phase3_extraction_task(run.id, [r1.id, r2.id])
    run.refresh_from_db()
    assert run.extraction_status == AEOExecutionRun.STAGE_COMPLETED
    assert run.extraction_count == 2
    assert AEOExtractionSnapshot.objects.filter(response_snapshot__profile=profile).count() == 2
    assert queued == [run.id]


@pytest.mark.django_db
def test_phase4_creates_score_snapshot_with_expected_fields(monkeypatch, settings):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = False
    user = User.objects.create_user(username="p4", email="p4@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
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
    trigger_seo_warmup_after_aeo_task(run.id)
    trigger_seo_warmup_after_aeo_task(run.id)
    run.refresh_from_db()
    assert calls["count"] == 1
    assert run.seo_trigger_status == "success"
