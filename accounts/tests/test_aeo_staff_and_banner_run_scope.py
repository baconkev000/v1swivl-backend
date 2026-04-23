"""Staff full AEO re-run control and prompt-coverage banner scoped to latest execution run."""

from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from accounts.models import (
    AEOExecutionRun,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    AEOScoreSnapshot,
    BusinessProfile,
)
from accounts.tasks import (
    try_enqueue_aeo_full_monitored_pipeline,
    try_enqueue_aeo_phase5_recommendations_only,
)

User = get_user_model()


@pytest.mark.django_db
def test_try_enqueue_full_pipeline_respects_inflight(monkeypatch):
    user = User.objects.create_user(username="enq1", email="enq1@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="B",
        selected_aeo_prompts=["alpha"],
    )
    AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_RUNNING)
    out = try_enqueue_aeo_full_monitored_pipeline(profile.id, source="test")
    assert out["queued"] is False
    assert out["reason"] == "duplicate_inflight"


@pytest.mark.django_db(transaction=True)
def test_try_enqueue_creates_run_and_calls_phase1_delay(monkeypatch):
    delayed: list[tuple] = []

    def capture_delay(*args, **kwargs):
        delayed.append((args, kwargs))

    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", capture_delay)

    user = User.objects.create_user(username="enq2", email="enq2@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="B",
        selected_aeo_prompts=["alpha"],
    )
    out = try_enqueue_aeo_full_monitored_pipeline(profile.id, source="test")
    assert out["queued"] is True
    assert out["run_id"] is not None
    assert len(delayed) == 1
    assert delayed[0][0][0] == out["run_id"]
    assert delayed[0][1].get("force_refresh") is True


@pytest.mark.django_db
def test_staff_full_rerun_post_forbidden_for_non_staff():
    client = Client()
    user = User.objects.create_user(username="ns", email="ns@example.com", password="pw")
    client.force_login(user)
    resp = client.post(
        reverse("staff-aeo-pass-counts"),
        data={
            "action": "full_rerun",
            "aeo_full_rerun_profile_id": "1",
            "redirect_run_id": "all",
            "redirect_profile_id": "1",
        },
    )
    assert resp.status_code == 403


@pytest.mark.django_db
def test_staff_full_rerun_post_calls_enqueue_helper(monkeypatch):
    calls: list[tuple] = []

    def stub(pid, *, source=""):
        calls.append((pid, source))
        return {
            "ok": True,
            "queued": True,
            "run_id": 42,
            "reason": "enqueued",
            "message": "Queued.",
            "prompt_count": 3,
        }

    monkeypatch.setattr("accounts.tasks.try_enqueue_aeo_full_monitored_pipeline", stub)

    client = Client()
    staff = User.objects.create_user(
        username="stf",
        email="stf@example.com",
        password="pw",
        is_staff=True,
    )
    client.force_login(staff)
    resp = client.post(
        reverse("staff-aeo-pass-counts"),
        data={
            "action": "full_rerun",
            "aeo_full_rerun_profile_id": "7",
            "redirect_run_id": "all",
            "redirect_profile_id": "7",
        },
    )
    assert resp.status_code == 302
    assert calls == [(7, "staff_aeo_pass_counts")]
    assert "profile_id=7" in resp.url or resp.url.endswith("7")


@pytest.mark.django_db(transaction=True)
def test_staff_phase5_recommendations_post_calls_phase5_delay_once(monkeypatch, settings):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = True
    delayed: list[int] = []

    def capture_delay(rid: int):
        delayed.append(int(rid))

    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase5_recommendation_task.delay", capture_delay
    )

    client = Client()
    staff = User.objects.create_user(
        username="p5st",
        email="p5st@example.com",
        password="pw",
        is_staff=True,
    )
    client.force_login(staff)
    user = User.objects.create_user(username="p5u", email="p5u@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="P5Co",
        selected_aeo_prompts=["alpha"],
    )
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        scoring_status=AEOExecutionRun.STAGE_COMPLETED,
        extraction_status=AEOExecutionRun.STAGE_COMPLETED,
    )
    snap = AEOScoreSnapshot.objects.create(profile=profile, execution_run=run, total_prompts=1)
    run.score_snapshot_id = snap.id
    run.save(update_fields=["score_snapshot_id"])

    resp = client.post(
        reverse("staff-aeo-pass-counts"),
        data={
            "action": "phase5_recommendations_only",
            "aeo_phase5_rerun_profile_id": str(profile.id),
            "redirect_run_id": "all",
            "redirect_profile_id": str(profile.id),
        },
    )
    assert resp.status_code == 302
    assert delayed == [run.id]


@pytest.mark.django_db
def test_staff_phase5_post_all_profiles_warning(monkeypatch, settings):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = True
    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase5_recommendation_task.delay",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not enqueue")),
    )
    client = Client()
    staff = User.objects.create_user(
        username="p5all",
        email="p5all@example.com",
        password="pw",
        is_staff=True,
    )
    client.force_login(staff)
    resp = client.post(
        reverse("staff-aeo-pass-counts"),
        data={
            "action": "phase5_recommendations_only",
            "aeo_phase5_rerun_profile_id": "all",
            "redirect_run_id": "all",
            "redirect_profile_id": "all",
        },
        follow=True,
    )
    assert resp.status_code == 200


@pytest.mark.django_db
def test_staff_phase5_inflight_main_pipeline_no_delay(monkeypatch, settings):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = True
    delayed: list[int] = []

    monkeypatch.setattr(
        "accounts.tasks.run_aeo_phase5_recommendation_task.delay",
        lambda rid, *a, **k: delayed.append(int(rid)),
    )

    client = Client()
    staff = User.objects.create_user(
        username="p5inf",
        email="p5inf@example.com",
        password="pw",
        is_staff=True,
    )
    client.force_login(staff)
    user = User.objects.create_user(username="p5u2", email="p5u2@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="P5Inf",
        selected_aeo_prompts=["alpha"],
    )
    AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_RUNNING)
    AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        scoring_status=AEOExecutionRun.STAGE_COMPLETED,
        extraction_status=AEOExecutionRun.STAGE_COMPLETED,
    )

    resp = client.post(
        reverse("staff-aeo-pass-counts"),
        data={
            "action": "phase5_recommendations_only",
            "aeo_phase5_rerun_profile_id": str(profile.id),
            "redirect_run_id": "all",
            "redirect_profile_id": str(profile.id),
        },
    )
    assert resp.status_code == 302
    assert delayed == []


@pytest.mark.django_db
def test_try_enqueue_phase5_only_respects_main_pipeline_inflight(settings):
    settings.AEO_ENABLE_RECOMMENDATION_STAGE = True
    user = User.objects.create_user(username="tp5", email="tp5@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="T",
        selected_aeo_prompts=["x"],
    )
    AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_RUNNING)
    out = try_enqueue_aeo_phase5_recommendations_only(profile.id, source="test")
    assert out["queued"] is False
    assert out["reason"] == "duplicate_inflight"


@pytest.mark.django_db
def test_prompt_coverage_banner_prompt_scan_scoped_to_latest_run_only():
    pytest.importorskip("stripe")
    from accounts.views import _build_aeo_prompt_coverage_payload

    user = User.objects.create_user(username="bnr", email="bnr@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        selected_aeo_prompts=["One"],
    )
    old_run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
    )
    new_run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
    )

    def _triple_with_extraction(run):
        for plat in ("openai", "gemini", "perplexity"):
            rsp = AEOResponseSnapshot.objects.create(
                profile=profile,
                execution_run=run,
                prompt_text="One",
                prompt_hash=f"h-{plat}",
                raw_response="x",
                platform=plat,
            )
            AEOExtractionSnapshot.objects.create(
                response_snapshot=rsp,
                brand_mentioned=False,
                mention_position="none",
                mention_count=0,
                competitors_json=[],
                citations_json=[],
                sentiment="neutral",
                confidence_score=0.5,
                extraction_model="t",
                extraction_parse_failed=False,
            )

    _triple_with_extraction(old_run)

    payload = _build_aeo_prompt_coverage_payload(profile)
    assert payload["prompt_scan_total"] == 1
    assert payload["prompt_scan_completed"] == 0

    _triple_with_extraction(new_run)
    payload2 = _build_aeo_prompt_coverage_payload(profile)
    assert payload2["prompt_scan_completed"] == 1


@pytest.mark.django_db
def test_aeo_scheduled_tick_skips_when_disabled(monkeypatch, settings):
    settings.AEO_SCHEDULED_FULL_MONITORING_ENABLED = False
    settings.AEO_SCHEDULED_FULL_MONITORING_PROFILE_IDS = [1, 2]
    called: list[int] = []

    def capture(pid, *, source=""):
        called.append(pid)
        return {"ok": True, "queued": True, "run_id": 1, "reason": "x", "message": "", "prompt_count": 0}

    monkeypatch.setattr("accounts.tasks.try_enqueue_aeo_full_monitored_pipeline", capture)
    from accounts.tasks import aeo_scheduled_full_monitoring_tick_task

    aeo_scheduled_full_monitoring_tick_task.run()
    assert called == []


@pytest.mark.django_db
def test_staff_seo_snapshot_refresh_post_queues_full_seo_celery_task(monkeypatch):
    queued: list[int] = []

    monkeypatch.setattr(
        "accounts.home_views.sync_enrich_seo_snapshot_for_profile_task.delay",
        lambda pid: queued.append(int(pid)),
    )

    client = Client()
    staff = User.objects.create_user(
        username="seostf",
        email="seostf@example.com",
        password="pw",
        is_staff=True,
    )
    owner = User.objects.create_user(username="seoo", email="seoo@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=owner,
        is_main=True,
        business_name="SEO Co",
        website_url="https://seo-co.example.com",
        selected_aeo_prompts=["a"],
    )
    client.force_login(staff)
    resp = client.post(
        reverse("staff-aeo-pass-counts"),
        data={
            "action": "seo_snapshot_refresh",
            "seo_snapshot_refresh_profile_id": str(profile.id),
            "redirect_run_id": "all",
            "redirect_profile_id": str(profile.id),
        },
    )
    assert resp.status_code == 302
    assert queued == [profile.id]


@pytest.mark.django_db
def test_staff_seo_snapshot_refresh_post_all_profiles_no_refresh(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("get_or_refresh should not run")

    monkeypatch.setattr(
        "accounts.dataforseo_utils.get_or_refresh_seo_score_for_user", boom
    )
    client = Client()
    staff = User.objects.create_user(
        username="seoall",
        email="seoall@example.com",
        password="pw",
        is_staff=True,
    )
    client.force_login(staff)
    resp = client.post(
        reverse("staff-aeo-pass-counts"),
        data={
            "action": "seo_snapshot_refresh",
            "seo_snapshot_refresh_profile_id": "all",
            "redirect_run_id": "all",
            "redirect_profile_id": "all",
        },
    )
    assert resp.status_code == 302


@pytest.mark.django_db
def test_staff_seo_snapshot_refresh_skips_without_website(monkeypatch):
    called: list[bool] = []

    def capture(*_a, **_k):
        called.append(True)
        return {}

    monkeypatch.setattr(
        "accounts.dataforseo_utils.get_or_refresh_seo_score_for_user", capture
    )
    client = Client()
    staff = User.objects.create_user(
        username="seonoweb",
        email="seonoweb@example.com",
        password="pw",
        is_staff=True,
    )
    owner = User.objects.create_user(username="seonowebu", email="seonowebu@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=owner,
        is_main=True,
        business_name="No Web",
        website_url="",
        selected_aeo_prompts=["a"],
    )
    client.force_login(staff)
    resp = client.post(
        reverse("staff-aeo-pass-counts"),
        data={
            "action": "seo_snapshot_refresh",
            "seo_snapshot_refresh_profile_id": str(profile.id),
            "redirect_run_id": "all",
            "redirect_profile_id": str(profile.id),
        },
    )
    assert resp.status_code == 302
    assert called == []
