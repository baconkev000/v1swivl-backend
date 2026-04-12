"""Unit tests for ``accounts.aeo.prompt_full_ready`` (fully_ready + ETA helpers)."""

import pytest
from django.conf import settings
from django.utils import timezone

# Phase-2 completion helpers mirror ``run_aeo_phase2_confidence_task``; isolate from host env keys.
pytestmark = pytest.mark.usefixtures("_clear_perplexity_key_for_phase2_tests")


@pytest.fixture
def _clear_perplexity_key_for_phase2_tests(settings):
    settings.PERPLEXITY_API_KEY = ""

from accounts.aeo.prompt_full_ready import (
    compute_full_phase_eta_seconds,
    full_phase_eta_cold_start,
    merge_eta_state_after_completions,
    monitored_prompt_fully_ready,
    phase2_passes_complete,
    recommendations_pipeline_settled_for_visibility,
    triple_extraction_complete_for_key,
)
from accounts.aeo.progressive_onboarding import PASSES_PER_PROVIDER_TARGET
from accounts.models import (
    AEOPromptExecutionAggregate,
    AEOExecutionRun,
    BusinessProfile,
)


class _FakeExtractionRelation:
    def __init__(self, has: bool) -> None:
        self._has = has

    def exists(self) -> bool:
        return self._has


class _FakeResp:
    def __init__(self, platform: str, has_extraction: bool) -> None:
        self.platform = platform
        self.extraction_snapshots = _FakeExtractionRelation(has_extraction)


def _latest_per_platform(rows: list) -> dict:
    best: dict[str, object] = {}
    for r in rows:
        p = str(getattr(r, "platform", "") or "").strip().lower()
        if not p or p in best:
            continue
        best[p] = r
    return best


@pytest.mark.django_db
def test_phase2_passes_complete_none():
    assert phase2_passes_complete(None) is False


@pytest.mark.django_db
def test_phase2_passes_complete_needs_more_passes_when_both_providers_active():
    agg = AEOPromptExecutionAggregate(
        openai_pass_count=1,
        gemini_pass_count=1,
        openai_third_pass_required=False,
        gemini_third_pass_required=False,
        openai_third_pass_ran=False,
        gemini_third_pass_ran=False,
    )
    assert phase2_passes_complete(agg) is False


@pytest.mark.django_db
def test_phase2_passes_complete_gemini_only_skips_openai():
    agg = AEOPromptExecutionAggregate(
        openai_pass_count=0,
        gemini_pass_count=PASSES_PER_PROVIDER_TARGET,
        openai_third_pass_required=False,
        gemini_third_pass_required=False,
        openai_third_pass_ran=False,
        gemini_third_pass_ran=False,
    )
    assert phase2_passes_complete(agg) is True


@pytest.mark.django_db
def test_phase2_third_pass_branch():
    agg = AEOPromptExecutionAggregate(
        openai_pass_count=PASSES_PER_PROVIDER_TARGET,
        gemini_pass_count=PASSES_PER_PROVIDER_TARGET,
        openai_third_pass_required=True,
        gemini_third_pass_required=False,
        openai_third_pass_ran=False,
        gemini_third_pass_ran=False,
    )
    assert phase2_passes_complete(agg) is False
    agg.openai_third_pass_ran = True
    assert phase2_passes_complete(agg) is True


@pytest.mark.django_db
def test_triple_extraction_requires_perplexity():
    by_prompt = {
        "p": [
            _FakeResp("openai", True),
            _FakeResp("gemini", True),
        ],
    }
    assert triple_extraction_complete_for_key("p", by_prompt, _latest_per_platform) is False
    by_prompt["p"].append(_FakeResp("perplexity", True))
    assert triple_extraction_complete_for_key("p", by_prompt, _latest_per_platform) is True


@pytest.mark.django_db
def test_monitored_prompt_fully_ready_requires_rec_settled_and_aggregate(
    django_user_model,
):
    from accounts.aeo.aeo_execution_utils import hash_prompt

    user = django_user_model.objects.create_user(username="fr1", email="fr1@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="B", selected_aeo_prompts=["x"])
    by_prompt = {
        "x": [
            _FakeResp("openai", True),
            _FakeResp("gemini", True),
            _FakeResp("perplexity", True),
        ],
    }
    assert (
        monitored_prompt_fully_ready("x", profile, by_prompt, _latest_per_platform, recs_settled=False) is False
    )
    assert (
        monitored_prompt_fully_ready("x", profile, by_prompt, _latest_per_platform, recs_settled=True) is False
    )
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    AEOPromptExecutionAggregate.objects.create(
        profile=profile,
        execution_run=run,
        prompt_hash=hash_prompt("x"),
        openai_pass_count=PASSES_PER_PROVIDER_TARGET,
        gemini_pass_count=PASSES_PER_PROVIDER_TARGET,
    )
    assert (
        monitored_prompt_fully_ready("x", profile, by_prompt, _latest_per_platform, recs_settled=True) is True
    )


@pytest.mark.django_db
def test_recommendations_pipeline_settled_phase5_off(django_user_model):
    user = django_user_model.objects.create_user(username="fr2", email="fr2@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True)
    prev = settings.AEO_ENABLE_RECOMMENDATION_STAGE
    try:
        settings.AEO_ENABLE_RECOMMENDATION_STAGE = False
        assert recommendations_pipeline_settled_for_visibility(profile) is True
    finally:
        settings.AEO_ENABLE_RECOMMENDATION_STAGE = prev


@pytest.mark.django_db
def test_recommendations_pipeline_settled_phase5_blocks_while_rec_running(django_user_model):
    user = django_user_model.objects.create_user(username="fr3", email="fr3@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True)
    prev = settings.AEO_ENABLE_RECOMMENDATION_STAGE
    try:
        settings.AEO_ENABLE_RECOMMENDATION_STAGE = True
        AEOExecutionRun.objects.create(
            profile=profile,
            status=AEOExecutionRun.STATUS_COMPLETED,
            scoring_status=AEOExecutionRun.STAGE_COMPLETED,
            recommendation_status=AEOExecutionRun.STAGE_RUNNING,
        )
        assert recommendations_pipeline_settled_for_visibility(profile) is False
    finally:
        settings.AEO_ENABLE_RECOMMENDATION_STAGE = prev


def test_compute_eta_cold_start_no_samples():
    eta, cold = compute_full_phase_eta_seconds([], remaining=3, full_phase_completed=0)
    assert eta is None and cold is True


def test_compute_eta_uses_average_after_samples():
    eta, cold = compute_full_phase_eta_seconds([60.0, 120.0], remaining=2, full_phase_completed=2)
    assert cold is False
    assert eta == int(min(3600, 2 * 90.0))


@pytest.mark.django_db
def test_merge_eta_state_appends_once(django_user_model):
    from datetime import timedelta

    from accounts.models import AEOResponseSnapshot

    user = django_user_model.objects.create_user(username="fr4", email="fr4@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, selected_aeo_prompts=["a"])
    snap = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="a",
        prompt_hash="h",
        raw_response="x",
        platform="openai",
    )
    t0 = timezone.now() - timedelta(seconds=100)
    AEOResponseSnapshot.objects.filter(pk=snap.pk).update(created_at=t0)
    now = timezone.now()
    per = {"a": True}
    merge_eta_state_after_completions(profile, ["a"], per, now=now)
    profile.refresh_from_db()
    d1 = (profile.aeo_full_phase_eta_state or {}).get("durations") or []
    assert len(d1) == 1
    merge_eta_state_after_completions(profile, ["a"], per, now=now)
    profile.refresh_from_db()
    d2 = (profile.aeo_full_phase_eta_state or {}).get("durations") or []
    assert len(d2) == 1


def test_full_phase_eta_cold_start_helper():
    assert full_phase_eta_cold_start([], 0) is True
    assert full_phase_eta_cold_start([1.0], 1) is False
