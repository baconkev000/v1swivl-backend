import pytest
from django.contrib.auth import get_user_model

from accounts.aeo.progressive_onboarding import (
    PHASE1_CATEGORIES,
    build_phase1_provider_batches,
    classify_prompt_category,
    recompute_stability,
)
from accounts.models import AEOPromptExecutionAggregate, AEOExecutionRun, AEOScoreSnapshot, BusinessProfile
from accounts.tasks import run_aeo_phase4_scoring_task

User = get_user_model()


def _mk_prompt(text: str, ptype: str = "transactional") -> dict:
    return {"prompt": text, "type": ptype, "weight": 1.0, "dynamic": True}


@pytest.mark.django_db
def test_phase1_distribution_provider_balanced():
    prompts = [
        _mk_prompt("authority dentist credentials", "authority"),
        _mk_prompt("authority dentist board certified", "authority"),
        _mk_prompt("comparison invisalign vs braces", "comparison"),
        _mk_prompt("comparison best provider near me", "comparison"),
        _mk_prompt("pricing dental implant cost", "transactional"),
        _mk_prompt("pricing veneer quote", "transactional"),
        _mk_prompt("trust safest clinic", "trust"),
        _mk_prompt("trust most trusted dentist", "trust"),
        _mk_prompt("service emergency dentist service", "transactional"),
        _mk_prompt("service same day crown provider", "transactional"),
    ]
    batches = build_phase1_provider_batches(prompts)
    assert len(batches["openai"]) == 5
    assert len(batches["gemini"]) == 5
    cats_openai = {classify_prompt_category(p) for p in batches["openai"]}
    cats_gemini = {classify_prompt_category(p) for p in batches["gemini"]}
    assert cats_openai == set(PHASE1_CATEGORIES)
    assert cats_gemini == set(PHASE1_CATEGORIES)


def test_phase1_distribution_has_exactly_10_calls():
    prompts = [
        _mk_prompt("authority a", "authority"),
        _mk_prompt("authority b", "authority"),
        _mk_prompt("comparison a", "comparison"),
        _mk_prompt("comparison b", "comparison"),
        _mk_prompt("pricing a cost", "transactional"),
        _mk_prompt("pricing b quote", "transactional"),
        _mk_prompt("trust a", "trust"),
        _mk_prompt("trust b", "trust"),
        _mk_prompt("service a", "transactional"),
        _mk_prompt("service b", "transactional"),
    ]
    batches = build_phase1_provider_batches(prompts)
    assert len(batches["openai"]) + len(batches["gemini"]) == 10


@pytest.mark.django_db
def test_stability_unstable_when_sets_change():
    user = User.objects.create_user(username="stab", email="stab@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    agg = AEOPromptExecutionAggregate.objects.create(
        profile=profile,
        execution_run=run,
        prompt_hash="h",
        openai_pass_count=2,
        gemini_pass_count=2,
        last_openai_citations_json=["a.com"],
        last_gemini_citations_json=["b.com"],
        last_openai_competitors_json=[{"name": "A"}],
        last_gemini_competitors_json=[{"name": "B"}],
    )
    status, reasons = recompute_stability(agg)
    assert status == AEOPromptExecutionAggregate.STABILITY_UNSTABLE
    assert "citation_set_changed" in reasons
    assert "competitor_set_changed" in reasons


@pytest.mark.django_db
def test_score_layers_persist_queryable(monkeypatch, settings):
    settings.AEO_TESTING_MODE = True
    settings.AEO_TEST_PROMPT_COUNT = 10
    settings.PERPLEXITY_API_KEY = ""
    user = User.objects.create_user(username="layer", email="layer@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    AEOPromptExecutionAggregate.objects.create(
        profile=profile,
        execution_run=run,
        prompt_hash="h1",
        total_pass_count=4,
        openai_pass_count=2,
        gemini_pass_count=2,
        total_brand_cited_count=2,
    )
    monkeypatch.setattr("accounts.tasks._aeo_recommendation_stage_enabled", lambda: False)
    monkeypatch.setattr("accounts.tasks._enqueue_seo_after_aeo", lambda _rid: None)
    run_aeo_phase4_scoring_task(run.id)
    layers = set(AEOScoreSnapshot.objects.filter(profile=profile).values_list("score_layer", flat=True))
    assert {"sample", "confidence"}.issubset(layers)
