import pytest
from django.contrib.auth import get_user_model
from django.test import override_settings

from accounts.aeo.aeo_execution_utils import PLATFORM_GEMINI, PLATFORM_OPENAI, PLATFORM_PERPLEXITY, hash_prompt
from accounts.aeo.progressive_onboarding import update_prompt_aggregate_from_extraction
from accounts.models import (
    AEOPromptExecutionAggregate,
    AEOExecutionRun,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    BusinessProfile,
)
from accounts.tasks import run_aeo_phase2_confidence_task
from accounts.third_party_usage import build_aeo_pass_count_analytics_context

User = get_user_model()


def _mk_response(profile, run, prompt_text, platform, idx):
    return AEOResponseSnapshot.objects.create(
        profile=profile,
        execution_run=run,
        prompt_text=prompt_text,
        prompt_hash=hash_prompt(prompt_text),
        raw_response=f"raw-{platform}-{idx}",
        platform=platform,
    )


def _mk_extraction(resp, mentioned: bool, *, competitors=None, citations=None, wrong_url_status=None):
    return AEOExtractionSnapshot.objects.create(
        response_snapshot=resp,
        brand_mentioned=mentioned,
        mention_position="top" if mentioned else "none",
        mention_count=1 if mentioned else 0,
        competitors_json=list(competitors or []),
        citations_json=list(citations or []),
        sentiment="neutral",
        confidence_score=0.9,
        extraction_model="t",
        extraction_parse_failed=False,
        brand_mentioned_url_status=wrong_url_status,
    )


@pytest.mark.django_db
def test_provider_stable_after_two_no_third_required():
    user = User.objects.create_user(username="ps1", email="ps1@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    prompt = "best dentist near me"
    for i in range(2):
        ro = _mk_response(profile, run, prompt, PLATFORM_OPENAI, i)
        ex = _mk_extraction(ro, True)
        agg = update_prompt_aggregate_from_extraction(
            profile=profile,
            execution_run_id=run.id,
            response_snapshot=ro,
            extraction_snapshot=ex,
            prompt_category="service",
        )
    assert agg.openai_pass_count == 2
    assert agg.openai_stability_status == AEOPromptExecutionAggregate.STABILITY_STABLE
    assert agg.openai_third_pass_required is False


@pytest.mark.django_db
def test_provider_unstable_after_two_requires_third():
    user = User.objects.create_user(username="ps2", email="ps2@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    prompt = "dentist pricing"
    ro1 = _mk_response(profile, run, prompt, PLATFORM_OPENAI, 1)
    ex1 = _mk_extraction(ro1, True)
    update_prompt_aggregate_from_extraction(
        profile=profile,
        execution_run_id=run.id,
        response_snapshot=ro1,
        extraction_snapshot=ex1,
        prompt_category="pricing",
    )
    ro2 = _mk_response(profile, run, prompt, PLATFORM_OPENAI, 2)
    ex2 = _mk_extraction(ro2, False)
    agg = update_prompt_aggregate_from_extraction(
        profile=profile,
        execution_run_id=run.id,
        response_snapshot=ro2,
        extraction_snapshot=ex2,
        prompt_category="pricing",
    )
    assert agg.openai_pass_count == 2
    assert agg.openai_stability_status == AEOPromptExecutionAggregate.STABILITY_UNSTABLE
    assert agg.openai_third_pass_required is True
    assert agg.openai_third_pass_ran is False


@pytest.mark.django_db
def test_mixed_provider_only_unstable_provider_gets_third_pass_batch(monkeypatch):
    user = User.objects.create_user(username="ps3", email="ps3@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    prompt = "best emergency dentist"
    profile.selected_aeo_prompts = [prompt]
    profile.save(update_fields=["selected_aeo_prompts", "updated_at"])

    # OpenAI unstable after 2
    ro1 = _mk_response(profile, run, prompt, PLATFORM_OPENAI, 1)
    ex1 = _mk_extraction(ro1, True)
    update_prompt_aggregate_from_extraction(
        profile=profile, execution_run_id=run.id, response_snapshot=ro1, extraction_snapshot=ex1, prompt_category="service"
    )
    ro2 = _mk_response(profile, run, prompt, PLATFORM_OPENAI, 2)
    ex2 = _mk_extraction(ro2, False)
    update_prompt_aggregate_from_extraction(
        profile=profile, execution_run_id=run.id, response_snapshot=ro2, extraction_snapshot=ex2, prompt_category="service"
    )
    # Gemini stable after 2
    rg1 = _mk_response(profile, run, prompt, PLATFORM_GEMINI, 1)
    eg1 = _mk_extraction(rg1, True)
    update_prompt_aggregate_from_extraction(
        profile=profile, execution_run_id=run.id, response_snapshot=rg1, extraction_snapshot=eg1, prompt_category="service"
    )
    rg2 = _mk_response(profile, run, prompt, PLATFORM_GEMINI, 2)
    eg2 = _mk_extraction(rg2, True)
    update_prompt_aggregate_from_extraction(
        profile=profile, execution_run_id=run.id, response_snapshot=rg2, extraction_snapshot=eg2, prompt_category="service"
    )

    calls = []

    def fake_batch(prompt_set, business_profile, **kwargs):
        calls.append((tuple(kwargs.get("providers") or []), [p.get("prompt") for p in prompt_set]))
        return {"executed": 0, "failed": 0, "results": []}

    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda _rid: None)
    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch", fake_batch)
    run_aeo_phase2_confidence_task(run.id, [{"prompt": prompt, "type": "transactional", "weight": 1.0, "dynamic": True}])
    assert any(c[0] == ("openai",) for c in calls)
    assert not any(c[0] == ("gemini",) for c in calls)


@pytest.mark.django_db
def test_pass_count_reporting_aggregation():
    user = User.objects.create_user(username="ps4", email="ps4@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    AEOPromptExecutionAggregate.objects.create(
        profile=profile,
        execution_run=run,
        prompt_hash="h1",
        openai_pass_count=2,
        gemini_pass_count=2,
        openai_stability_status="stable",
        gemini_stability_status="stable",
    )
    AEOPromptExecutionAggregate.objects.create(
        profile=profile,
        execution_run=run,
        prompt_hash="h2",
        openai_pass_count=3,
        gemini_pass_count=2,
        openai_stability_status="stabilized_after_third",
        gemini_stability_status="stable",
        openai_third_pass_required=False,
        openai_third_pass_ran=True,
    )
    out = build_aeo_pass_count_analytics_context(execution_run_id=run.id, profile_id=profile.id)
    assert out["total_prompts"] == 2
    assert out["providers"]["openai"]["third_completed"] == 1
    assert out["providers"]["gemini"]["stable_at_2"] >= 1
    assert "perplexity" in out["providers"]
    assert out["providers"]["perplexity"]["total"] == 0


@pytest.mark.django_db
def test_per_provider_totals_stable_stable_ends_at_four_and_keeps_pass_histories():
    user = User.objects.create_user(username="ps5", email="ps5@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    prompt = "family dentist near me"
    # OpenAI stable (False, False)
    for i in range(2):
        ro = _mk_response(profile, run, prompt, PLATFORM_OPENAI, i)
        ex = _mk_extraction(
            ro,
            False,
            competitors=[{"name": f"o-comp-{i}"}],
            citations=[{"url": f"https://o{i}.example"}],
        )
        agg = update_prompt_aggregate_from_extraction(
            profile=profile, execution_run_id=run.id, response_snapshot=ro, extraction_snapshot=ex, prompt_category="service"
        )
    # Gemini stable (True, True)
    for i in range(2):
        rg = _mk_response(profile, run, prompt, PLATFORM_GEMINI, i)
        eg = _mk_extraction(
            rg,
            True,
            competitors=[{"name": f"g-comp-{i}"}],
            citations=[{"url": f"https://g{i}.example"}],
        )
        agg = update_prompt_aggregate_from_extraction(
            profile=profile, execution_run_id=run.id, response_snapshot=rg, extraction_snapshot=eg, prompt_category="service"
        )
    assert agg.openai_pass_count == 2
    assert agg.gemini_pass_count == 2
    assert agg.total_pass_count == 4
    assert len(agg.openai_pass_history_json) == 2
    assert len(agg.gemini_pass_history_json) == 2
    assert agg.openai_pass_history_json[0]["competitors"][0]["name"] == "o-comp-0"
    assert agg.gemini_pass_history_json[1]["citations"][0]["url"] == "https://g1.example"


@pytest.mark.django_db
def test_per_provider_totals_stable_unstable_ends_at_five():
    user = User.objects.create_user(username="ps6", email="ps6@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    prompt = "dental implant options"
    # OpenAI unstable then third resolves
    for i, m in enumerate([True, False, True]):
        ro = _mk_response(profile, run, prompt, PLATFORM_OPENAI, i)
        ex = _mk_extraction(ro, m, competitors=[{"name": f"o-{i}"}])
        agg = update_prompt_aggregate_from_extraction(
            profile=profile, execution_run_id=run.id, response_snapshot=ro, extraction_snapshot=ex, prompt_category="comparison"
        )
    # Gemini stable two passes
    for i in range(2):
        rg = _mk_response(profile, run, prompt, PLATFORM_GEMINI, i)
        eg = _mk_extraction(rg, False, competitors=[{"name": f"g-{i}"}])
        agg = update_prompt_aggregate_from_extraction(
            profile=profile, execution_run_id=run.id, response_snapshot=rg, extraction_snapshot=eg, prompt_category="comparison"
        )
    assert agg.openai_pass_count == 3
    assert agg.gemini_pass_count == 2
    assert agg.total_pass_count == 5
    assert len(agg.openai_pass_history_json) == 3
    assert len(agg.gemini_pass_history_json) == 2


@pytest.mark.django_db
def test_per_provider_totals_unstable_unstable_ends_at_six_not_global_three_cap():
    user = User.objects.create_user(username="ps7", email="ps7@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    prompt = "best root canal specialist"
    # OpenAI unstable then third
    for i, m in enumerate([True, False, True]):
        ro = _mk_response(profile, run, prompt, PLATFORM_OPENAI, i)
        ex = _mk_extraction(ro, m, competitors=[{"name": f"o-u-{i}"}], citations=[{"url": f"https://ou{i}.ex"}])
        agg = update_prompt_aggregate_from_extraction(
            profile=profile, execution_run_id=run.id, response_snapshot=ro, extraction_snapshot=ex, prompt_category="authority"
        )
    # Gemini unstable then third
    for i, m in enumerate([False, True, False]):
        rg = _mk_response(profile, run, prompt, PLATFORM_GEMINI, i)
        eg = _mk_extraction(rg, m, competitors=[{"name": f"g-u-{i}"}], citations=[{"url": f"https://gu{i}.ex"}])
        agg = update_prompt_aggregate_from_extraction(
            profile=profile, execution_run_id=run.id, response_snapshot=rg, extraction_snapshot=eg, prompt_category="authority"
        )
    assert agg.openai_pass_count == 3
    assert agg.gemini_pass_count == 3
    assert agg.total_pass_count == 6
    assert len(agg.openai_pass_history_json) == 3
    assert len(agg.gemini_pass_history_json) == 3


@pytest.mark.django_db
def test_combined_rollups_dedupe_within_pass_and_provider_breakdown_deterministic():
    user = User.objects.create_user(username="ps8", email="ps8@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    prompt = "invisalign options"
    ro1 = _mk_response(profile, run, prompt, PLATFORM_OPENAI, 1)
    ex1 = _mk_extraction(
        ro1,
        True,
        competitors=[{"name": "Acme Dental"}, {"name": "acme   dental"}],
        citations=[{"url": "https://www.example.com/a"}, {"url": "http://example.com/b"}],
    )
    update_prompt_aggregate_from_extraction(
        profile=profile, execution_run_id=run.id, response_snapshot=ro1, extraction_snapshot=ex1, prompt_category="service"
    )
    ro2 = _mk_response(profile, run, prompt, PLATFORM_OPENAI, 2)
    ex2 = _mk_extraction(
        ro2,
        True,
        competitors=[{"name": "Acme Dental", "url": "https://acme.test"}],
        citations=[{"url": "https://example.com/c"}],
    )
    update_prompt_aggregate_from_extraction(
        profile=profile, execution_run_id=run.id, response_snapshot=ro2, extraction_snapshot=ex2, prompt_category="service"
    )
    rg1 = _mk_response(profile, run, prompt, PLATFORM_GEMINI, 1)
    eg1 = _mk_extraction(
        rg1,
        False,
        competitors=[{"name": "Beta Ortho"}],
        citations=[{"url": "https://sample.org/1"}],
    )
    agg = update_prompt_aggregate_from_extraction(
        profile=profile, execution_run_id=run.id, response_snapshot=rg1, extraction_snapshot=eg1, prompt_category="service"
    )
    # dedupe within pass counts once; then sum across passes
    assert agg.combined_competitor_counts["acme dental"] == 1
    assert agg.combined_competitor_counts["acme dental|acme.test"] == 1
    assert agg.combined_citation_counts["example.com"] == 2
    assert agg.combined_provider_breakdown["openai"]["citations"]["example.com"] == 2
    assert agg.combined_provider_breakdown["gemini"]["citations"]["sample.org"] == 1
    assert "perplexity" in agg.combined_provider_breakdown
    assert agg.combined_total_passes_observed == 3
    assert agg.combined_total_unique_competitors >= 2
    assert agg.combined_total_unique_citations >= 2


@pytest.mark.django_db
def test_perplexity_extraction_updates_aggregate_pass_counts_and_history():
    user = User.objects.create_user(username="ps_p1", email="ps_p1@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    prompt = "best electric toothbrush"
    rp = _mk_response(profile, run, prompt, PLATFORM_PERPLEXITY, 0)
    ep = _mk_extraction(rp, True, competitors=[{"name": "P Co"}], citations=[{"url": "https://p.example/a"}])
    agg = update_prompt_aggregate_from_extraction(
        profile=profile,
        execution_run_id=run.id,
        response_snapshot=rp,
        extraction_snapshot=ep,
        prompt_category="comparison",
    )
    assert agg.perplexity_pass_count == 1
    assert len(agg.perplexity_pass_history_json) == 1
    assert agg.combined_total_passes_observed == 1
    assert agg.combined_provider_breakdown["perplexity"]["competitors"]["p co"] == 1


@pytest.mark.django_db
@override_settings(PERPLEXITY_API_KEY="pk-p2")
def test_phase2_enqueues_perplexity_when_openai_and_gemini_complete():
    user = User.objects.create_user(username="ps_p2", email="ps_p2@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile, status=AEOExecutionRun.STATUS_COMPLETED)
    prompt = "compare project tools"
    profile.selected_aeo_prompts = [prompt]
    profile.save(update_fields=["selected_aeo_prompts", "updated_at"])

    for i in range(2):
        ro = _mk_response(profile, run, prompt, PLATFORM_OPENAI, i)
        ex = _mk_extraction(ro, True)
        update_prompt_aggregate_from_extraction(
            profile=profile, execution_run_id=run.id, response_snapshot=ro, extraction_snapshot=ex, prompt_category="comparison"
        )
    for i in range(2):
        rg = _mk_response(profile, run, prompt, PLATFORM_GEMINI, i)
        eg = _mk_extraction(rg, True)
        update_prompt_aggregate_from_extraction(
            profile=profile, execution_run_id=run.id, response_snapshot=rg, extraction_snapshot=eg, prompt_category="comparison"
        )

    calls: list[tuple[str, ...]] = []

    def fake_batch(prompt_set, business_profile, **kwargs):
        calls.append(tuple(kwargs.get("providers") or []))
        return {"executed": 0, "failed": 0, "results": []}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("accounts.tasks.run_aeo_phase4_scoring_task.delay", lambda _rid: None)
    monkeypatch.setattr("accounts.aeo.aeo_execution_utils.run_aeo_prompt_batch", fake_batch)
    try:
        run_aeo_phase2_confidence_task(
            run.id, [{"prompt": prompt, "type": "transactional", "weight": 1.0, "dynamic": True}]
        )
    finally:
        monkeypatch.undo()
    assert ("perplexity",) in calls


@pytest.mark.django_db
@override_settings(PERPLEXITY_API_KEY="pk-an")
def test_pass_count_analytics_includes_perplexity_when_rows_exist():
    user = User.objects.create_user(username="ps_p3", email="ps_p3@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")
    run = AEOExecutionRun.objects.create(profile=profile)
    AEOPromptExecutionAggregate.objects.create(
        profile=profile,
        execution_run=run,
        prompt_hash="hx",
        openai_pass_count=2,
        gemini_pass_count=2,
        perplexity_pass_count=2,
        openai_stability_status="stable",
        gemini_stability_status="stable",
        perplexity_stability_status="stable",
    )
    out = build_aeo_pass_count_analytics_context(execution_run_id=run.id, profile_id=profile.id)
    assert out["providers"]["perplexity"]["total"] == 1
    assert out["providers"]["perplexity"]["stable_at_2"] == 1
    assert out["perplexity_analytics_in_scope"] is True
    assert out["prompts_stable_at_2"] == 1

