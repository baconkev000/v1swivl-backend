"""Post-expansion Phase 1 backfill for new monitored prompts (Pro/Advanced)."""

import pytest
from django.contrib.auth import get_user_model

from accounts.models import BusinessProfile
from accounts.tasks import schedule_aeo_prompt_plan_expansion

User = get_user_model()


@pytest.mark.django_db
def test_expansion_schedules_backfill_when_new_prompt_strings(settings, monkeypatch):
    settings.AEO_TESTING_MODE = False
    monkeypatch.setattr("django.db.transaction.on_commit", lambda fn: fn())

    calls: list[tuple] = []

    def capture_delay(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "accounts.tasks.aeo_backfill_monitored_prompt_execution_task.delay",
        capture_delay,
    )

    def fake_build(profile, **kwargs):
        target = int(kwargs.get("target_combined_count") or 0)
        return {
            "combined": [{"prompt": f"p{i}"} for i in range(target)],
            "meta": {"openai_status": "ok", "combined_shortfall": 0},
        }

    monkeypatch.setattr("accounts.aeo.aeo_utils.build_full_aeo_prompt_plan", fake_build)

    user = User.objects.create_user(username="bf1@example.com", email="bf1@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="B",
        plan=BusinessProfile.PLAN_PRO,
        website_url="https://example.com",
        stripe_subscription_status="active",
        selected_aeo_prompts=[f"keep{i}" for i in range(10)],
    )

    schedule_aeo_prompt_plan_expansion.run(profile.id)
    assert len(calls) == 1
    assert calls[0][0][0] == profile.id
    assert calls[0][1]["before_prompt_count"] == 10
    assert calls[0][1]["after_prompt_count"] == 75


@pytest.mark.django_db
def test_expansion_no_backfill_when_already_at_cap(settings, monkeypatch):
    settings.AEO_TESTING_MODE = False
    monkeypatch.setattr("django.db.transaction.on_commit", lambda fn: fn())

    calls: list[tuple] = []

    monkeypatch.setattr(
        "accounts.tasks.aeo_backfill_monitored_prompt_execution_task.delay",
        lambda *a, **k: calls.append(1),
    )

    user = User.objects.create_user(username="bf2@example.com", email="bf2@example.com", password="x")
    prompts = [f"p{i}" for i in range(75)]
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="B",
        plan=BusinessProfile.PLAN_PRO,
        website_url="https://example.com",
        stripe_subscription_status="active",
        selected_aeo_prompts=prompts,
    )

    schedule_aeo_prompt_plan_expansion.run(profile.id)
    assert calls == []


@pytest.mark.django_db
def test_backfill_enqueues_phase1_with_missing_prompts_only(settings, monkeypatch):
    settings.AEO_TESTING_MODE = False
    from accounts.models import AEOResponseSnapshot, AEOExtractionSnapshot
    from accounts.tasks import aeo_backfill_monitored_prompt_execution_task, run_aeo_phase1_execution_task

    phase1_calls: list[tuple] = []

    def capture_phase1(*args, **kwargs):
        phase1_calls.append((args, kwargs))

    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", capture_phase1)

    user = User.objects.create_user(username="bf3@example.com", email="bf3@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="B",
        plan=BusinessProfile.PLAN_PRO,
        website_url="https://example.com",
        selected_aeo_prompts=["has_all_three", "needs_work"],
    )

    def add_full_stack(prompt_text: str) -> None:
        for plat in ("openai", "gemini", "perplexity"):
            snap = AEOResponseSnapshot.objects.create(
                profile=profile,
                prompt_text=prompt_text,
                prompt_hash=f"h-{plat}-{prompt_text[:8]}",
                raw_response="x",
                platform=plat,
            )
            AEOExtractionSnapshot.objects.create(
                response_snapshot=snap,
                brand_mentioned=False,
            )

    add_full_stack("has_all_three")

    aeo_backfill_monitored_prompt_execution_task.run(
        profile.id,
        before_prompt_count=2,
        after_prompt_count=2,
    )

    assert len(phase1_calls) == 1
    kw = phase1_calls[0][1]
    assert kw.get("force_refresh") is True
    payload = kw.get("prompt_set") or phase1_calls[0][0][1]
    assert payload is not None
    texts = [str(p.get("prompt") or "") for p in payload]
    assert texts == ["needs_work"]


@pytest.mark.django_db
def test_backfill_force_bypasses_local_gates(settings, monkeypatch):
    settings.AEO_TESTING_MODE = True

    from accounts.tasks import aeo_backfill_monitored_prompt_execution_task

    phase1_calls: list[tuple] = []

    def capture_phase1(*args, **kwargs):
        phase1_calls.append((args, kwargs))

    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", capture_phase1)

    user = User.objects.create_user(username="bf4@example.com", email="bf4@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="B4",
        plan=BusinessProfile.PLAN_STARTER,
        website_url="https://example.com",
        selected_aeo_prompts=["needs_work"],
    )

    aeo_backfill_monitored_prompt_execution_task.run(profile.id, force=True)
    assert len(phase1_calls) == 1
