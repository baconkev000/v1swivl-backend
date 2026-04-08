import pytest
from datetime import timedelta
from django.contrib.auth import get_user_model
from django.utils import timezone

from accounts.models import AEOExecutionRun, BusinessProfile, OnboardingOnPageCrawl
from accounts.tasks import onboarding_prompt_generation_task

User = get_user_model()


class _BizInput:
    def as_dict(self):
        return {}


@pytest.mark.django_db
def test_onboarding_prompt_generation_task_empty_plan_marks_failed(monkeypatch):
    user = User.objects.create_user(username="opt1", email="opt1@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        website_url="https://example.com",
        business_address="US",
    )
    crawl = OnboardingOnPageCrawl.objects.create(
        user=user,
        business_profile=profile,
        domain="example.com",
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        ranked_keywords=[{"keyword": "best dentist", "rank": 3}],
        context={"business_name": "Biz", "location": "US"},
    )
    monkeypatch.setattr("accounts.aeo.aeo_utils.aeo_business_input_from_onboarding_payload", lambda **kwargs: _BizInput())
    monkeypatch.setattr(
        "accounts.aeo.aeo_utils.build_full_aeo_prompt_plan",
        lambda *args, **kwargs: {"combined": [], "meta": {"openai_status": "failed_empty"}},
    )
    onboarding_prompt_generation_task(crawl.id)
    crawl.refresh_from_db()
    assert crawl.prompt_plan_status == OnboardingOnPageCrawl.PROMPT_PLAN_FAILED
    assert "empty_prompt_plan" in (crawl.prompt_plan_error or "")


@pytest.mark.django_db
def test_onboarding_prompt_generation_task_valid_plan_saves_prompts_and_enqueues_phase1(monkeypatch):
    user = User.objects.create_user(username="opt2", email="opt2@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        website_url="https://example.com",
        business_address="US",
    )
    crawl = OnboardingOnPageCrawl.objects.create(
        user=user,
        business_profile=profile,
        domain="example.com",
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        ranked_keywords=[{"keyword": "best dentist", "rank": 3}],
        context={"business_name": "Biz", "location": "US"},
    )
    monkeypatch.setattr("accounts.aeo.aeo_utils.aeo_business_input_from_onboarding_payload", lambda **kwargs: _BizInput())
    combined = [
        {"prompt": "prompt a", "type": "transactional", "weight": 1.0, "dynamic": True},
        {"prompt": "prompt b", "type": "trust", "weight": 1.0, "dynamic": True},
    ]
    monkeypatch.setattr(
        "accounts.aeo.aeo_utils.build_full_aeo_prompt_plan",
        lambda *args, **kwargs: {"combined": combined, "meta": {"openai_status": "ok"}},
    )
    calls = []
    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", lambda rid, payload: calls.append((rid, payload)))
    onboarding_prompt_generation_task(crawl.id)
    crawl.refresh_from_db()
    profile.refresh_from_db()
    assert crawl.prompt_plan_status == OnboardingOnPageCrawl.PROMPT_PLAN_COMPLETED
    assert crawl.prompt_plan_prompt_count == 2
    assert profile.selected_aeo_prompts == ["prompt a", "prompt b"]
    run = AEOExecutionRun.objects.filter(profile=profile).order_by("-id").first()
    assert run is not None
    assert run.status == AEOExecutionRun.STATUS_PENDING
    assert len(calls) == 1
    assert calls[0][0] == run.id


@pytest.mark.django_db
def test_onboarding_prompt_generation_task_marks_stale_inflight_failed_then_enqueues(monkeypatch):
    user = User.objects.create_user(username="opt3", email="opt3@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        website_url="https://example.com",
        business_address="US",
    )
    stale = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_PENDING,
        prompt_count_requested=1,
    )
    AEOExecutionRun.objects.filter(id=stale.id).update(created_at=timezone.now() - timedelta(minutes=30))
    crawl = OnboardingOnPageCrawl.objects.create(
        user=user,
        business_profile=profile,
        domain="example.com",
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        ranked_keywords=[{"keyword": "best dentist", "rank": 3}],
        context={"business_name": "Biz", "location": "US"},
    )
    monkeypatch.setattr("accounts.aeo.aeo_utils.aeo_business_input_from_onboarding_payload", lambda **kwargs: _BizInput())
    monkeypatch.setattr(
        "accounts.aeo.aeo_utils.build_full_aeo_prompt_plan",
        lambda *args, **kwargs: {
            "combined": [{"prompt": "prompt a", "type": "transactional", "weight": 1.0, "dynamic": True}],
            "meta": {"openai_status": "ok"},
        },
    )
    calls = []
    monkeypatch.setattr("accounts.tasks.run_aeo_phase1_execution_task.delay", lambda rid, payload: calls.append((rid, payload)))
    onboarding_prompt_generation_task(crawl.id)
    stale.refresh_from_db()
    assert stale.status == AEOExecutionRun.STATUS_FAILED
    assert "stale_inflight" in (stale.error_message or "")
    assert len(calls) == 1

