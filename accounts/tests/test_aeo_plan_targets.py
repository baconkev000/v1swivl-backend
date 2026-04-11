import pytest
from django.contrib.auth import get_user_model

from accounts.aeo.aeo_plan_targets import (
    AEO_PLAN_CAP_ADVANCED,
    AEO_PLAN_CAP_PRO,
    AEO_PLAN_CAP_STARTER,
    aeo_effective_monitored_target_for_profile,
    aeo_http_call_bounds_for_monitoring,
    aeo_monitored_prompt_cap_for_plan_slug,
    aeo_onboarding_complete_min_prompts,
)
from accounts.models import BusinessProfile

User = get_user_model()


@pytest.mark.django_db
def test_cap_by_plan_slug():
    assert aeo_monitored_prompt_cap_for_plan_slug("") == AEO_PLAN_CAP_STARTER
    assert aeo_monitored_prompt_cap_for_plan_slug("starter") == AEO_PLAN_CAP_STARTER
    assert aeo_monitored_prompt_cap_for_plan_slug("pro") == AEO_PLAN_CAP_PRO
    assert aeo_monitored_prompt_cap_for_plan_slug("advanced") == AEO_PLAN_CAP_ADVANCED


@pytest.mark.django_db
def test_http_bounds_hint():
    b = aeo_http_call_bounds_for_monitoring(10)
    assert b["http_calls_approx_lo"] == 40
    assert b["http_calls_approx_hi"] == 60


@pytest.mark.django_db
def test_onboarding_min_respects_cap_when_testing(monkeypatch, settings):
    settings.AEO_TESTING_MODE = True
    settings.AEO_TEST_PROMPT_COUNT = 3
    user = User.objects.create_user(username="t1@example.com", email="t1@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="A", plan=BusinessProfile.PLAN_PRO)
    assert aeo_effective_monitored_target_for_profile(profile) == 3
    assert aeo_onboarding_complete_min_prompts(profile) == 3


@pytest.mark.django_db
def test_onboarding_min_production_baseline(settings):
    settings.AEO_TESTING_MODE = False
    user = User.objects.create_user(username="t2@example.com", email="t2@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="B", plan=BusinessProfile.PLAN_PRO)
    assert aeo_effective_monitored_target_for_profile(profile) == AEO_PLAN_CAP_PRO
    assert aeo_onboarding_complete_min_prompts(profile) == AEO_PLAN_CAP_STARTER
