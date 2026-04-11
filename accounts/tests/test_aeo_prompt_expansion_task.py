import pytest
from django.contrib.auth import get_user_model

from accounts.models import BusinessProfile
from accounts.tasks import schedule_aeo_prompt_plan_expansion

User = get_user_model()


@pytest.mark.django_db
def test_expansion_merges_delta(monkeypatch, settings):
    settings.AEO_TESTING_MODE = False

    def fake_build(profile, **kwargs):
        target = int(kwargs.get("target_combined_count") or 0)
        return {
            "combined": [{"prompt": f"p{i}"} for i in range(target)],
            "meta": {"openai_status": "ok", "combined_shortfall": 0},
        }

    monkeypatch.setattr("accounts.aeo.aeo_utils.build_full_aeo_prompt_plan", fake_build)

    user = User.objects.create_user(username="exp@example.com", email="exp@example.com", password="x")
    existing = [f"keep{i}" for i in range(10)]
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="E",
        plan=BusinessProfile.PLAN_PRO,
        website_url="https://example.com",
        selected_aeo_prompts=list(existing),
    )

    schedule_aeo_prompt_plan_expansion.run(profile.id)
    profile.refresh_from_db()
    assert len(profile.selected_aeo_prompts) == 50
    assert profile.aeo_prompt_expansion_status == BusinessProfile.AEO_PROMPT_EXPANSION_COMPLETE
    for k in range(10):
        assert f"keep{k}" in profile.selected_aeo_prompts


@pytest.mark.django_db
def test_expansion_skipped_in_testing_mode(settings):
    settings.AEO_TESTING_MODE = True
    user = User.objects.create_user(username="ex2@example.com", email="ex2@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="E2",
        plan=BusinessProfile.PLAN_PRO,
        selected_aeo_prompts=["a"],
    )
    schedule_aeo_prompt_plan_expansion.run(profile.id)
    profile.refresh_from_db()
    assert len(profile.selected_aeo_prompts) == 1
