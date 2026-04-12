"""Tiered OpenAI max_tokens for AEO prompt batch generation."""

import pytest

from accounts.aeo.aeo_utils import (
    aeo_business_input_from_profile,
    aeo_openai_max_output_tokens_for_target,
    build_full_aeo_prompt_plan,
)


def test_aeo_openai_max_output_tokens_starter_pro_advanced(settings):
    settings.AEO_OPENAI_MAX_TOKENS_STARTER = 111
    settings.AEO_OPENAI_MAX_TOKENS_PRO = 222
    settings.AEO_OPENAI_MAX_TOKENS_ADVANCED = 333

    assert aeo_openai_max_output_tokens_for_target(10) == 111
    assert aeo_openai_max_output_tokens_for_target(1) == 111
    assert aeo_openai_max_output_tokens_for_target(50) == 222
    assert aeo_openai_max_output_tokens_for_target(11) == 222
    assert aeo_openai_max_output_tokens_for_target(100) == 333
    assert aeo_openai_max_output_tokens_for_target(51) == 333


@pytest.mark.django_db
def test_build_full_aeo_prompt_plan_passes_scaled_max_tokens(monkeypatch, settings, django_user_model):
    settings.AEO_OPENAI_MAX_TOKENS_ADVANCED = 7777
    seen: list[int] = []

    class _Msg:
        content = "[]"

    class _Choice:
        message = _Msg()

    class _Comp:
        choices = [_Choice()]

    def fake_logged(*args, max_tokens=None, **kwargs):
        seen.append(int(max_tokens or 0))
        return _Comp()

    monkeypatch.setattr("accounts.aeo.aeo_utils.chat_completion_create_logged", fake_logged)

    from accounts.models import BusinessProfile

    user = django_user_model.objects.create_user(username="mx1", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        website_url="https://biz.example",
    )
    ctx = aeo_business_input_from_profile(profile)
    build_full_aeo_prompt_plan(profile, business_input=ctx, target_combined_count=100)

    assert len(seen) == 4
    assert all(m == 7777 for m in seen)
