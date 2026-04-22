"""Unit tests for AEO LLM-only topic sanitization and brand-leak filtering."""

import json

import pytest
from django.contrib.auth import get_user_model

from accounts.aeo.aeo_utils import (
    AEOPromptBusinessInput,
    aeo_business_input_from_onboarding_payload,
    build_full_aeo_prompt_plan,
    build_openai_batch_user_content,
    prompt_contains_tracked_brand_leakage,
    sanitize_topic,
)
from accounts.models import BusinessProfile


def test_sanitize_multiword_business_phrase():
    assert (
        sanitize_topic("Acme Dental implants near me", "Acme Dental", "acme.com")
        == "implants near me"
    )


def test_sanitize_hyphenated_registrable_label_and_host():
    out = sanitize_topic("book mh-usa repair in Austin", "", "mh-usa.example.co.uk")
    assert "mh-usa" not in out.lower()
    assert "mh usa" not in out.lower()
    assert "example.co.uk" not in out.lower()
    assert "book" in out.lower() and "austin" in out.lower()


def test_sanitize_punctuation_tolerant_phrase():
    assert (
        sanitize_topic("Foo  Bar  widgets", "Foo Bar", "")
        == "widgets"
    )


def test_sanitize_empty_after_removal_falls_back():
    assert sanitize_topic("Acme Only", "Acme Only", "") == "this category"


def test_sanitize_single_token_brand_does_not_mangle_supercharger():
    """Boundary-safe: 'Super' must not strip the prefix of unrelated words."""
    assert sanitize_topic("supercharger repair", "Super", "unrelated.com") == "supercharger repair"


def test_sanitize_short_single_word_brand_skipped():
    """Very short brands are not stripped as single tokens (avoids 'us' in 'custom')."""
    assert sanitize_topic("custom metal fabrication", "US", "example.com") == "custom metal fabrication"


def test_prompt_leakage_detects_domain():
    assert prompt_contains_tracked_brand_leakage(
        "Is acme.example.com good for hosting?",
        "Other",
        "acme.example.com",
    )


def test_build_openai_batch_user_content_sanitizes_llm_paths_only():
    biz = AEOPromptBusinessInput(
        business_name="Gamma Lock LLC",
        website_domain="gammalock.example.com",
        industry="Gamma Lock LLC widgets, dental care",
        services=["Gamma Lock LLC repair", "dental care"],
        city="Austin, TX",
    )
    details = [{"keyword": "Gamma Lock LLC dental", "rank": 1}]
    raw = build_openai_batch_user_content(biz, [], 5, onboarding_topic_details=details)
    assert "Gamma Lock LLC dental" not in raw
    assert details[0]["keyword"] == "Gamma Lock LLC dental"
    _start = raw.index("Business context (JSON):\n") + len("Business context (JSON):\n")
    _end = raw.index("\n\nGenerate at most", _start)
    payload = json.loads(raw[_start:_end])
    assert "Gamma Lock LLC" not in " ".join(payload.get("services") or [])
    assert "Gamma Lock LLC" not in (payload.get("industry") or "")


def test_onboarding_business_input_uses_local_city_state_when_reach_is_local():
    out = aeo_business_input_from_onboarding_payload(
        business_name="Local Co",
        website_url="https://local.example.com",
        location="United States",
        language="English",
        selected_topics=["plumbing"],
        customer_reach="local",
        customer_reach_state="TX",
        customer_reach_city="Austin",
    )
    assert out.city == "Austin, TX"


def test_onboarding_local_appends_state_when_infer_is_city_only():
    out = aeo_business_input_from_onboarding_payload(
        business_name="Local Co",
        website_url="https://local.example.com",
        location="Los Angeles",
        language="English",
        selected_topics=["plumbing"],
        customer_reach="local",
        customer_reach_state="CA",
        customer_reach_city="",
    )
    assert out.city == "Los Angeles, CA"


def test_onboarding_local_does_not_duplicate_state_when_city_has_abbrev():
    out = aeo_business_input_from_onboarding_payload(
        business_name="Local Co",
        website_url="https://local.example.com",
        location="United States",
        language="English",
        selected_topics=["plumbing"],
        customer_reach="local",
        customer_reach_state="Texas",
        customer_reach_city="Austin, TX",
    )
    assert out.city == "Austin, TX"


def test_onboarding_business_input_keeps_existing_online_location_behavior():
    out = aeo_business_input_from_onboarding_payload(
        business_name="Online Co",
        website_url="https://online.example.com",
        location="Austin, TX",
        language="English",
        selected_topics=["software"],
        customer_reach="online",
        customer_reach_state="CA",
        customer_reach_city="San Diego",
    )
    assert out.city == "Austin, TX"


@pytest.mark.django_db
def test_build_full_aeo_prompt_plan_prompts_by_topic_keys_match_original_keywords(
    monkeypatch,
    settings,
):
    settings.AEO_TESTING_MODE = True

    user = get_user_model().objects.create_user(
        username="u-aeo-topic-keys",
        email="u-aeo-topic-keys@example.com",
        password="pw",
    )
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Tracked Biz Name",
        business_address="123 Main St, Austin, TX",
        industry="services",
        website_url="https://tracked.example.com",
    )
    selected = ["Gamma Lock LLC Services", "Delta plain keyword"]
    details = [{"keyword": selected[0], "vol": 1}, {"keyword": selected[1], "vol": 2}]
    bio = aeo_business_input_from_onboarding_payload(
        business_name="Tracked Biz Name",
        website_url="https://tracked.example.com",
        location="Austin, TX",
        language="English",
        selected_topics=selected,
    )

    seq = iter(range(100))

    def _mock_openai(*args, **kwargs):
        n = int(kwargs.get("max_additional", 0))
        return [
            {
                "prompt": f"openai prompt {next(seq)}",
                "type": "transactional",
                "weight": 1.0,
                "dynamic": True,
            }
            for _ in range(n)
        ]

    monkeypatch.setattr("accounts.aeo.aeo_utils.run_prompt_batch_via_openai", _mock_openai)

    plan = build_full_aeo_prompt_plan(
        profile,
        business_input=bio,
        onboarding_topic_details=details,
        target_combined_count=10,
    )
    assert list(plan["prompts_by_topic"].keys()) == selected
