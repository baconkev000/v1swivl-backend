import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import BusinessProfile, SEOOverviewSnapshot


User = get_user_model()


@pytest.mark.django_db
def test_seo_refresh_does_not_call_aeo_helper(monkeypatch):
    user = User.objects.create_user(username="u-dec-seo", email="u-dec-seo@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, website_url="https://whitepinedentalcare.com")
    SEOOverviewSnapshot.objects.create(
        user=user,
        business_profile=profile,
        period_start=__import__("datetime").date.today().replace(day=1),
        cached_domain="whitepinedentalcare.com",
    )

    monkeypatch.setattr(
        "accounts.serializers.get_aeo_content_readiness_for_site",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("AEO helper should not be called")),
    )
    monkeypatch.setattr("accounts.dataforseo_utils.enrich_with_gap_keywords", lambda **kwargs: None)
    monkeypatch.setattr("accounts.dataforseo_utils.enrich_with_llm_keywords", lambda **kwargs: None)
    monkeypatch.setattr(
        "accounts.dataforseo_utils.enrich_keyword_ranks_from_labs",
        lambda **kwargs: {"total": 0, "non_null_after": 0, "filled_from_ranked": 0, "filled_from_gap": 0},
    )
    monkeypatch.setattr("accounts.tasks.generate_snapshot_next_steps_task.delay", lambda *args, **kwargs: None)
    monkeypatch.setattr("accounts.tasks.generate_keyword_action_suggestions_task.delay", lambda *args, **kwargs: None)

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post("/api/seo/refresh-snapshot/")
    assert resp.status_code == 200
    body = resp.json()
    assert "aeo_score" not in body
    assert "question_coverage_score" not in body
    assert "seo_score" in body


@pytest.mark.django_db
def test_aeo_refresh_does_not_call_seo_helper(monkeypatch):
    user = User.objects.create_user(username="u-dec-aeo", email="u-dec-aeo@example.com", password="pw")
    BusinessProfile.objects.create(user=user, is_main=True, website_url="https://whitepinedentalcare.com")

    monkeypatch.setattr(
        "accounts.serializers.get_or_refresh_seo_score_for_user",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("SEO helper should not be called")),
    )
    monkeypatch.setattr(
        "accounts.serializers.get_aeo_content_readiness_for_site",
        lambda **kwargs: {
            "question_coverage_score": 60,
            "questions_found": ["what is invisalign"],
            "questions_missing": [],
            "faq_readiness_score": 50,
            "faq_blocks_found": 1,
            "faq_schema_present": True,
            "snippet_readiness_score": 40,
            "answer_blocks_found": 1,
            "aeo_status": "ready",
            "aeo_last_computed_at": "2026-03-20T00:00:00+00:00",
        },
    )

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post("/api/aeo/refresh-snapshot/")
    assert resp.status_code == 200
    body = resp.json()
    assert "seo_score" not in body
    assert "search_visibility_percent" not in body
    assert "aeo_score" in body


@pytest.mark.django_db
def test_scoped_profile_payloads():
    user = User.objects.create_user(username="u-dec-read", email="u-dec-read@example.com", password="pw")
    BusinessProfile.objects.create(user=user, is_main=True, website_url="https://whitepinedentalcare.com")
    client = APIClient()
    client.force_authenticate(user=user)

    seo_resp = client.get("/api/seo/profile/")
    assert seo_resp.status_code == 200
    seo_body = seo_resp.json()
    assert "seo_score" in seo_body
    assert "aeo_score" not in seo_body

    aeo_resp = client.get("/api/aeo/profile/")
    assert aeo_resp.status_code == 200
    aeo_body = aeo_resp.json()
    assert "aeo_score" in aeo_body
    assert "seo_score" not in aeo_body
