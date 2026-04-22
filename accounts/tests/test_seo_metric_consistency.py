from datetime import date

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.dataforseo_utils import recompute_snapshot_metrics_from_keywords
from accounts.models import BusinessProfile, SEOOverviewSnapshot


User = get_user_model()


@pytest.mark.django_db
def test_recompute_metrics_ranked_keywords_visibility_above_zero():
    metrics = recompute_snapshot_metrics_from_keywords(
        top_keywords=[
            {"keyword": "dentist near me", "search_volume": 1000, "rank": 3},
            {"keyword": "emergency dentist", "search_volume": 500, "rank": 7},
        ],
        domain="whitepinedentalcare.com",
        location_code=2840,
        language_code="en",
    )
    assert metrics["total_search_volume"] == 1500
    assert metrics["estimated_traffic"] > 0
    assert metrics["search_visibility_percent"] > 0
    assert metrics["missed_searches_monthly"] < 1500


@pytest.mark.django_db
def test_recompute_metrics_all_rank_null_visibility_zero():
    metrics = recompute_snapshot_metrics_from_keywords(
        top_keywords=[
            {"keyword": "dentist near me", "search_volume": 1000, "rank": None},
            {"keyword": "emergency dentist", "search_volume": 500, "rank": None},
        ],
        domain="whitepinedentalcare.com",
        location_code=2840,
        language_code="en",
    )
    assert metrics["total_search_volume"] == 1500
    assert metrics["estimated_traffic"] == 0
    assert metrics["search_visibility_percent"] == 0
    assert metrics["missed_searches_monthly"] == 1500


@pytest.mark.django_db
def test_refresh_snapshot_updates_metrics_consistently_with_ranks(monkeypatch):
    user = User.objects.create_user(username="seo-u", email="seo-u@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user, is_main=True, website_url="https://whitepinedentalcare.com"
    )
    snapshot = SEOOverviewSnapshot.objects.create(
        user=user,
        business_profile=profile,
        period_start=date.today().replace(day=1),
        cached_domain="whitepinedentalcare.com",
        top_keywords=[
            {"keyword": "dentist near me", "search_volume": 1000, "rank": None},
            {"keyword": "emergency dentist", "search_volume": 500, "rank": None},
        ],
        organic_visitors=0,
        total_search_volume=1500,
        search_visibility_percent=0,
        missed_searches_monthly=1500,
    )

    monkeypatch.setattr("accounts.dataforseo_utils.enrich_with_gap_keywords", lambda **kwargs: None)
    monkeypatch.setattr("accounts.dataforseo_utils.enrich_with_llm_keywords", lambda **kwargs: None)

    def fake_enrich_keyword_ranks_from_labs(**kwargs):
        top_keywords = kwargs["top_keywords"]
        top_keywords[0]["rank"] = 4
        top_keywords[1]["rank"] = 9
        return {"total": 2, "non_null_after": 2, "filled_from_ranked": 2, "filled_from_gap": 0}

    monkeypatch.setattr("accounts.dataforseo_utils.enrich_keyword_ranks_from_labs", fake_enrich_keyword_ranks_from_labs)

    class _NoopTask:
        @staticmethod
        def delay(_snapshot_id):
            return None

    monkeypatch.setattr("accounts.tasks.generate_snapshot_next_steps_task", _NoopTask)
    monkeypatch.setattr("accounts.tasks.generate_keyword_action_suggestions_task", _NoopTask)

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post("/api/seo/refresh-snapshot/")
    assert resp.status_code == 200

    snapshot.refresh_from_db()
    ranked = [k for k in (snapshot.top_keywords or []) if (k.get("rank") or 0) > 0]
    assert len(ranked) > 0
    assert snapshot.search_visibility_percent > 0
    assert snapshot.organic_visitors > 0
    assert snapshot.missed_searches_monthly < snapshot.total_search_volume
    data = resp.json()
    assert (data.get("search_visibility_percent") or 0) > 0


@pytest.mark.django_db
def test_no_ranked_keywords_with_zero_appeared_only_when_no_volume():
    metrics = recompute_snapshot_metrics_from_keywords(
        top_keywords=[{"keyword": "x", "search_volume": 0, "rank": 3}],
        domain="whitepinedentalcare.com",
        location_code=2840,
        language_code="en",
    )
    assert metrics["keywords_ranking"] > 0
    assert metrics["total_search_volume"] == 0
    assert metrics["estimated_traffic"] == 0
    assert metrics["search_visibility_percent"] == 0
