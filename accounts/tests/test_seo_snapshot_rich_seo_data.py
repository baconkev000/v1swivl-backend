"""Rich seo_data from SEOOverviewSnapshot for async next-steps tasks."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone as django_tz

from accounts.models import BusinessProfile, SEOOverviewSnapshot
from accounts.tasks import (
    _seo_snapshot_corpus_newer_than_next_steps,
    seo_data_dict_from_seo_overview_snapshot,
)

User = get_user_model()


@pytest.mark.django_db
def test_seo_data_dict_includes_snapshot_metrics_and_context():
    user = User.objects.create_user(username="sd1", email="sd1@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        website_url="https://example.com",
    )
    snap = SEOOverviewSnapshot.objects.create(
        user=user,
        business_profile=profile,
        period_start=date(2026, 2, 1),
        cached_domain="example.com",
        cached_location_mode="organic",
        cached_location_code=0,
        cached_location_label="US",
        local_verification_applied=True,
        local_verified_keyword_count=2,
        top_keywords=[{"keyword": "a", "search_volume": 100, "rank": 5}],
        total_search_volume=500,
        search_performance_score=42,
        seo_structured_issues=[{"issue_id": "x", "evidence": {}}],
    )
    d = seo_data_dict_from_seo_overview_snapshot(snap)
    assert d["search_performance_score"] == 42
    assert d["cached_domain"] == "example.com"
    assert len(d["top_keywords"]) == 1
    assert d["seo_structured_issues"][0]["issue_id"] == "x"
    assert isinstance(d["snapshot_context_for_rewrite"], dict)
    assert d["snapshot_context_for_rewrite"]["snapshot_id"] == snap.id


@pytest.mark.django_db(transaction=True)
def test_generate_snapshot_next_steps_uses_rich_seo_data(monkeypatch):
    captured: list[dict] = []

    def fake_generate(seo_data, *, snapshot=None):
        captured.append(dict(seo_data))
        return []

    monkeypatch.setattr(
        "accounts.openai_utils.generate_seo_next_steps", fake_generate
    )

    from accounts.tasks import generate_snapshot_next_steps_task

    user = User.objects.create_user(username="sd2", email="sd2@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        website_url="https://co.example",
    )
    snap = SEOOverviewSnapshot.objects.create(
        user=user,
        business_profile=profile,
        period_start=date(2026, 3, 1),
        cached_domain="co.example",
        cached_location_mode="organic",
        cached_location_code=0,
        top_keywords=[{"keyword": "seed", "search_volume": 900, "rank": None}],
        search_performance_score=55,
    )
    generate_snapshot_next_steps_task.run(snap.id)
    assert len(captured) == 1
    assert "snapshot_context_for_rewrite" in captured[0]
    assert captured[0]["search_performance_score"] == 55
    assert captured[0]["top_keywords"][0]["keyword"] == "seed"


def test_ttl_bypass_when_keywords_enriched_after_next_steps():
    class O:
        pass

    o = O()
    now = django_tz.now()
    o.seo_next_steps_refreshed_at = now - timedelta(days=1)
    o.keywords_enriched_at = now
    o.refreshed_at = None
    assert _seo_snapshot_corpus_newer_than_next_steps(o) is True
