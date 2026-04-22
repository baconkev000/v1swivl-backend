import pytest
from datetime import timedelta
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.models import BusinessProfile, SEOOverviewSnapshot

User = get_user_model()


@pytest.mark.django_db
def test_seo_overview_uses_snapshot_cache_for_up_to_7_days(monkeypatch):
    user = User.objects.create_user(
        username="seo_cache@example.com",
        email="seo_cache@example.com",
        password="pw",
    )
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Cache Co",
        website_url="https://cacheco.example.com",
    )
    today = timezone.now().date()
    start_current = today.replace(day=1)
    snap = SEOOverviewSnapshot.objects.create(
        user=user,
        business_profile=profile,
        period_start=start_current,
        cached_location_mode="organic",
        cached_location_code=0,
        organic_visitors=1234,
        prev_organic_visitors=1200,
        keywords_ranking=88,
        top3_positions=17,
    )
    fresh_ts = timezone.now() - timedelta(days=3)
    SEOOverviewSnapshot.objects.filter(pk=snap.pk).update(last_fetched_at=fresh_ts)

    def _should_not_call(*_a, **_k):
        raise AssertionError("DataForSEO live endpoint should not be called on fresh snapshot.")

    monkeypatch.setattr("accounts.views.get_ranked_keywords_visibility", _should_not_call)

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/seo/overview/")
    assert res.status_code == 200
    body = res.json()
    assert body["organic_visitors"] == 1234
    assert body["keywords_ranking"] == 88
    assert body["top3_positions"] == 17
