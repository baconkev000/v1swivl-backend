import pytest
from django.contrib.auth import get_user_model
from django.db import transaction as django_transaction
from rest_framework.test import APIClient

from accounts.models import BusinessProfile, OnboardingOnPageCrawl


User = get_user_model()


@pytest.mark.django_db
def test_onpage_crawl_409_when_domain_on_other_users_profile():
    owner = User.objects.create_user(
        username="owner-dom",
        email="owner-dom@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(
        user=owner,
        is_main=True,
        website_url="https://example.com",
    )

    other = User.objects.create_user(
        username="other-dom",
        email="other-dom@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(user=other, is_main=True)

    client = APIClient()
    client.force_authenticate(user=other)
    resp = client.post(
        "/api/onboarding/onpage-crawl/",
        {
            "website_url": "https://www.example.com/about",
            "business_name": "Other Co",
            "location": "US",
        },
        format="json",
    )
    assert resp.status_code == 409
    assert "another account" in (resp.data.get("error") or "").lower()


@pytest.mark.django_db
def test_onpage_crawl_reuses_completed_crawl_with_keywords(monkeypatch):
    calls = []
    backfill_calls = []

    def fake_delay(_cid):
        calls.append(_cid)

    def fake_backfill(cid):
        backfill_calls.append(cid)

    monkeypatch.setattr(django_transaction, "on_commit", lambda cb: cb())
    monkeypatch.setattr(
        "accounts.tasks.onboarding_onpage_crawl_task.delay",
        fake_delay,
    )
    monkeypatch.setattr(
        "accounts.tasks.onboarding_review_topics_backfill_task.delay",
        fake_backfill,
    )

    user = User.objects.create_user(
        username="reuse-u",
        email="reuse-u@example.com",
        password="pw",
    )
    profile = BusinessProfile.objects.create(user=user, is_main=True)
    OnboardingOnPageCrawl.objects.create(
        user=user,
        business_profile=profile,
        domain="example.com",
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        ranked_keywords=[
            {"keyword": "widgets", "search_volume": 100, "rank": 5},
        ],
    )

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/onboarding/onpage-crawl/",
        {
            "website_url": "https://example.com",
            "business_name": "Co",
            "location": "US",
        },
        format="json",
    )
    assert resp.status_code == 200
    assert resp.data.get("reused") is True
    assert len(resp.data.get("ranked_keywords") or []) == 1
    assert resp.data.get("review_topics") == []
    assert calls == []
    assert len(backfill_calls) == 1
    assert OnboardingOnPageCrawl.objects.filter(user=user).count() == 1


@pytest.mark.django_db
def test_onpage_crawl_enqueues_when_no_reusable_rows(monkeypatch):
    calls = []

    def fake_delay(cid):
        calls.append(cid)

    monkeypatch.setattr(django_transaction, "on_commit", lambda cb: cb())
    monkeypatch.setattr(
        "accounts.tasks.onboarding_onpage_crawl_task.delay",
        fake_delay,
    )

    user = User.objects.create_user(
        username="fresh-u",
        email="fresh-u@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(user=user, is_main=True)

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/onboarding/onpage-crawl/",
        {
            "website_url": "https://newdomain.test",
            "business_name": "Co",
            "location": "US",
            "customer_reach": "local",
            "customer_reach_state": "CA",
            "customer_reach_city": "Los Angeles",
        },
        format="json",
    )
    assert resp.status_code == 200
    assert resp.data.get("reused") is not True
    assert len(calls) == 1
    assert OnboardingOnPageCrawl.objects.filter(user=user).count() == 1
    crawl = OnboardingOnPageCrawl.objects.get(user=user)
    assert crawl.context.get("customer_reach") == "local"
    assert crawl.context.get("customer_reach_state") == "CA"
    assert crawl.context.get("customer_reach_city") == "Los Angeles"


@pytest.mark.django_db
def test_onpage_crawl_requires_state_for_local_customer_reach():
    user = User.objects.create_user(
        username="reach-required",
        email="reach-required@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(user=user, is_main=True)
    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/onboarding/onpage-crawl/",
        {
            "website_url": "https://newdomain.test",
            "business_name": "Co",
            "location": "US",
            "customer_reach": "local",
            "customer_reach_state": "",
        },
        format="json",
    )
    assert resp.status_code == 400
    assert "customer_reach_state" in str(resp.data.get("error") or "")


@pytest.mark.django_db
def test_onboarding_crawl_latest_filters_by_domain():
    user = User.objects.create_user(
        username="poll-u",
        email="poll-u@example.com",
        password="pw",
    )
    profile = BusinessProfile.objects.create(user=user, is_main=True)
    OnboardingOnPageCrawl.objects.create(
        user=user,
        business_profile=profile,
        domain="b.com",
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        ranked_keywords=[{"keyword": "b", "search_volume": 1, "rank": 1}],
    )
    OnboardingOnPageCrawl.objects.create(
        user=user,
        business_profile=profile,
        domain="a.com",
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        ranked_keywords=[{"keyword": "a", "search_volume": 2, "rank": 2}],
    )

    client = APIClient()
    client.force_authenticate(user=user)
    r_new = client.get("/api/onboarding/crawl/latest/")
    assert r_new.status_code == 200
    # Latest by created_at: a.com row is created after b.com
    assert r_new.data["domain"] == "a.com"

    r_a = client.get("/api/onboarding/crawl/latest/", {"domain": "a.com"})
    assert r_a.status_code == 200
    assert r_a.data["domain"] == "a.com"
    assert (r_a.data["ranked_keywords"] or [{}])[0].get("keyword") == "a"
    assert r_a.data.get("review_topics") == []
    assert r_a.data.get("ranked_keywords_pending") is False


@pytest.mark.django_db
def test_onboarding_crawl_latest_ranked_keywords_pending():
    user = User.objects.create_user(
        username="pending-u",
        email="pending-u@example.com",
        password="pw",
    )
    profile = BusinessProfile.objects.create(user=user, is_main=True)
    OnboardingOnPageCrawl.objects.create(
        user=user,
        business_profile=profile,
        domain="pending.test",
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        ranked_keywords=[],
        ranked_keywords_fetch_status=OnboardingOnPageCrawl.RANKED_FETCH_PENDING,
        review_topics=[{"topic": "Services", "category": "general"}],
    )

    client = APIClient()
    client.force_authenticate(user=user)
    r = client.get("/api/onboarding/crawl/latest/", {"domain": "pending.test"})
    assert r.status_code == 200
    assert r.data.get("ranked_keywords_pending") is True
    assert len(r.data.get("review_topics") or []) == 1


@pytest.mark.django_db
def test_onpage_crawl_reuses_completed_crawl_from_sibling_profile_for_same_domain(monkeypatch):
    monkeypatch.setattr(django_transaction, "on_commit", lambda cb: cb())
    calls = []
    monkeypatch.setattr("accounts.tasks.onboarding_onpage_crawl_task.delay", lambda cid: calls.append(cid))

    user = User.objects.create_user(
        username="reuse-cross-profile",
        email="reuse-cross-profile@example.com",
        password="pw",
    )
    source_profile = BusinessProfile.objects.create(user=user, is_main=True)
    target_profile = BusinessProfile.objects.create(user=user, is_main=False)
    OnboardingOnPageCrawl.objects.create(
        user=user,
        business_profile=source_profile,
        domain="cachedomain.test",
        status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        ranked_keywords=[{"keyword": "cached", "search_volume": 50, "rank": 3}],
        review_topics=[{"topic": "Cached Topic", "category": "service"}],
    )

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.post(
        "/api/onboarding/onpage-crawl/",
        {
            "profile_id": target_profile.id,
            "website_url": "https://cachedomain.test",
            "business_name": "Target Co",
            "location": "US",
        },
        format="json",
    )
    assert resp.status_code == 200
    assert resp.data.get("reused") is True
    assert calls == []
    cloned = OnboardingOnPageCrawl.objects.filter(user=user, business_profile=target_profile).first()
    assert cloned is not None
    assert (cloned.ranked_keywords or [{}])[0].get("keyword") == "cached"
    assert cloned.ranked_keywords_fetch_status == OnboardingOnPageCrawl.RANKED_FETCH_COMPLETE
