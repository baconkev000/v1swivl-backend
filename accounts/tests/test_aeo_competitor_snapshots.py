from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.aeo.competitor_snapshots import calculate_competitor_visibility
from accounts.models import (
    AEOCompetitorSnapshot,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    BusinessProfile,
    TrackedCompetitor,
)

User = get_user_model()


def _make_slot(profile: BusinessProfile, prompt_hash: str, competitors_json: list[dict] | None) -> None:
    rsp = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text=prompt_hash,
        prompt_hash=prompt_hash,
        platform="openai",
    )
    if competitors_json is not None:
        AEOExtractionSnapshot.objects.create(
            response_snapshot=rsp,
            competitors_json=competitors_json,
        )


@pytest.mark.django_db
def test_competitor_visibility_dedupes_per_slot_excludes_self_and_sorts() -> None:
    user = User.objects.create_user(username="c1@example.com", email="c1@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Self",
        website_url="https://self.example.com",
    )
    _make_slot(
        profile,
        "p1",
        [
            {"name": "Alpha", "url": "https://alpha.com/a"},
            {"name": "Alpha Duplicate", "url": "https://alpha.com/b"},
            {"name": "Self", "url": "https://self.example.com"},
            {"name": "Beta", "url": "https://beta.com"},
        ],
    )
    _make_slot(profile, "p2", [{"name": "Beta", "url": "https://beta.com"}])
    _make_slot(profile, "p3", [{"name": "Charlie", "url": "https://charlie.com"}])
    _make_slot(profile, "p4", None)  # slot counted in denominator

    out = calculate_competitor_visibility(profile)
    assert out.total_slots == 4
    domains = [r.domain for r in out.rows]
    assert "self.example.com" not in domains
    assert domains == ["beta.com", "alpha.com", "charlie.com"]
    beta = next(r for r in out.rows if r.domain == "beta.com")
    alpha = next(r for r in out.rows if r.domain == "alpha.com")
    assert beta.appearances == 2
    assert beta.visibility_pct == 50.0
    assert alpha.appearances == 1  # deduped in p1 despite duplicate competitor rows
    assert alpha.visibility_pct == 25.0


@pytest.mark.django_db
def test_competitor_visibility_tie_break_is_alphabetical() -> None:
    user = User.objects.create_user(username="c1b@example.com", email="c1b@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Self",
        website_url="https://self.com",
    )
    domains = ["zeta.com", "alpha.com", "gamma.com", "beta.com", "delta.com", "epsilon.com"]
    for i, dom in enumerate(domains, start=1):
        _make_slot(profile, f"slot-{i}", [{"name": dom.split(".")[0], "url": f"https://{dom}"}])
    out = calculate_competitor_visibility(profile)
    assert [r.domain for r in out.rows] == sorted(domains)


@pytest.mark.django_db
def test_competitors_api_lazy_computes_snapshot_and_returns_sections(client) -> None:
    user = User.objects.create_user(username="c2@example.com", email="c2@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Self",
        website_url="https://self.com",
    )
    tracked = TrackedCompetitor.objects.create(name="Tracked Beta", domain="beta.com")
    profile.tracked_competitors.add(tracked)

    _make_slot(profile, "p1", [{"name": "Beta", "url": "https://beta.com"}])
    _make_slot(profile, "p2", [{"name": "Alpha", "url": "https://alpha.com"}])
    _make_slot(profile, "p3", [{"name": "Alpha", "url": "https://alpha.com"}])

    assert AEOCompetitorSnapshot.objects.count() == 0
    client.force_login(user)
    res = client.get("/api/aeo/competitors/")
    assert res.status_code == 200
    payload = res.json()
    assert payload["total_slots"] == 3
    assert payload["has_data"] is True
    assert len(payload["tracked_competitors"]) == 1
    assert payload["tracked_competitors"][0]["domain"] == "beta.com"
    assert payload["tracked_competitors"][0]["appearances"] == 1
    assert any(x["domain"] == "alpha.com" for x in payload["suggested_competitors"])
    assert AEOCompetitorSnapshot.objects.count() == 1


@pytest.mark.django_db
def test_backfill_competitor_snapshots_respects_force() -> None:
    u1 = User.objects.create_user(username="c3@example.com", email="c3@example.com", password="x")
    u2 = User.objects.create_user(username="c4@example.com", email="c4@example.com", password="x")
    p1 = BusinessProfile.objects.create(user=u1, is_main=True, business_name="A", website_url="https://a.com")
    p2 = BusinessProfile.objects.create(user=u2, is_main=True, business_name="B", website_url="https://b.com")
    _make_slot(p1, "p1", [{"name": "X", "url": "https://x.com"}])
    _make_slot(p2, "p2", [{"name": "Y", "url": "https://y.com"}])

    AEOCompetitorSnapshot.objects.create(
        profile=p1,
        platform_scope="all",
        window_start=None,
        window_end=None,
        total_slots=1,
        rows_json=[{"domain": "x.com", "display_name": "X", "appearances": 1, "visibility_pct": 100.0, "rank": 1, "last_seen_at": timezone.now().isoformat()}],
    )

    call_command("backfill_competitor_snapshots")
    assert AEOCompetitorSnapshot.objects.filter(profile=p1).count() == 1
    assert AEOCompetitorSnapshot.objects.filter(profile=p2).count() == 1

    before = AEOCompetitorSnapshot.objects.get(profile=p1).updated_at
    call_command("backfill_competitor_snapshots", "--force")
    after = AEOCompetitorSnapshot.objects.get(profile=p1).updated_at
    assert after >= before


@pytest.mark.django_db
def test_competitors_api_limits_suggested_to_top_five(client) -> None:
    user = User.objects.create_user(username="c5@example.com", email="c5@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Self",
        website_url="https://self.com",
    )
    for i, dom in enumerate(["a.com", "b.com", "c.com", "d.com", "e.com", "f.com"], start=1):
        _make_slot(profile, f"slot-{i}", [{"name": dom, "url": f"https://{dom}"}])
    client.force_login(user)
    res = client.get("/api/aeo/competitors/")
    assert res.status_code == 200
    payload = res.json()
    assert len(payload["suggested_competitors"]) == 5


@pytest.mark.django_db
def test_competitors_api_hides_suggested_when_more_than_three_tracked(client) -> None:
    user = User.objects.create_user(username="c6@example.com", email="c6@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Self",
        website_url="https://self.com",
    )
    tracked_domains = ["beta.com", "charlie.com", "delta.com", "echo.com"]
    for dom in tracked_domains:
        tc = TrackedCompetitor.objects.create(name=dom.split(".")[0].title(), domain=dom)
        profile.tracked_competitors.add(tc)
    for i, dom in enumerate(["alpha.com", "beta.com", "gamma.com", "theta.com"], start=1):
        _make_slot(profile, f"slot-x-{i}", [{"name": dom, "url": f"https://{dom}"}])

    client.force_login(user)
    res = client.get("/api/aeo/competitors/")
    assert res.status_code == 200
    payload = res.json()
    assert len(payload["tracked_competitors"]) == 4
    assert payload["suggested_competitors"] == []


@pytest.mark.django_db
def test_competitors_api_post_adds_tracked_competitor() -> None:
    user = User.objects.create_user(username="c-post@example.com", email="c-post@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Self",
        website_url="https://mysite.com",
    )
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/aeo/competitors/", {"domain": "rival.com", "name": "Rival Co"}, format="json")
    assert res.status_code == 200
    assert res.json().get("ok") is True
    profile.refresh_from_db()
    assert profile.tracked_competitors.filter(domain="rival.com").exists()


@pytest.mark.django_db
def test_competitors_api_post_rejects_own_domain() -> None:
    user = User.objects.create_user(username="c-own@example.com", email="c-own@example.com", password="x")
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Self",
        website_url="https://www.mysite.com/about",
    )
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.post("/api/aeo/competitors/", {"domain": "mysite.com", "name": "Me"}, format="json")
    assert res.status_code == 400


@pytest.mark.django_db
def test_competitors_api_delete_is_disabled() -> None:
    user = User.objects.create_user(username="c-del@example.com", email="c-del@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Self",
        website_url="https://self.com",
    )
    tc = TrackedCompetitor.objects.create(name="Beta", domain="beta.com")
    profile.tracked_competitors.add(tc)
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.delete("/api/aeo/competitors/", {"domain": "beta.com"}, format="json")
    assert res.status_code == 403
    profile.refresh_from_db()
    assert profile.tracked_competitors.count() == 1
