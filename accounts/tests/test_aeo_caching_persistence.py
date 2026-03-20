import pytest
from datetime import timedelta
from django.contrib.auth import get_user_model
from django.utils import timezone

from accounts.models import AEOOverviewSnapshot, BusinessProfile
from accounts.serializers import BusinessProfileSerializer


User = get_user_model()


@pytest.mark.django_db
def test_aeo_serializer_cache_isolated_by_profile_domain_location(monkeypatch):
    user = User.objects.create_user(username="u1", email="u1@example.com", password="pw")
    p1 = BusinessProfile.objects.create(user=user, website_url="https://alpha.example.com", industry="dental")
    p2 = BusinessProfile.objects.create(user=user, website_url="https://beta.example.com", industry="dental")

    calls = {"count": 0}

    def fake_get_aeo_content_readiness_for_site(**kwargs):
        calls["count"] += 1
        domain = kwargs.get("target_domain") or ""
        if "alpha.example.com" in domain:
            return {
                "question_coverage_score": 90,
                "questions_found": ["how to choose dentist"],
                "questions_missing": [],
                "faq_readiness_score": 80,
                "faq_blocks_found": 3,
                "faq_schema_present": True,
                "snippet_readiness_score": 70,
                "answer_blocks_found": 2,
            }
        return {
            "question_coverage_score": 25,
            "questions_found": [],
            "questions_missing": ["how to choose dentist"],
            "faq_readiness_score": 20,
            "faq_blocks_found": 0,
            "faq_schema_present": False,
            "snippet_readiness_score": 10,
            "answer_blocks_found": 0,
        }

    monkeypatch.setattr(
        "accounts.serializers.get_aeo_content_readiness_for_site",
        fake_get_aeo_content_readiness_for_site,
    )
    monkeypatch.setattr(
        "accounts.serializers.generate_aeo_recommendations",
        lambda aeo_data, seo_data: [f"rec-{int(aeo_data.get('question_coverage_score') or 0)}"] * 5,
    )
    monkeypatch.setattr("accounts.serializers.get_or_refresh_seo_score_for_user", lambda *args, **kwargs: {})

    serializer = BusinessProfileSerializer([p1, p2], many=True)
    data = serializer.data

    assert len(data) == 2
    assert data[0]["question_coverage_score"] == 90
    assert data[1]["question_coverage_score"] == 25
    # No cross-profile leak in many=True context.
    assert calls["count"] == 2


@pytest.mark.django_db
def test_aeo_serializer_uses_and_matches_persisted_snapshot(monkeypatch):
    user = User.objects.create_user(username="u2", email="u2@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        website_url="https://whitepinedentalcare.com",
        industry="dental",
    )

    calls = {"count": 0}

    def fake_get_aeo_content_readiness_for_site(**kwargs):
        calls["count"] += 1
        return {
            "question_coverage_score": 75,
            "questions_found": ["what is invisalign"],
            "questions_missing": ["how long invisalign takes"],
            "faq_readiness_score": 70,
            "faq_blocks_found": 2,
            "faq_schema_present": True,
            "snippet_readiness_score": 60,
            "answer_blocks_found": 1,
        }

    monkeypatch.setattr(
        "accounts.serializers.get_aeo_content_readiness_for_site",
        fake_get_aeo_content_readiness_for_site,
    )
    monkeypatch.setattr(
        "accounts.serializers.generate_aeo_recommendations",
        lambda aeo_data, seo_data: ["a", "b", "c", "d", "e"],
    )
    monkeypatch.setattr("accounts.serializers.get_or_refresh_seo_score_for_user", lambda *args, **kwargs: {})

    first = BusinessProfileSerializer(profile).data
    snapshot = AEOOverviewSnapshot.objects.get(profile=profile)

    assert first["aeo_score"] == snapshot.aeo_score
    assert first["question_coverage_score"] == snapshot.question_coverage_score
    assert first["faq_readiness_score"] == snapshot.faq_readiness_score
    assert first["snippet_readiness_score"] == snapshot.snippet_readiness_score
    assert first["aeo_recommendations"] == snapshot.aeo_recommendations

    # Replace helper with a failure; serializer should now serve from snapshot.
    monkeypatch.setattr(
        "accounts.serializers.get_aeo_content_readiness_for_site",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    second = BusinessProfileSerializer(profile).data

    assert second["aeo_score"] == first["aeo_score"]
    assert second["question_coverage_score"] == first["question_coverage_score"]
    assert second["faq_readiness_score"] == first["faq_readiness_score"]
    assert second["snippet_readiness_score"] == first["snippet_readiness_score"]
    assert second["aeo_recommendations"] == first["aeo_recommendations"]
    assert calls["count"] == 1


@pytest.mark.django_db
def test_aeo_recommendations_refresh_every_7_days(monkeypatch):
    user = User.objects.create_user(username="u3", email="u3@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        website_url="https://fresh.example.com",
        industry="dental",
    )
    monkeypatch.setattr(
        "accounts.serializers.get_aeo_content_readiness_for_site",
        lambda **kwargs: {
            "question_coverage_score": 40,
            "questions_found": [],
            "questions_missing": ["q1"],
            "faq_readiness_score": 30,
            "faq_blocks_found": 0,
            "faq_schema_present": False,
            "snippet_readiness_score": 20,
            "answer_blocks_found": 0,
            "aeo_status": "ready",
        },
    )
    rec_calls = {"count": 0}

    def fake_generate_aeo_recommendations(aeo_data, seo_data):
        rec_calls["count"] += 1
        return [f"rec-{rec_calls['count']}-{i}" for i in range(5)]

    monkeypatch.setattr("accounts.serializers.generate_aeo_recommendations", fake_generate_aeo_recommendations)
    monkeypatch.setattr("accounts.serializers.get_or_refresh_seo_score_for_user", lambda *args, **kwargs: {})

    first = BusinessProfileSerializer(profile).data
    assert len(first["aeo_recommendations"]) == 5
    assert rec_calls["count"] == 1

    # Within TTL, should reuse persisted recommendations.
    second = BusinessProfileSerializer(profile).data
    assert second["aeo_recommendations"] == first["aeo_recommendations"]
    assert rec_calls["count"] == 1

    # Move recommendation refresh timestamp beyond TTL and ensure regeneration.
    snap = AEOOverviewSnapshot.objects.get(profile=profile)
    snap.aeo_recommendations_refreshed_at = timezone.now() - timedelta(days=8)
    snap.save(update_fields=["aeo_recommendations_refreshed_at"])

    third = BusinessProfileSerializer(profile, context={"force_aeo_refresh": True}).data
    assert len(third["aeo_recommendations"]) == 5
    assert rec_calls["count"] == 2
