import pytest
from django.contrib.auth import get_user_model
from django.db import connection
from rest_framework.test import APIClient

from accounts.models import AEORecommendationRun, AEOResponseSnapshot, BusinessProfile
from accounts.views import _improvement_recommendations_for_prompt

User = get_user_model()


def test_grouped_create_content_matches_by_matched_prompt_texts():
    recs = [
        {
            "action_type": "create_content",
            "prompt": "A (+3 similar prompts)",
            "nl_explanation": "Add a dedicated service page.",
            "priority": "high",
            "references": {
                "matched_prompt_texts": ["A", "B", "C", "D"],
            },
        }
    ]
    out = _improvement_recommendations_for_prompt("C", set(), recs)
    assert len(out) == 1
    assert out[0]["text"] == "Add a dedicated service page."


def test_grouped_acquire_citation_matches_by_matched_response_snapshot_ids():
    recs = [
        {
            "action_type": "acquire_citation",
            "nl_explanation": "Update your trusted listings.",
            "priority": "medium",
            "references": {
                "matched_response_snapshot_ids": [10, 11, 12],
            },
        }
    ]
    out = _improvement_recommendations_for_prompt("Any", {11}, recs)
    assert len(out) == 1
    assert out[0]["action_type"] == "acquire_citation"


def test_legacy_mapping_still_works_and_dedupes_normalized_text():
    recs = [
        {
            "action_type": "create_content",
            "prompt": "Best dentist near me",
            "nl_explanation": "Add an FAQ section.  ",
            "priority": "high",
        },
        {
            "action_type": "create_content",
            "prompt": "Best dentist near me",
            "nl_explanation": "  add an faq section.",
            "priority": "high",
        },
        {
            "action_type": "acquire_citation",
            "reason": "Update your GBP listing.",
            "references": {"response_snapshot_id": 7},
        },
    ]
    out = _improvement_recommendations_for_prompt("Best dentist near me", {7}, recs)
    assert len(out) == 2
    assert any(i["text"].lower().startswith("add an faq section") for i in out)
    assert any(i["text"] == "Update your GBP listing." for i in out)


@pytest.mark.django_db
@pytest.mark.skipif(connection.vendor == "sqlite", reason="Local SQLite test DB migration incompatibility")
def test_prompt_coverage_endpoint_maps_grouped_recommendations():
    user = User.objects.create_user(username="map1", email="map1@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        selected_aeo_prompts=["A", "B", "C", "D"],
    )

    rsp = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="C",
        prompt_hash="h_c",
        raw_response="r",
        platform="openai",
    )

    AEORecommendationRun.objects.create(
        profile=profile,
        recommendations_json=[
            {
                "action_type": "create_content",
                "prompt": "A (+3 similar prompts)",
                "priority": "high",
                "nl_explanation": "Create a dedicated service page.",
                "references": {
                    "matched_prompt_texts": ["A", "B", "C", "D"],
                    "matched_response_snapshot_ids": [rsp.id],
                },
            }
        ],
    )

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/aeo/prompt-coverage/")
    assert res.status_code == 200
    data = res.json()
    rows = data.get("prompts") or []
    row_c = next((r for r in rows if r.get("prompt") == "C"), None)
    assert row_c is not None
    recs = row_c.get("improvement_recommendations") or []
    assert len(recs) == 1
    assert recs[0]["text"] == "Create a dedicated service page."

