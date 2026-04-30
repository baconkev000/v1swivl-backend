"""Cache behavior for domain_intersection/live merged gap results."""

import pytest
from django.core.cache import cache


@pytest.mark.django_db
def test_get_keyword_gap_keywords_cache_hit_skips_post(monkeypatch):
    cache.clear()

    calls: list[tuple[str, object]] = []

    def fake_post(endpoint, payload, business_profile=None):
        calls.append((endpoint, payload))
        if "domain_intersection" in endpoint:
            return {
                "tasks": [
                    {
                        "result": [
                            {
                                "items": [
                                    {
                                        "keyword_data": {
                                            "keyword": "widget software",
                                            "keyword_info": {"search_volume": 900},
                                        },
                                        "first_domain_serp_element": {"rank_absolute": 4},
                                        "second_domain_serp_element": {"url": "https://c.example/x"},
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        return None

    monkeypatch.setattr("accounts.dataforseo_utils._post", fake_post)

    from accounts.dataforseo_utils import get_keyword_gap_keywords

    r1 = get_keyword_gap_keywords(
        "Example.com",
        ["Other.COM", "other.com"],
        location_code=2840,
        language_code="en",
        limit=50,
    )
    r2 = get_keyword_gap_keywords(
        "example.com",
        ["other.com"],
        location_code=2840,
        language_code="en",
        limit=50,
    )
    assert len(r1) >= 1
    assert len(r2) >= 1
    intersection_calls = [c for c in calls if "domain_intersection" in c[0]]
    assert len(intersection_calls) == 1

    get_keyword_gap_keywords(
        "example.com",
        ["other.com"],
        location_code=2840,
        language_code="en",
        limit=100,
    )
    intersection_calls = [c for c in calls if "domain_intersection" in c[0]]
    assert len(intersection_calls) == 2

    get_keyword_gap_keywords(
        "example.com",
        ["other.com"],
        location_code=2840,
        language_code="en",
        limit=50,
        force_refresh=True,
    )
    intersection_calls = [c for c in calls if "domain_intersection" in c[0]]
    assert len(intersection_calls) == 3


@pytest.mark.django_db
def test_get_keyword_gap_skips_task_internal_error_and_merges_ok_competitor(monkeypatch):
    """DataForSEO can return HTTP 200 with task status_code 50000; skip that competitor and continue."""
    cache.clear()

    def fake_post(endpoint, payload, business_profile=None):
        assert "domain_intersection" in endpoint
        t2 = (payload[0] or {}).get("target2") if payload else None
        if t2 == "bad-competitor.example":
            return {
                "status_code": 20000,
                "tasks": [
                    {
                        "status_code": 50000,
                        "status_message": "Internal Error.",
                        "result": [None],
                    }
                ],
            }
        if t2 == "good-competitor.example":
            return {
                "status_code": 20000,
                "tasks": [
                    {
                        "status_code": 20000,
                        "status_message": "Ok.",
                        "result": [
                            {
                                "items": [
                                    {
                                        "keyword_data": {
                                            "keyword": "recovered gap keyword",
                                            "keyword_info": {"search_volume": 500},
                                        },
                                        "first_domain_serp_element": {"rank_absolute": 10},
                                        "second_domain_serp_element": {"url": "https://good-competitor.example/x"},
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
        return None

    monkeypatch.setattr("accounts.dataforseo_utils._post", fake_post)

    from accounts.dataforseo_utils import get_keyword_gap_keywords

    out = get_keyword_gap_keywords(
        "example.com",
        ["bad-competitor.example", "good-competitor.example"],
        location_code=2840,
        language_code="en",
        limit=50,
        force_refresh=True,
    )
    assert any((k or {}).get("keyword") == "recovered gap keyword" for k in out)
