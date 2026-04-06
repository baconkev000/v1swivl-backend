from types import SimpleNamespace

import pytest

from accounts.aeo.aeo_recommendation_utils import (
    _build_onpage_crawl_summary,
    _build_sanitized_nl_signals,
    _competitor_display_names,
    _region_label_for_profile,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([{"name": "Acme", "url": "acme.com"}], ["Acme"]),
        ([{"url": "https://beta.io"}], ["https://beta.io"]),
        ([{"name": "", "url": "gamma.test"}], ["gamma.test"]),
        ([{"n": 1}], []),
        (["Plain Co", {"name": "Dict Co"}], ["Plain Co", "Dict Co"]),
    ],
)
def test_competitor_display_names(raw, expected):
    assert _competitor_display_names(raw) == expected


def test_region_label_avoids_bare_united_states():
    p = SimpleNamespace(business_address="United States", industry="", website_url="")
    assert _region_label_for_profile(p, "") == "your market"
    p2 = SimpleNamespace(business_address="United States", industry="Dental", website_url="")
    assert _region_label_for_profile(p2, "") == "the Dental market"


def test_region_label_uses_city_when_specific():
    p = SimpleNamespace(business_address="", industry="", website_url="")
    assert _region_label_for_profile(p, "Austin, TX") == "Austin, TX"


def test_sanitized_nl_signals_omits_reason_and_nl_explanation():
    raw = {
        "gap_kind": "visibility_miss",
        "prompt_text": "Best dentist?",
        "business_name": "Smile Co",
        "region_label": "Austin, TX",
        "onpage_crawl_summary": "Topic seeds: cleaning",
        "competitors_in_answer": [{"name": "Acme", "url": "x.com"}],
        "reason": "Visibility gap: long generic text",
        "nl_explanation": "Old NL",
        "score": {"visibility_pct": 12.0, "citation_share_pct": 8.0},
        "action_type": "create_content",
    }
    s = _build_sanitized_nl_signals(raw)
    assert set(s.keys()) == {
        "prompt",
        "action_type",
        "competitors",
        "business_name",
        "region",
        "gap_kind",
        "score",
        "crawl_summary",
    }
    assert "reason" not in s and "nl_explanation" not in s
    assert s["prompt"] == "Best dentist?"
    assert s["competitors"] == ["Acme"]
    assert s["score"]["visibility_pct"] == 12.0


def test_build_onpage_crawl_summary_from_namespace():
    crawl = SimpleNamespace(
        crawl_topic_seeds=[{"label": "Teeth cleaning", "tokens": ["cleaning"]}],
        pages=[
            {
                "url": "https://example.com/a",
                "page_title": "Home",
                "h1": "Welcome",
                "meta_description": "We fix smiles daily.",
                "schema_types": ["LocalBusiness", "FAQPage"],
            }
        ],
    )
    s = _build_onpage_crawl_summary(crawl)
    assert "Teeth cleaning" in s
    assert "Home" in s
    assert "Welcome" in s
    assert "smiles" in s
    assert "LocalBusiness" in s
