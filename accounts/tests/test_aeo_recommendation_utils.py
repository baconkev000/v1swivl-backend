from types import SimpleNamespace

import pytest

from accounts.aeo.aeo_recommendation_utils import (
    _build_onpage_crawl_summary,
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
