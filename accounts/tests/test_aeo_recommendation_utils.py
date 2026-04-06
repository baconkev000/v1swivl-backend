from types import SimpleNamespace

import pytest

from accounts.aeo.aeo_recommendation_utils import (
    _build_onpage_crawl_summary,
    _build_sanitized_nl_signals,
    _competitor_display_names,
    _industry_snippet_for_copy,
    _nl_template,
    _prompt_short_label,
    _region_label_for_profile,
    analyze_visibility_gaps,
)
from accounts.models import AEOExtractionSnapshot


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


def test_sanitized_nl_signals_prompt_is_short_label():
    long = " ".join(f"w{i}" for i in range(15))
    raw = {
        "gap_kind": "visibility_miss",
        "prompt_text": long,
        "business_name": "Co",
        "region_label": "Austin, TX",
        "competitors_in_answer": [],
    }
    s = _build_sanitized_nl_signals(raw)
    assert s["prompt"].endswith("…")
    assert len(s["prompt"].replace("…", "").split()) == 10


def test_prompt_short_label_empty():
    assert _prompt_short_label("") == "this type of question"
    assert _prompt_short_label("   ") == "this type of question"


def test_industry_snippet_first_segment_and_cap():
    assert _industry_snippet_for_copy("material handling usa, mh usa, datum usa") == "material handling usa"
    long = "x" * 80
    sn = _industry_snippet_for_copy(long, max_len=60)
    assert len(sn) == 60
    assert sn.endswith("…")


def test_sanitized_nl_signals_visibility_miss_url_identity():
    raw = {
        "gap_kind": "visibility_miss",
        "prompt_text": "Find a dentist",
        "business_name": "Smile Co",
        "region_label": "Austin, TX",
        "canonical_domain": "smileco.com",
        "brand_mentioned_url_status": AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE,
        "cited_domain_in_answer": ["wrong.com"],
        "url_identity_summary": "Modeled answers tied your brand to wrong.com while your registered site is smileco.com.",
        "verification_summary": "dns_ok;http_ok",
    }
    s = _build_sanitized_nl_signals(raw)
    assert s["canonical_domain"] == "smileco.com"
    assert s["brand_mentioned_url_status"] == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE
    assert s["cited_domain_in_answer"] == ["wrong.com"]
    assert "wrong.com" in s["url_identity_summary"]
    assert s["verification_summary"] == "dns_ok;http_ok"


def test_sanitized_nl_signals_omits_not_mentioned_status():
    raw = {
        "gap_kind": "visibility_miss",
        "prompt_text": "Q",
        "business_name": "Co",
        "region_label": "X",
        "brand_mentioned_url_status": "not_mentioned",
        "canonical_domain": "co.com",
    }
    s = _build_sanitized_nl_signals(raw)
    assert s["canonical_domain"] == "co.com"
    assert "brand_mentioned_url_status" not in s


def test_nl_template_wrong_live_vs_broken():
    base = {
        "gap_kind": "visibility_miss",
        "business_name": "Acme",
        "region_label": "Austin, TX",
        "industry": "",
        "competitors_in_answer": [],
        "url_identity_summary": "Modeled answers tied your brand to wrong.com while your registered site is acme.com.",
    }
    live = _nl_template(
        {
            **base,
            "brand_mentioned_url_status": AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE,
        }
    )
    broken = _nl_template(
        {
            **base,
            "brand_mentioned_url_status": AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN,
        }
    )
    assert "sameAs" in live or "sameas" in live.lower()
    assert "footer" in broken.lower() or "official" in broken.lower()
    assert "disambiguat" in live.lower() or "disambiguation" in live.lower()


def test_analyze_visibility_gaps_merges_url_identity():
    ex = SimpleNamespace(
        id=42,
        brand_mentioned=False,
        competitors_json=[],
        response_snapshot=SimpleNamespace(id=99, prompt_text="Who is the best?"),
        brand_mentioned_url_status=AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN,
        cited_domain_or_url="dead.example",
        url_verification_notes={"dns_ok": False, "http_ok": False},
    )
    gaps = analyze_visibility_gaps(
        None,
        [ex],
        tracked_website_url="https://acme.com",
        canonical_domain="acme.com",
    )
    assert len(gaps) == 1
    assert gaps[0]["brand_mentioned_url_status"] == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN
    assert gaps[0]["cited_domain_in_answer"] == ["dead.example"]
    assert "dead.example" in gaps[0]["url_identity_summary"]
    assert gaps[0]["canonical_domain"] == "acme.com"


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
