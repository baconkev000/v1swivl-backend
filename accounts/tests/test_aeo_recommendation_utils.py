from types import SimpleNamespace

import pytest

from accounts.aeo.aeo_recommendation_utils import (
    _build_onpage_crawl_summary,
    _build_sanitized_nl_signals,
    _competitor_display_names,
    _group_gap_objects_for_recommendations,
    _industry_snippet_for_copy,
    _nl_template,
    _prompt_short_label,
    _region_label_for_profile,
    analyze_citation_gaps,
    analyze_visibility_gaps,
    build_recommendation_strategies_from_flat,
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
        "absence_reason",
        "intent_type",
        "content_angle",
    }
    assert "reason" not in s and "nl_explanation" not in s
    assert s["prompt"] == "Best dentist?"
    assert s["competitors"] == ["Acme"]
    assert s["score"]["visibility_pct"] == 12.0
    assert s["absence_reason"] == "missing_category_page"
    assert s["intent_type"] == "comparison"
    assert s["content_angle"] == "comparison"


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
    assert s["absence_reason"] == "entity_confusion"


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


def test_sanitized_nl_signals_preserves_allowed_absence_and_intent():
    raw = {
        "gap_kind": "visibility_miss",
        "prompt_text": "How much does this service cost?",
        "business_name": "Co",
        "region_label": "X",
        "absence_reason": "missing_trust_signal",
        "intent_type": "transactional",
    }
    s = _build_sanitized_nl_signals(raw)
    assert s["absence_reason"] == "missing_trust_signal"
    assert s["intent_type"] == "transactional"
    assert s["content_angle"] == "trust_proof"


def test_analyze_citation_gaps_filters_competitor_and_non_directory_domains():
    ex = SimpleNamespace(
        id=1,
        response_snapshot_id=1,
        citations_json=["competitor.com", "yelp.com", "randombrand.ai"],
        competitors_json=[{"name": "Comp", "url": "https://competitor.com"}],
    )
    gaps = analyze_citation_gaps(None, [ex], citation_share=10.0)
    domains = [g.get("source_domain") for g in gaps if g.get("source_domain")]
    assert "competitor.com" not in domains
    assert "randombrand.ai" not in domains
    assert "yelp.com" in domains


def test_group_gaps_by_absence_reason_and_content_angle():
    grouped = _group_gap_objects_for_recommendations(
        [
            {
                "gap_kind": "visibility_miss",
                "prompt_text": "best forklift rental",
                "absence_reason": "missing_category_page",
                "content_angle": "service_offer",
            },
            {
                "gap_kind": "visibility_miss",
                "prompt_text": "forklift rental pricing",
                "absence_reason": "missing_category_page",
                "content_angle": "service_offer",
            },
            {
                "gap_kind": "visibility_miss",
                "prompt_text": "is forklift rental insured",
                "absence_reason": "missing_trust_signal",
                "content_angle": "trust_proof",
            },
        ],
        action_type="create_content",
    )
    assert len(grouped) == 2
    first = [g for g in grouped if g.get("absence_reason") == "missing_category_page"][0]
    assert "(+1 similar prompts)" in first.get("prompt_text", "")


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
    assert "official website" in live.lower()
    assert "footer" in broken.lower() or "official website" in broken.lower()
    assert "canonical" not in live.lower()
    assert "disambiguation" not in live.lower()


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


def test_build_recommendation_strategies_groups_and_dedupes_actions():
    gid = "aeo_grp_test123"
    flat = [
        {
            "parent_group_id": gid,
            "action_type": "create_content",
            "angle": "schema",
            "priority": "high",
            "rec_id": "a1",
            "applies_to": {
                "prompt_count": 2,
                "prompt_examples": ["best cookies nationwide", "cookie delivery options"],
                "response_snapshot_ids": [1, 2],
                "cluster_summary": "intent:comparison; focus:service offer",
            },
            "actions": [
                {
                    "title": "Add JSON-LD for this page type",
                    "description": "First",
                    "priority": "high",
                },
                {
                    "title": "Tie work to an existing site page",
                    "description": "Use crawl context as the anchor: Topic seeds: foo",
                    "priority": "low",
                },
            ],
        },
        {
            "parent_group_id": gid,
            "action_type": "create_content",
            "angle": "on_page",
            "priority": "medium",
            "rec_id": "a2",
            "applies_to": {
                "prompt_count": 2,
                "prompt_examples": ["best cookies nationwide"],
                "response_snapshot_ids": [1],
                "cluster_summary": "intent:comparison; focus:service offer",
            },
            "actions": [
                {
                    "title": "Add JSON-LD for this page type",
                    "description": "Second longer description for merge",
                    "priority": "medium",
                },
            ],
        },
    ]
    strategies = build_recommendation_strategies_from_flat(flat, monitored_prompt_count=10)
    assert len(strategies) == 1
    s0 = strategies[0]
    assert s0["strategy_id"] == gid
    assert "title" in s0 and len(s0["title"].split()) <= 10
    assert "summary" in s0 and len(s0["summary"]) > 20
    assert s0["applies_to"]["prompt_count"] >= 1
    assert len(s0["applies_to"]["prompt_examples"]) <= 3
    assert len(s0["angles"]) == 1
    assert s0["angles"][0]["angle"] == "todo"
    all_titles = [a["title"] for b in s0["angles"] for a in b["actions"]]
    assert all_titles.count("Add JSON-LD for this page type") == 1
    assert not any("Tie work" in t for t in all_titles)
    assert not any("crawl" in t.lower() for t in all_titles)


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
