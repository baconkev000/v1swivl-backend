"""Tests for Phase 3 AEO extraction grounding (brand_mentioned vs raw answer text)."""

from accounts.aeo.aeo_extraction_utils import (
    _sanitize_competitors,
    normalize_extraction_payload,
    parse_competitor_raw_item,
)


def _base_payload(**overrides):
    base = {
        "brand_mentioned": False,
        "mention_position": "none",
        "mention_count": 0,
        "competitors": [],
        "ranking_order": [],
        "citations": [],
        "sentiment": "neutral",
        "confidence_score": 0.9,
    }
    base.update(overrides)
    return base


def test_competitors_only_no_target_name_brand_false():
    raw = (
        "Here are some options: Murray Dental, Summit Smiles, and River City Family Dentistry "
        "are well regarded in the area."
    )
    out = normalize_extraction_payload(
        _base_payload(
            brand_mentioned=True,
            mention_position="top",
            mention_count=1,
            competitors=["Murray Dental", "Summit Smiles"],
        ),
        raw_response=raw,
        tracked_business_name="White Pine Dental",
        tracked_website_domain="whitepinedental.com",
    )
    assert out["brand_mentioned"] is False
    assert out["mention_position"] == "none"
    assert out["mention_count"] == 0
    assert any(c.get("name") == "Murray Dental" for c in out["competitors"])


def test_target_name_in_raw_brand_true():
    raw = "Patients often recommend White Pine Dental for routine cleanings and cosmetic work."
    out = normalize_extraction_payload(
        _base_payload(
            brand_mentioned=True,
            mention_position="middle",
            mention_count=1,
        ),
        raw_response=raw,
        tracked_business_name="White Pine Dental",
        tracked_website_domain="",
    )
    assert out["brand_mentioned"] is True
    assert out["mention_count"] >= 1
    assert out["mention_position"] != "none"


def test_model_true_without_target_in_text_overridden_false(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    raw = "Only Murray Dental and generic dental offices are mentioned here."
    out = normalize_extraction_payload(
        _base_payload(
            brand_mentioned=True,
            mention_position="top",
            mention_count=2,
        ),
        raw_response=raw,
        tracked_business_name="White Pine Dental",
        tracked_website_domain="",
    )
    assert out["brand_mentioned"] is False
    assert out["mention_position"] == "none"
    assert out["mention_count"] == 0
    assert any("did not ground target" in r.message for r in caplog.records)


def test_invariant_zero_count_and_none_position_forces_brand_false():
    """When grounding is skipped (no tracked name), model cannot claim a mention with no position/count."""
    out = normalize_extraction_payload(
        _base_payload(
            brand_mentioned=True,
            mention_position="none",
            mention_count=0,
        ),
        raw_response="Some answer text without business context.",
        tracked_business_name="",
        tracked_website_domain="",
    )
    assert out["brand_mentioned"] is False
    assert out["mention_position"] == "none"
    assert out["mention_count"] == 0


def test_sanitize_competitors_dedupes_same_root_url():
    raw = [
        {"name": "Acme Dental", "url": "https://acme.com/about"},
        {"name": "Acme Dentistry", "url": "https://www.acme.com/team"},
        {"name": "Beta Clinic", "url": "https://beta.org"},
    ]
    out = _sanitize_competitors(raw)
    assert len(out) == 2
    assert out[0]["name"] == "Acme Dental"
    assert out[1]["name"] == "Beta Clinic"


def test_parse_competitor_python_repr_string():
    """DB may store dicts as Python repr strings instead of JSON objects."""
    s = "{'name': 'Cottonwood Dental', 'url': 'https://www.cottonwooddental.com'}"
    out = parse_competitor_raw_item(s)
    assert out["name"] == "Cottonwood Dental"
    assert out["url"] == "https://www.cottonwooddental.com"


def test_sanitize_competitors_repr_strings_become_json_shapes():
    raw = [
        "{'name': 'Murray Dental', 'url': 'https://www.murraydental.com'}",
        '{"name": "Beta", "url": "https://beta.example.com"}',
    ]
    out = _sanitize_competitors(raw)
    assert len(out) == 2
    assert out[0] == {"name": "Murray Dental", "url": "https://www.murraydental.com"}
    assert out[1]["name"] == "Beta"


def test_sanitize_competitors_legacy_strings():
    out = _sanitize_competitors(["Foo", "Bar", "Foo"])
    assert len(out) == 2
    assert out[0] == {"name": "Foo", "url": ""}
    assert out[1]["name"] == "Bar"


def test_domain_only_grounds_brand():
    raw = "Book online at https://whitepinedental.com/appointments"
    out = normalize_extraction_payload(
        _base_payload(brand_mentioned=False, mention_position="none", mention_count=0),
        raw_response=raw,
        tracked_business_name="White Pine Dental",
        tracked_website_domain="whitepinedental.com",
    )
    assert out["brand_mentioned"] is True
    assert out["mention_count"] >= 1
