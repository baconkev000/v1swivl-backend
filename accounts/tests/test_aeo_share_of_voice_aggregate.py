"""Tests for AEO share-of-voice aggregation from extraction snapshots."""

from types import SimpleNamespace

import pytest

from accounts.aeo.aeo_scoring_utils import aggregate_aeo_share_of_voice_from_extractions


def _ex(mention_count: int, competitors: list[str]) -> SimpleNamespace:
    return SimpleNamespace(mention_count=mention_count, competitors_json=competitors)


def test_share_of_voice_top_three_and_others_percentages():
    extractions = [
        _ex(2, ["Alpha Dental", "Beta Smile"]),
        _ex(0, ["Alpha Dental", "Gamma Care", "Delta Dental"]),
    ]
    out = aggregate_aeo_share_of_voice_from_extractions(
        extractions,
        business_display_name="White Pine Dental",
    )
    assert out["total_prompts"] == 2
    # your: 2, comps: Alpha 2, Beta 1, Gamma 1, Delta 1 => 7 total units
    assert out["total_mention_units"] == 7
    assert out["your_mention_units"] == 2
    assert out["has_data"] is True
    rows = out["rows"]
    assert rows[0]["name"] == "White Pine Dental"
    assert rows[0]["you"] is True
    assert rows[0]["pct"] == pytest.approx(100.0 * 2 / 7, rel=0, abs=0.15)
    names = [r["name"] for r in rows]
    assert "Alpha Dental" in names
    assert "Beta Smile" in names
    assert "Gamma Care" in names
    assert "Others" in names
    assert names.index("Others") == len(names) - 1
    # Delta is the only name not in top 3 by count (Alpha=2, others=1)
    others_row = next(r for r in rows if r["name"] == "Others")
    assert others_row["units"] == 1
    assert others_row["pct"] == pytest.approx(100.0 / 7, rel=0, abs=0.15)


def test_share_of_voice_no_competitors_only_you():
    extractions = [_ex(3, [])]
    out = aggregate_aeo_share_of_voice_from_extractions(
        extractions,
        business_display_name="Solo Co",
    )
    assert out["has_data"] is True
    assert len(out["rows"]) == 1
    assert out["rows"][0]["pct"] == 100.0


def test_share_of_voice_empty_extractions():
    out = aggregate_aeo_share_of_voice_from_extractions(
        [],
        business_display_name="Nobody",
    )
    assert out["total_prompts"] == 0
    assert out["has_data"] is False
    assert out["rows"][0]["pct"] == 0.0
