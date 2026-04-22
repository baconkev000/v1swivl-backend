"""Deterministic SEO issue engine behavior."""

from datetime import date

import pytest
from django.contrib.auth import get_user_model

from accounts.models import BusinessProfile, SEOOverviewSnapshot
from accounts.openai_utils import generate_seo_next_steps
from accounts.seo.seo_issue_engine import build_structured_issues
from accounts.seo.seo_issue_engine import build_structured_recommendations

User = get_user_model()


def test_high_volume_keyword_without_rank_generates_create_page_recommendation():
    issues = build_structured_issues(
        ranked_keywords=[
            {
                "keyword": "enterprise payroll software",
                "search_volume": 4200,
                "rank": None,
            },
        ],
    )
    assert any(i["issue_id"] == "missing_keyword_page" for i in issues)
    recs = build_structured_recommendations(issues)
    target = next(r for r in recs if r["issue_id"] == "missing_keyword_page")
    assert "Create" in target["exact_fix"] or "create" in target["exact_fix"]
    assert "enterprise payroll software" in target["exact_fix"]
    why_l = target["why_it_matters"].lower()
    assert any(x in why_l for x in ("ai", "assistant", "overview", "cite", "quotable"))
    assert "execution" in target and isinstance(target["execution"], dict)
    assert "impact" in target and isinstance(target["impact"], dict)


def test_competitor_outranking_generates_competitor_recommendation():
    issues = build_structured_issues(
        ranked_keywords=[],
        domain_intersection=[
            {
                "keyword": "best fleet tracking software",
                "search_volume": 1800,
                "your_rank": None,
                "top_competitor_rank": 3,
                "top_competitor_domain": "competitor.example",
                "your_url": "",
            },
        ],
    )
    assert any(i["issue_id"] == "competitor_outperforming" for i in issues)
    recs = build_structured_recommendations(issues)
    target = next(r for r in recs if r["issue_id"] == "competitor_outperforming")
    assert "competitor" in target["why_it_matters"].lower()
    assert "best fleet tracking software" in target["exact_fix"]


def test_thin_content_gap_generates_content_expansion_recommendation():
    issues = build_structured_issues(
        ranked_keywords=[],
        on_page={
            "keyword": "warehouse automation platform",
            "search_volume": 1100,
            "rank": 18,
            "user_url": "https://example.com/warehouse-automation",
            "user_word_count": 600,
            "competitor_word_counts": [1300, 1500, 1400],
        },
    )
    assert any(i["issue_id"] == "insufficient_content_depth" for i in issues)
    recs = build_structured_recommendations(issues)
    target = next(r for r in recs if r["issue_id"] == "insufficient_content_depth")
    assert "Expand" in target["exact_fix"] or "expand" in target["exact_fix"]
    assert "example.com/warehouse-automation" in target["exact_fix"]


@pytest.mark.django_db
def test_generate_seo_next_steps_persists_structured_issues_on_snapshot(monkeypatch):
    def fake_rewrite(_rec, *, snapshot_context=None):
        return {
            "title": "Action title for SEO",
            "why_it_matters": "This matters for visibility and click-through from organic search results.",
            "exact_fix": "Publish a dedicated page with headings, FAQs, and internal links from the homepage.",
            "example": "Example: /services/example-topic/ with schema and proof sections.",
        }

    monkeypatch.setattr("accounts.openai_utils._rewrite_structured_seo_recommendation", fake_rewrite)

    user = User.objects.create_user(username="seoiss", email="seoiss@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        website_url="https://example.com",
    )
    snap = SEOOverviewSnapshot.objects.create(
        user=user,
        business_profile=profile,
        period_start=date(2026, 1, 1),
        cached_domain="example.com",
        cached_location_mode="organic",
        cached_location_code=0,
        top_keywords=[
            {
                "keyword": "enterprise payroll software",
                "search_volume": 4200,
                "rank": None,
            },
        ],
    )
    seo_data = {
        "seo_score": 40,
        "missed_searches_monthly": 100,
        "organic_visitors": 50,
        "total_search_volume": 5000,
        "search_visibility_percent": 10,
        "top_keywords": snap.top_keywords,
        "on_page": {},
        "serp": [{"type": "people_also_ask", "items": []}],
    }
    generate_seo_next_steps(seo_data, snapshot=snap)
    snap.refresh_from_db()
    assert snap.seo_structured_issues_refreshed_at is not None
    assert len(snap.seo_structured_issues) >= 1
    assert any(
        (row or {}).get("issue_id") == "missing_keyword_page"
        for row in (snap.seo_structured_issues or [])
    )

