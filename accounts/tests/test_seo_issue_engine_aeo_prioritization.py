"""AEO-focused deterministic SEO issue behavior: clustering, priority, and limits."""

from accounts.seo.seo_issue_engine import (
    build_structured_issues,
    build_structured_recommendations,
)


def _find_issue(issues: list[dict], issue_id: str) -> dict:
    for row in issues:
        if str((row or {}).get("issue_id")) == issue_id:
            return row
    raise AssertionError(f"missing issue_id={issue_id}")


def test_keyword_variants_cluster_into_single_create_page_issue():
    issues = build_structured_issues(
        ranked_keywords=[
            {"keyword": "dentist specialization", "search_volume": 900, "rank": None},
            {"keyword": "dentist specialties", "search_volume": 700, "rank": None},
            {"keyword": "dentistry specialties", "search_volume": 650, "rank": None},
        ],
        on_page={"business_name": "Downtown Dental"},
    )
    create_rows = [r for r in issues if (r or {}).get("issue_id") == "missing_keyword_page"]
    assert len(create_rows) == 1
    ev = create_rows[0].get("evidence") or {}
    kws = [str(x).lower() for x in list(ev.get("target_keywords") or [])]
    assert len(kws) >= 2
    assert any("dentist" in x or "dentistry" in x for x in kws)


def test_local_high_volume_keywords_prioritize_local_trust_issue_high():
    issues = build_structured_issues(
        ranked_keywords=[
            {"keyword": "emergency dentist near me", "search_volume": 1800, "rank": 27},
        ],
        on_page={
            "business_name": "Oak Street Dental",
            "local_trust_signals_count": 0,
            "local_business_schema_present": False,
            "review_signals_present": False,
        },
    )
    local_issue = _find_issue(issues, "missing_local_trust_signals")
    assert str(local_issue.get("priority")) == "high"


def test_create_page_recommendations_are_capped_at_two():
    issues = build_structured_issues(
        ranked_keywords=[
            {"keyword": "orthodontist financing", "search_volume": 1200, "rank": None},
            {"keyword": "teeth whitening cost", "search_volume": 1100, "rank": None},
            {"keyword": "invisalign options", "search_volume": 950, "rank": None},
            {"keyword": "dental implant timeline", "search_volume": 1000, "rank": None},
        ],
        on_page={"business_name": "Smile Dental Group"},
    )
    create_rows = [
        r
        for r in issues
        if str((r or {}).get("recommended_action_type")) == "create_cluster_page"
    ]
    assert len(create_rows) <= 2
    assert len(issues) <= 8


def test_structured_recommendation_contains_aeo_fields():
    issues = build_structured_issues(
        ranked_keywords=[
            {"keyword": "best payroll software", "search_volume": 1400, "rank": 33},
        ],
        on_page={
            "business_name": "Acme Payroll",
            "quick_answer_present": False,
            "answer_blocks_found": 0,
            "structured_facts_present": False,
            "table_blocks_found": 0,
            "comparison_table_present": False,
        },
    )
    recs = build_structured_recommendations(issues)
    assert len(recs) >= 1
    r0 = recs[0]
    assert str(r0.get("type") or "")
    assert str(r0.get("issue_type") or "")
    assert isinstance(r0.get("target_keywords") or [], list)
    assert isinstance((r0.get("impact") or {}).get("traffic_gain_estimate"), int)
    assert str((r0.get("impact") or {}).get("ai_citation_lift") or "") in {"low", "medium", "high"}
    assert isinstance((r0.get("evidence") or {}).get("detected_issues") or [], list)
    assert isinstance(r0.get("execution") or {}, dict)
    assert str(r0.get("aeo_boost") or "")
