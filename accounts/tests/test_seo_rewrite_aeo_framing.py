"""SEO structured rewrite prompts, coercion, and owner-facing validation."""

import json

import pytest

from accounts.openai_utils import (
    SEO_STRUCTURED_REWRITE_SYSTEM_PROMPT,
    _coerce_rewrite_payload_to_legacy,
    _rewrite_structured_seo_recommendation,
    _seo_rewrite_output_is_valid,
)


def test_system_prompt_targets_non_technical_business_owner():
    s = SEO_STRUCTURED_REWRITE_SYSTEM_PROMPT.lower()
    assert "business coach" in s or "marketing consultant" in s
    assert "internal_notes" in s
    assert "non-technical" in s or "business owner" in s
    assert "json" in s
    assert "why_it_matters" in s
    assert "expected_benefit" in s


def test_validation_rejects_vague_seo_fix_without_concrete_action():
    rec = {"evidence": {"keyword": "plumbing repair", "url": "https://example.com/p"}}
    bad = {
        "title": "Improve plumbing page",
        "why_it_matters": "You should improve SEO to get more traffic for plumbing repair.",
        "exact_fix": "Optimize SEO and boost rankings for better visibility online.",
        "example": "",
    }
    assert _seo_rewrite_output_is_valid(bad, rec) is False


def test_validation_rejects_aeo_vague_buzzword_without_action():
    rec = {"evidence": {"keyword": "crm software", "url": ""}}
    bad = {
        "title": "CRM",
        "why_it_matters": "AI visibility matters for crm software.",
        "exact_fix": "Improve AI visibility without changing the site structure.",
        "example": "",
    }
    assert _seo_rewrite_output_is_valid(bad, rec) is False


def test_validation_rejects_slug_like_title():
    rec = {"evidence": {"keyword": "plumbing repair", "url": "https://example.com/p"}}
    bad = {
        "title": "local_trust_gap",
        "why_it_matters": "Customers need to find your plumbing repair services nearby.",
        "exact_fix": "Add your business name, address, and phone on the homepage and contact page for plumbing repair.",
        "example": "",
    }
    assert _seo_rewrite_output_is_valid(bad, rec) is False


def test_validation_rejects_forbidden_jargon_in_user_fields():
    rec = {"evidence": {"keyword": "dentist", "url": "https://example.com"}}
    bad = {
        "title": "Add structured data",
        "why_it_matters": "JSON-LD helps search engines understand your dentist listings.",
        "exact_fix": "Implement LocalBusiness schema and FAQ schema on the homepage with dentist hours.",
        "example": "",
    }
    assert _seo_rewrite_output_is_valid(bad, rec) is False


def test_validation_accepts_concrete_fix_with_keyword_anchor():
    rec = {"evidence": {"keyword": "crm software", "url": ""}}
    good = {
        "title": "Help visitors compare CRM software on your site",
        "why_it_matters": "When people compare crm software, a clear on-page section keeps them on your website longer.",
        "exact_fix": "Add a five-question FAQ section on the pricing page so visitors see crm software facts before they call.",
        "example": "Example: publish short bullet points visitors can scan quickly.",
    }
    assert _seo_rewrite_output_is_valid(good, rec) is True


def test_validation_accepts_url_anchor_when_keyword_missing():
    rec = {"evidence": {"keyword": "", "url": "https://example.com/about"}}
    good = {
        "title": "Clarify who you serve on your About page",
        "why_it_matters": "Visitors decide faster when your About page states who you help and where you work.",
        "exact_fix": "Update https://example.com/about with a simple comparison table and two short FAQ answers.",
        "example": "",
    }
    assert _seo_rewrite_output_is_valid(good, rec) is True


@pytest.mark.parametrize(
    "needle",
    [
        "internal_notes",
        "business coach",
        "JSON",
        "why_it_matters",
    ],
)
def test_chat_rewrite_receives_system_prompt_with_owner_framing(monkeypatch, needle):
    captured: list[list[dict]] = []

    class _Msg:
        content = json.dumps(
            {
                "title": "Add FAQ for widgets",
                "why_it_matters": "Shoppers often read short answers before buying widgets.",
                "exact_fix": "Add an on-page FAQ section about widgets with five questions and clear contact details on the same page.",
                "example": "Example: /widgets with five short answers visitors can scan quickly.",
            }
        )

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    def _fake_chat_completion_create_logged(*_a, **kw):
        captured.append(kw.get("messages") or [])
        return _Completion()

    monkeypatch.setattr(
        "accounts.openai_utils.chat_completion_create_logged",
        _fake_chat_completion_create_logged,
    )
    monkeypatch.setattr("accounts.openai_utils._get_client", lambda *_a, **_k: object())
    monkeypatch.setattr("accounts.openai_utils._get_model", lambda: "gpt-test-model")

    rec = {
        "issue_id": "missing_keyword_page",
        "issue": "Missing widgets page",
        "priority": "high",
        "why_it_matters": "old why",
        "exact_fix": "old fix",
        "example": "",
        "evidence": {"keyword": "widgets", "search_volume": 900, "rank": None},
        "recommended_action_type": "create_keyword_page",
        "impact_score": 0.5,
        "effort_score": 0.3,
        "confidence": 0.9,
    }
    out = _rewrite_structured_seo_recommendation(rec, snapshot_context=None)
    assert len(captured) == 1
    system_text = (captured[0][0].get("content") or "").lower()
    user_text = (captured[0][1].get("content") or "").lower()
    assert needle.lower() in system_text or needle.lower() in user_text
    assert out.get("title")
    assert "widgets" in (out.get("exact_fix") or "").lower()


def test_coerce_execution_tasks_payload_to_legacy_shape():
    payload = {
        "execution_tasks": [
            {
                "title": "Add trust block",
                "priority": "high",
                "type": "local_trust",
                "target_url": "/emergency-dentist-near-me/",
                "goal": "Increase citation confidence for local intent.",
                "implementation": [
                    "Add above-the-fold trust block",
                    "Add LocalBusiness schema",
                ],
                "content_requirements": {
                    "trust_block_example": "Serving Denver metro, 4.8/5 from 120+ reviews.",
                },
                "ai_optimization_notes": "Provides verifiable local signals.",
            }
        ]
    }
    out = _coerce_rewrite_payload_to_legacy(payload)
    assert out["title"] == "Add trust block"
    assert "Increase citation confidence" in out["why_it_matters"]
    assert "1) Add above-the-fold trust block" in out["exact_fix"]
    assert "Serving Denver metro" in out["example"]


def test_coerce_actions_card_payload_to_legacy_shape():
    payload = {
        "actions": [
            {
                "id": "seo-1",
                "source": "seo",
                "category_label": "SEO",
                "pillar": "Content",
                "priority": "high",
                "title": "Create cluster page",
                "subtitle": "Own high-intent cluster",
                "why_it_matters": "Shoppers compare options online before they call.",
                "goal": "Increase inclusion in cited AI answers.",
                "whats_missing": ["No ranking URL for primary keyword"],
                "expected_benefit": "More qualified calls from people ready to book.",
                "internal_notes": ["Add FAQPage schema when publishing"],
                "steps": [
                    {"step_number": 1, "title": "Create page", "instruction": "Publish /service-page/ with quick answer and FAQ"}
                ],
                "copy_paste_content": {
                    "local_trust_block": "",
                    "faq": [{"q": "How long?", "a": "Most projects take 2-4 weeks."}],
                    "quick_facts": ["Typical timeline: 2-4 weeks"],
                },
                "schema_requirements": [],
                "internal_linking": [],
                "target_url": "/service-page/",
                "ai_optimization_notes": {"why_it_helps": ["Structured answers"], "expected_impact": ["Higher inclusion"]},
                "evidence": {
                    "keyword": "service page",
                    "search_volume": 900,
                    "rank": None,
                    "competitor_rank": 3,
                    "competitor_domains": ["example.com"],
                    "location": "",
                    "source_issue_ids": ["missing_keyword_page"],
                },
                "display_hints": {
                    "expanded_by_default": False,
                    "show_copy_paste_section": True,
                    "show_schema_section": False,
                    "show_internal_linking_section": True,
                },
            }
        ]
    }
    out = _coerce_rewrite_payload_to_legacy(payload)
    assert out["title"] == "Create cluster page"
    assert "Shoppers compare options" in out["why_it_matters"]
    assert "Create page" in out["exact_fix"]
    assert out.get("expected_benefit") == "More qualified calls from people ready to book."
    assert "FAQPage schema" in " ".join(out.get("internal_notes") or [])
    assert "action_card" in out and isinstance(out["action_card"], dict)
