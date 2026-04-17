"""AEO/citation framing for SEO structured rewrite prompts and validation."""

import json

import pytest

from accounts.openai_utils import (
    SEO_STRUCTURED_REWRITE_SYSTEM_PROMPT,
    _coerce_rewrite_payload_to_legacy,
    _rewrite_structured_seo_recommendation,
    _seo_rewrite_output_is_valid,
)


def test_system_prompt_includes_aeo_framing_keywords():
    s = SEO_STRUCTURED_REWRITE_SYSTEM_PROMPT.lower()
    assert "cited" in s or "cite" in s
    assert "assistant" in s or "ai-generated" in s or "overviews" in s
    assert "perplexity" in s or "overview" in s
    assert "json" in s
    assert "why_it_matters" in s
    assert "exact_fix" in s


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


def test_validation_accepts_concrete_fix_with_keyword_anchor():
    rec = {"evidence": {"keyword": "crm software", "url": ""}}
    good = {
        "title": "Add citeable FAQ for CRM software",
        "why_it_matters": "Assistants often quote short answers; crm software queries need an explicit FAQ block.",
        "exact_fix": "Add a five-question FAQ section on the pricing page so AI answers can cite crm software facts.",
        "example": "Example: publish bullet comparisons assistants can quote.",
    }
    assert _seo_rewrite_output_is_valid(good, rec) is True


def test_validation_accepts_url_anchor_when_keyword_missing():
    rec = {"evidence": {"keyword": "", "url": "https://example.com/about"}}
    good = {
        "title": "Entity clarity",
        "why_it_matters": "AI Overviews favor pages with clear statements about who you serve.",
        "exact_fix": "Update https://example.com/about with a comparison table and two FAQ answers.",
        "example": "",
    }
    assert _seo_rewrite_output_is_valid(good, rec) is True


@pytest.mark.parametrize(
    "needle",
    [
        "cited",
        "Perplexity",
        "assistant",
        "JSON",
        "why_it_matters",
    ],
)
def test_chat_rewrite_receives_system_prompt_with_aeo_framing(monkeypatch, needle):
    captured: list[list[dict]] = []

    class _Msg:
        content = json.dumps(
            {
                "title": "Add FAQ for widgets",
                "why_it_matters": "Assistants often cite short FAQ answers for widgets.",
                "exact_fix": "Add an on-page FAQ section about widgets with five Q&As and internal links.",
                "example": "Example: /widgets/faq with schema where appropriate.",
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
                "goal": "Increase inclusion in cited AI answers.",
                "whats_missing": ["No ranking URL for primary keyword"],
                "steps": [
                    {"step_number": 1, "title": "Create page", "instruction": "Publish /service-page/ with quick answer and FAQ"}
                ],
                "copy_paste_content": {
                    "local_trust_block": "",
                    "faq": [{"q": "How long?", "a": "Most projects take 2-4 weeks."}],
                    "quick_facts": ["Typical timeline: 2-4 weeks"],
                },
                "schema_requirements": ["FAQPage"],
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
                    "show_schema_section": True,
                    "show_internal_linking_section": True,
                },
            }
        ]
    }
    out = _coerce_rewrite_payload_to_legacy(payload)
    assert out["title"] == "Create cluster page"
    assert "Increase inclusion" in out["why_it_matters"]
    assert "Create page" in out["exact_fix"]
    assert "action_card" in out and isinstance(out["action_card"], dict)
