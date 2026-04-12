import json

from django.test import override_settings

from accounts.onboarding_review_topics import (
    dedupe_and_cap_topics,
    generate_review_topics_for_domain,
    parse_review_topics_json,
)


def test_parse_review_topics_json_valid():
    payload = {
        "topics": [
            {"topic": "Corporate Cards", "category": "product", "rationale": "Core SKU"},
            {"topic": "Expense Reporting", "category": "service"},
        ]
    }
    items, err = parse_review_topics_json(json.dumps(payload))
    assert err is None
    assert len(items) == 2
    assert items[0]["topic"] == "Corporate Cards"
    assert items[0]["category"] == "product"
    assert items[0]["rationale"] == "Core SKU"


def test_parse_review_topics_json_invalid_returns_error():
    items, err = parse_review_topics_json("not json")
    assert items == []
    assert err and "review_topics_invalid_json" in err


def test_dedupe_and_cap_topics():
    raw = [
        {"topic": "Same Topic"},
        {"topic": "same topic", "category": "x"},
        {"topic": "Other"},
    ]
    out = dedupe_and_cap_topics(raw, cap=20)
    assert len(out) == 2
    assert out[0]["topic"] == "Same Topic"
    assert out[1]["topic"] == "Other"


def test_dedupe_and_cap_topics_respects_max():
    raw = [{"topic": f"T{i}"} for i in range(25)]
    out = dedupe_and_cap_topics(raw, cap=20)
    assert len(out) == 20


@override_settings(PERPLEXITY_API_KEY="pk-test", ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK=False)
def test_generate_review_topics_perplexity_when_key_set(monkeypatch):
    called = {"gemini": 0, "pplx": 0}

    def no_gemini(**_kw):
        called["gemini"] += 1
        raise AssertionError("Gemini must not run when Perplexity key is set")

    payload = {"topics": [{"topic": " Widget Sales ", "category": "product", "rationale": "x"}]}

    def fake_pplx(**kw):
        called["pplx"] += 1
        assert kw.get("log_operation") == "perplexity.chat.completions.onboarding_review_topics"
        return json.dumps(payload), ""

    monkeypatch.setattr("accounts.onboarding_review_topics.generate_gemini_execution_text", no_gemini)
    monkeypatch.setattr("accounts.onboarding_review_topics.perplexity_chat_completion_raw", fake_pplx)

    items, err = generate_review_topics_for_domain(domain="acme.com", business_profile=None)
    assert err == ""
    assert called == {"gemini": 0, "pplx": 1}
    assert len(items) == 1
    assert items[0]["topic"] == "Widget Sales"
    assert items[0]["category"] == "product"


@override_settings(PERPLEXITY_API_KEY="", ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK=False)
def test_generate_review_topics_no_perplexity_key_and_no_fallback():
    items, err = generate_review_topics_for_domain(domain="acme.com", business_profile=None)
    assert items == []
    assert err == "review_topics_perplexity_not_configured"


@override_settings(PERPLEXITY_API_KEY="", ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK=True)
def test_generate_review_topics_gemini_fallback_when_configured(monkeypatch):
    def fake_gemini(**_kw):
        return json.dumps({"topics": [{"topic": "From Gemini", "category": "service"}]}), None

    monkeypatch.setattr("accounts.onboarding_review_topics.generate_gemini_execution_text", fake_gemini)

    items, err = generate_review_topics_for_domain(domain="example.org", business_profile=None)
    assert err == ""
    assert len(items) == 1
    assert items[0]["topic"] == "From Gemini"
