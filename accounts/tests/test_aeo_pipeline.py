import pytest
from django.core.cache import cache

from accounts.dataforseo_utils import get_aeo_content_readiness_for_site, _crawl_pages_for_aeo


@pytest.mark.django_db
def test_aeo_strong_technical_weak_demand(monkeypatch):
    # Weak demand coverage: high-volume question missing, low-volume found.
    monkeypatch.setattr(
        "accounts.dataforseo_utils.get_question_intent_keywords",
        lambda **kwargs: [
            {"keyword": "how to choose invisalign provider", "search_volume": 1000},
            {"keyword": "what is invisalign", "search_volume": 100},
        ],
    )
    # Strong technical signals.
    monkeypatch.setattr(
        "accounts.dataforseo_utils._crawl_pages_for_aeo",
        lambda **kwargs: [
            {
                "title": "FAQ - White Pine Dental",
                "headings": [{"text": "What is Invisalign?"}, {"text": "Frequently Asked Questions"}],
                "content": {
                    "@type": "FAQPage",
                    "text": "What is Invisalign? Invisalign is a clear aligner treatment used to straighten teeth.",
                },
            }
        ],
    )

    result = get_aeo_content_readiness_for_site(
        target_domain="whitepinedentalcare.com",
        niche="dental",
        force_refresh=True,
    )

    # Technical should be strong.
    assert result["faq_readiness_score"] >= 60
    assert result["snippet_readiness_score"] >= 60
    # Demand should be weak due to missing high-volume phrase.
    assert result["question_coverage_score"] < 50


@pytest.mark.django_db
def test_aeo_weak_technical_strong_demand(monkeypatch):
    monkeypatch.setattr(
        "accounts.dataforseo_utils.get_question_intent_keywords",
        lambda **kwargs: [
            {"keyword": "how to choose invisalign provider", "search_volume": 1000},
            {"keyword": "what is invisalign", "search_volume": 100},
        ],
    )
    # Demand text present, technical structure weak.
    monkeypatch.setattr(
        "accounts.dataforseo_utils._crawl_pages_for_aeo",
        lambda **kwargs: [
            {
                "title": "Home",
                "content": "how to choose invisalign provider what is invisalign",
                "headings": [{"text": "Our Services"}],
            }
        ],
    )

    result = get_aeo_content_readiness_for_site(
        target_domain="whitepinedentalcare.com",
        niche="dental",
        force_refresh=True,
    )

    # Strong demand coverage (both phrases matched).
    assert result["question_coverage_score"] >= 95
    # Weak technical due to no FAQ schema/sections and no question heading blocks.
    assert result["faq_readiness_score"] == 0
    assert result["snippet_readiness_score"] == 0


@pytest.mark.django_db
def test_aeo_crawl_failure_returns_safe_defaults(monkeypatch):
    monkeypatch.setattr(
        "accounts.dataforseo_utils.get_question_intent_keywords",
        lambda **kwargs: [
            {"keyword": "how to choose invisalign provider", "search_volume": 1000},
        ],
    )
    # Crawl fails / no pages returned.
    monkeypatch.setattr("accounts.dataforseo_utils._crawl_pages_for_aeo", lambda **kwargs: [])

    result = get_aeo_content_readiness_for_site(
        target_domain="whitepinedentalcare.com",
        niche="dental",
        force_refresh=True,
    )

    assert result["faq_readiness_score"] == 0
    assert result["snippet_readiness_score"] == 0
    assert result["question_coverage_score"] == 0
    assert isinstance(result["questions_missing"], list)


def test_crawl_homepage_fallback_when_onpage_pages_empty(monkeypatch):
    calls = {"count": 0}
    cache.clear()

    def fake_post(path, payload):
        calls["count"] += 1
        if path == "/v3/on_page/task_post":
            return {"tasks": [{"id": "task-1"}]}
        if path == "/v3/on_page/pages":
            return {"tasks": [{"result": [{"items": []}]}]}
        return None

    monkeypatch.setattr("accounts.dataforseo_utils._post", fake_post)
    monkeypatch.setattr(
        "accounts.dataforseo_utils._fetch_homepage_page_for_aeo",
        lambda domain: {"url": f"https://{domain}/", "title": "Home", "content": "faq content"},
    )

    result = _crawl_pages_for_aeo(target_domain="whitepinedentalcare.com", max_pages=5, timeout_seconds=2)
    pages = result["pages"]
    assert pages and len(pages) == 1
    assert pages[0]["url"] == "https://whitepinedentalcare.com/"
    assert result["exit_reason"] == "fallback_used"


def test_crawl_finishes_quickly_returns_pages(monkeypatch):
    cache.clear()

    def fake_post(path, payload):
        if path == "/v3/on_page/task_post":
            return {"tasks": [{"id": "task-quick"}]}
        if path == "/v3/on_page/pages":
            return {
                "tasks": [
                    {
                        "status_code": 20000,
                        "result": [{"items": [{"url": "https://site/a"}, {"url": "https://site/b"}]}],
                    }
                ]
            }
        return None

    monkeypatch.setattr("accounts.dataforseo_utils._post", fake_post)
    result = _crawl_pages_for_aeo(target_domain="quick.example.com", max_pages=20, timeout_seconds=5)
    assert result["aeo_status"] == "ready"
    assert result["exit_reason"] == "finished_with_pages"
    assert len(result["pages"]) == 2


def test_crawl_finishes_near_timeout_still_returns_pages(monkeypatch):
    cache.clear()
    state = {"pages_calls": 0}
    fake_now = {"t": 0.0}

    def fake_monotonic():
        return fake_now["t"]

    def fake_sleep(seconds):
        fake_now["t"] += float(seconds)

    def fake_post(path, payload):
        if path == "/v3/on_page/task_post":
            return {"tasks": [{"id": "task-near-timeout"}]}
        if path == "/v3/on_page/pages":
            state["pages_calls"] += 1
            if state["pages_calls"] < 4:
                return {"tasks": [{"status_code": 20000, "result": [{"items": []}]}]}
            return {"tasks": [{"status_code": 20000, "result": [{"items": [{"url": "https://site/final"}]}]}]}
        return None

    monkeypatch.setattr("accounts.dataforseo_utils.time.monotonic", fake_monotonic)
    monkeypatch.setattr("accounts.dataforseo_utils.time.sleep", fake_sleep)
    monkeypatch.setattr("accounts.dataforseo_utils._post", fake_post)
    result = _crawl_pages_for_aeo(target_domain="near.example.com", timeout_seconds=30)
    assert result["aeo_status"] == "ready"
    assert result["exit_reason"] == "finished_with_pages"
    assert len(result["pages"]) == 1


def test_crawl_timeout_returns_timed_out_without_homepage_fallback(monkeypatch):
    cache.clear()
    fake_now = {"t": 0.0}
    homepage_calls = {"count": 0}

    def fake_monotonic():
        return fake_now["t"]

    def fake_sleep(seconds):
        fake_now["t"] += float(seconds)

    def fake_post(path, payload):
        if path == "/v3/on_page/task_post":
            return {"tasks": [{"id": "task-timeout"}]}
        if path == "/v3/on_page/pages":
            return {"tasks": [{"status_code": 20000, "result": [{"items": []}]}]}
        return None

    monkeypatch.setattr("accounts.dataforseo_utils.time.monotonic", fake_monotonic)
    monkeypatch.setattr("accounts.dataforseo_utils.time.sleep", fake_sleep)
    monkeypatch.setattr("accounts.dataforseo_utils._post", fake_post)
    monkeypatch.setattr(
        "accounts.dataforseo_utils._fetch_homepage_page_for_aeo",
        lambda domain: homepage_calls.__setitem__("count", homepage_calls["count"] + 1),
    )
    result = _crawl_pages_for_aeo(target_domain="timeout.example.com", timeout_seconds=5)
    assert result["aeo_status"] == "timed_out"
    assert result["exit_reason"] == "timeout"
    assert result["pages"] == []
    assert homepage_calls["count"] == 0


def test_crawl_cooldown_reuses_task_no_duplicate_task_post(monkeypatch):
    cache.clear()
    calls = {"task_post": 0, "pages": 0}
    fake_now = {"t": 0.0}

    def fake_monotonic():
        return fake_now["t"]

    def fake_sleep(seconds):
        fake_now["t"] += float(seconds)

    def fake_post(path, payload):
        if path == "/v3/on_page/task_post":
            calls["task_post"] += 1
            return {"tasks": [{"id": "task-shared"}]}
        if path == "/v3/on_page/pages":
            calls["pages"] += 1
            return {"tasks": [{"status_code": 20000, "result": [{"items": []}]}]}
        return None

    monkeypatch.setattr("accounts.dataforseo_utils.time.monotonic", fake_monotonic)
    monkeypatch.setattr("accounts.dataforseo_utils.time.sleep", fake_sleep)
    monkeypatch.setattr("accounts.dataforseo_utils._post", fake_post)

    _crawl_pages_for_aeo(target_domain="shared.example.com", crawl_scope="profile:1", timeout_seconds=3)
    fake_now["t"] = 0.0
    _crawl_pages_for_aeo(target_domain="shared.example.com", crawl_scope="profile:1", timeout_seconds=3)
    assert calls["task_post"] == 1
