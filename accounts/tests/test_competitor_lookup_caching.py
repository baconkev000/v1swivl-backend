import pytest

from accounts import dataforseo_utils as d


@pytest.mark.django_db
def test_competitor_domains_empty_response_cached(monkeypatch):
    d.cache.clear()
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(d, "_post", fake_post)
    monkeypatch.setattr(d.settings, "DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS", False)

    first = d._get_competitor_domains("example.com", location_code=2840, language_code="en", limit=5)
    second = d._get_competitor_domains("example.com", location_code=2840, language_code="en", limit=5)

    assert first == []
    assert second == []
    assert calls["n"] == 1


@pytest.mark.django_db
def test_competitor_domains_exception_cached(monkeypatch):
    d.cache.clear()
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return {"tasks": [{"result": [{"items": [None]}]}]}

    monkeypatch.setattr(d, "_post", fake_post)
    monkeypatch.setattr(d.settings, "DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS", False)

    first = d._get_competitor_domains("example.com", location_code=2840, language_code="en", limit=5)
    second = d._get_competitor_domains("example.com", location_code=2840, language_code="en", limit=5)

    assert first == []
    assert second == []
    assert calls["n"] == 1


@pytest.mark.django_db
def test_competitor_avg_traffic_empty_response_cached(monkeypatch):
    d.cache.clear()
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(d, "_post", fake_post)
    monkeypatch.setattr(d.settings, "DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS", False)

    first = d._get_competitor_average_traffic("example.com", location_code=2840, language_code="en")
    second = d._get_competitor_average_traffic("example.com", location_code=2840, language_code="en")

    assert first == 0.0
    assert second == 0.0
    assert calls["n"] == 1


@pytest.mark.django_db
def test_competitor_avg_traffic_exception_cached(monkeypatch):
    d.cache.clear()
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return {"tasks": [{"result": [{"items": [None]}]}]}

    monkeypatch.setattr(d, "_post", fake_post)
    monkeypatch.setattr(d.settings, "DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS", False)

    first = d._get_competitor_average_traffic("example.com", location_code=2840, language_code="en")
    second = d._get_competitor_average_traffic("example.com", location_code=2840, language_code="en")

    assert first == 0.0
    assert second == 0.0
    assert calls["n"] == 1


@pytest.mark.django_db
def test_override_skips_auto_competitor_lookup(monkeypatch):
    calls = {"n": 0}

    def fake_get_competitor_domains(*_args, **_kwargs):
        calls["n"] += 1
        return ["should-not-be-used.com"]

    monkeypatch.setattr(d, "_get_competitor_domains", fake_get_competitor_domains)

    class P:
        industry = ""
        business_address = ""
        seo_competitor_domains_override = "foo.com,bar.com"

    result = d.get_competitors_for_domain_intersection(
        domain="example.com",
        location_code=2840,
        language_code="en",
        profile=P(),
    )

    assert calls["n"] == 0
    assert result["filtered_competitors_used"] == ["foo.com", "bar.com"]
    assert result["competitor_source"] == "profile_override"


@pytest.mark.django_db
def test_kill_switch_skips_competitor_post_calls(monkeypatch):
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return {"tasks": [{"result": [{"items": [{"domain": "a.com"}]}]}]}

    monkeypatch.setattr(d, "_post", fake_post)
    monkeypatch.setattr(d.settings, "DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS", True)

    domains = d._get_competitor_domains("example.com", location_code=2840, language_code="en", limit=5)
    avg = d._get_competitor_average_traffic("example.com", location_code=2840, language_code="en")

    assert domains == []
    assert avg == 0.0
    assert calls["n"] == 0


@pytest.mark.django_db
def test_reddit_dropped_from_labs_competitors_and_from_profile_override(monkeypatch):
    """reddit.com is excluded so it is not passed to domain_intersection."""

    def fake_get_competitor_domains(*_args, **_kwargs):
        return ["reddit.com", "real-competitor.example", "other-competitor.example"]

    monkeypatch.setattr(d, "_get_competitor_domains", fake_get_competitor_domains)

    class NoOverride:
        industry = "cookies"
        business_address = ""
        seo_competitor_domains_override = ""

    auto = d.get_competitors_for_domain_intersection(
        domain="cravecookies.com",
        location_code=2840,
        language_code="en",
        profile=NoOverride(),
        limit=5,
        min_competitors=2,
    )
    assert "reddit.com" in auto["raw_competitors"]
    assert "reddit.com" not in auto["filtered_competitors_used"]
    assert auto["competitor_source"] == "filtered_auto"

    class WithOverride:
        industry = ""
        business_address = ""
        seo_competitor_domains_override = "reddit.com,override-only.example"

    over = d.get_competitors_for_domain_intersection(
        domain="example.com",
        location_code=2840,
        language_code="en",
        profile=WithOverride(),
        limit=5,
        min_competitors=2,
    )
    assert over["competitor_source"] == "profile_override"
    assert over["filtered_competitors_used"] == ["override-only.example"]
