"""Tests for AEO wrong-URL classification and domain reachability (mocked network)."""

import socket
from types import SimpleNamespace

import pytest

from accounts.aeo import domain_verification
from accounts.aeo.aeo_extraction_utils import (
    canonical_registrable_domain,
    competitor_attributed_noncanonical_url,
    save_extraction_result,
)
from accounts.models import AEOExtractionSnapshot


def test_canonical_registrable_domain_strips_www_and_path():
    assert canonical_registrable_domain("https://WWW.Example.COM/about") == "example.com"


def test_competitor_attributed_noncanonical_url_finds_wrong_host():
    url = competitor_attributed_noncanonical_url(
        "Acme Dental",
        "acme.com",
        [{"name": "Acme Dental", "url": "https://other-clinic.com"}],
    )
    assert url == "https://other-clinic.com"


def test_competitor_attributed_skips_when_domain_matches():
    assert (
        competitor_attributed_noncanonical_url(
            "Acme Dental",
            "acme.com",
            [{"name": "Acme Dental", "url": "https://www.acme.com"}],
        )
        is None
    )


def test_normalize_request_url_rejects_non_http():
    assert domain_verification._normalize_request_url("ftp://evil.com/") is None


def test_verify_url_blocks_non_global_ip(monkeypatch):
    monkeypatch.setattr(
        domain_verification,
        "_resolve_hostname_ips",
        lambda hostname, timeout_s: ["10.0.0.1"],
    )

    r = domain_verification.verify_url_reachable("https://example.com/")
    assert r.dns_ok is True
    assert r.http_ok is False
    assert "ssrf" in r.error


def test_verify_url_dns_failure(monkeypatch):
    def _fail(hostname, timeout_s):
        raise socket.gaierror("nxdomain")

    monkeypatch.setattr(domain_verification, "_resolve_hostname_ips", _fail)
    r = domain_verification.verify_url_reachable("https://does-not-exist-12345.invalid/")
    assert r.dns_ok is False
    assert r.http_ok is False
    assert "dns" in r.error.lower()


class _FakeResp:
    def __init__(self, code: int = 200, final: str = "https://example.com/"):
        self._code = code
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    @property
    def status(self):
        return self._code

    def getcode(self):
        return self._code

    def geturl(self):
        return self._final

    def read(self, n=-1):
        return b""


class _FakeOpener:
    def __init__(self, code: int = 200):
        self._code = code
        self.addheaders = []

    def open(self, req, timeout=None):
        return _FakeResp(self._code)


def test_verify_url_head_ok(monkeypatch):
    monkeypatch.setattr(
        domain_verification,
        "_resolve_hostname_ips",
        lambda hostname, timeout_s: ["8.8.8.8"],
    )
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *a, **k: _FakeOpener(200),
    )
    r = domain_verification.verify_url_reachable("https://example.com/")
    assert r.dns_ok is True
    assert r.http_ok is True
    assert r.status_code == 200


def test_resolve_brand_url_status_matched():
    out = domain_verification.resolve_brand_url_status_fields(
        brand_mentioned=True,
        tracked_root="acme.com",
        attributed_wrong_url="https://wrong.com",
    )
    assert out["brand_mentioned_url_status"] == AEOExtractionSnapshot.URL_STATUS_MATCHED
    assert out["cited_domain_or_url"] == "acme.com"


def test_save_extraction_wrong_url_live(monkeypatch, settings):
    """ORM-free: patch create() so SQLite/Postgres migration quirks do not affect this suite."""
    settings.AEO_DOMAIN_VERIFY_ENABLED = True
    profile = SimpleNamespace(business_name="Acme Co", website_url="https://acme.com")
    rsp = SimpleNamespace(profile=profile, raw_response="Generic dental services in the area.")
    captured: dict = {}

    def fake_create(**kw):
        captured.update(kw)
        return SimpleNamespace(**kw)

    monkeypatch.setattr(
        "accounts.aeo.aeo_extraction_utils.AEOExtractionSnapshot.objects.create",
        fake_create,
    )
    monkeypatch.setattr(
        domain_verification,
        "verify_url_reachable",
        lambda url, **kw: domain_verification.UrlReachabilityResult(
            dns_ok=True,
            http_ok=True,
            final_url="https://wrong.com/",
            status_code=200,
            error="",
        ),
    )
    row = save_extraction_result(
        response_snapshot=rsp,
        data={
            "competitors": [{"name": "Acme Co", "url": "https://wrong.com"}],
            "citations": [],
            "sentiment": "neutral",
            "confidence_score": None,
        },
        extraction_model="unit",
        parse_failed=False,
    )
    assert row.brand_mentioned is False
    assert captured["brand_mentioned_url_status"] == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE
    assert captured["cited_domain_or_url"] == "wrong.com"
    assert captured["verified_at"] is not None
    assert captured["url_verification_notes"].get("http_ok") is True


def test_save_extraction_wrong_url_broken(monkeypatch, settings):
    settings.AEO_DOMAIN_VERIFY_ENABLED = True
    profile = SimpleNamespace(business_name="Acme Co", website_url="https://acme.com")
    rsp = SimpleNamespace(profile=profile, raw_response="text")
    captured: dict = {}

    monkeypatch.setattr(
        "accounts.aeo.aeo_extraction_utils.AEOExtractionSnapshot.objects.create",
        lambda **kw: captured.update(kw) or SimpleNamespace(**kw),
    )
    monkeypatch.setattr(
        domain_verification,
        "verify_url_reachable",
        lambda url, **kw: domain_verification.UrlReachabilityResult(
            dns_ok=False,
            http_ok=False,
            final_url="",
            status_code=None,
            error="dns_error:gaierror",
        ),
    )
    row = save_extraction_result(
        response_snapshot=rsp,
        data={
            "competitors": [{"name": "Acme Co", "url": "https://dead.invalid"}],
            "citations": [],
            "sentiment": "neutral",
            "confidence_score": None,
        },
        extraction_model="unit",
        parse_failed=False,
    )
    assert row.brand_mentioned is False
    assert row.brand_mentioned_url_status == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN
    assert captured["brand_mentioned_url_status"] == AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN


def test_save_extraction_parse_failed_leaves_url_status_null(monkeypatch):
    profile = SimpleNamespace(business_name="Acme Co", website_url="https://acme.com")
    rsp = SimpleNamespace(profile=profile, raw_response="text")
    captured: dict = {}

    monkeypatch.setattr(
        "accounts.aeo.aeo_extraction_utils.AEOExtractionSnapshot.objects.create",
        lambda **kw: captured.update(kw) or SimpleNamespace(**kw),
    )
    save_extraction_result(
        response_snapshot=rsp,
        data={},
        extraction_model="unit",
        parse_failed=True,
    )
    assert captured["brand_mentioned_url_status"] is None
    assert captured["cited_domain_or_url"] == ""
