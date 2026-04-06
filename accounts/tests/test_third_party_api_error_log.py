"""ThirdPartyApiErrorLog + record_third_party_api_error (failure persistence)."""

import pytest
from django.contrib.auth import get_user_model

from accounts.models import BusinessProfile, ThirdPartyApiErrorLog, ThirdPartyApiProvider
from accounts.third_party_usage import (
    classify_openai_sdk_exception,
    record_third_party_api_error,
)

User = get_user_model()


def test_record_third_party_api_error_swallows_inner_failure(monkeypatch):
    """Never break callers if the error log row cannot be written."""

    def boom(**kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("accounts.models.ThirdPartyApiErrorLog.objects.create", boom)
    record_third_party_api_error(
        provider=ThirdPartyApiProvider.OPENAI,
        operation="openai.chat.test",
        error_kind=ThirdPartyApiErrorLog.ErrorKind.UNKNOWN_EXCEPTION,
        message="boom",
    )


@pytest.mark.django_db
def test_record_third_party_api_error_truncates_detail():
    long_detail = "x" * 20_000
    record_third_party_api_error(
        provider=ThirdPartyApiProvider.OPENAI,
        operation="op",
        error_kind=ThirdPartyApiErrorLog.ErrorKind.UNKNOWN_EXCEPTION,
        message="m",
        detail=long_detail,
    )
    row = ThirdPartyApiErrorLog.objects.get()
    assert len(row.detail) < len(long_detail)
    assert "truncated" in row.detail.lower()


@pytest.mark.django_db
def test_perplexity_http_error_writes_one_error_row(settings, monkeypatch):
    settings.PERPLEXITY_API_KEY = "test-key"
    settings.PERPLEXITY_AEO_MODEL = "sonar"
    user = User.objects.create_user(username="e1", email="e1@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Biz")

    class FakeResp:
        status_code = 404
        content = b"{}"
        text = '{"not_found": true}'

        def json(self):
            return {"error": "model_not_found"}

    monkeypatch.setattr(
        "accounts.aeo.perplexity_execution_utils.requests.post",
        lambda **kw: FakeResp(),
    )
    monkeypatch.setattr(
        "accounts.aeo.perplexity_execution_utils.record_perplexity_request",
        lambda **kw: None,
    )

    from accounts.aeo.perplexity_execution_utils import run_single_aeo_prompt_perplexity

    out = run_single_aeo_prompt_perplexity({"prompt": "hello"}, profile, save=False)
    assert out["success"] is False
    rows = list(ThirdPartyApiErrorLog.objects.filter(provider=ThirdPartyApiProvider.PERPLEXITY))
    assert len(rows) == 1
    assert rows[0].http_status == 404
    assert rows[0].error_kind == ThirdPartyApiErrorLog.ErrorKind.HTTP_ERROR


@pytest.mark.django_db
def test_dataforseo_non_200_writes_error_row(monkeypatch):
    class FakeResp:
        status_code = 502
        content = b"bad gateway"
        text = "bad gateway"

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(
        "accounts.dataforseo_utils.requests.post",
        lambda **kw: FakeResp(),
    )
    monkeypatch.setattr("accounts.dataforseo_utils._get_auth", lambda: ("u", "p"))

    from accounts import dataforseo_utils

    out = dataforseo_utils._post("/v3/test/path", [{}], business_profile=None)
    assert out is None
    row = ThirdPartyApiErrorLog.objects.get()
    assert row.provider == ThirdPartyApiProvider.DATAFORSEO
    assert row.http_status == 502
    assert row.error_kind == ThirdPartyApiErrorLog.ErrorKind.HTTP_ERROR


def test_classify_openai_maps_timeout_by_name():
    class FakeTimeout(Exception):
        pass

    kind, http = classify_openai_sdk_exception(FakeTimeout("slow"))
    assert kind == ThirdPartyApiErrorLog.ErrorKind.TIMEOUT
