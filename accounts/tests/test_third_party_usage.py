from decimal import Decimal
from unittest.mock import MagicMock

from accounts.models import ThirdPartyApiRequestLog
from accounts.third_party_usage import record_dataforseo_request


def test_record_dataforseo_request_sums_task_costs(monkeypatch):
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(ThirdPartyApiRequestLog.objects, "create", fake_create)

    record_dataforseo_request(
        operation="/v3/dataforseo_labs/google/example/live",
        response_json={
            "tasks": [
                {"cost": 0.01},
                {"cost": 0.02},
            ],
        },
        business_profile=None,
    )

    assert captured["provider"] == ThirdPartyApiRequestLog.Provider.DATAFORSEO
    assert captured["cost_usd"] == Decimal("0.03")
    assert captured["operation"] == "/v3/dataforseo_labs/google/example/live"


def test_record_dataforseo_request_null_cost_when_absent(monkeypatch):
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(ThirdPartyApiRequestLog.objects, "create", fake_create)

    record_dataforseo_request(
        operation="/v3/foo/live",
        response_json={"tasks": [{"status_code": 20000}]},
        business_profile=None,
    )

    assert captured["cost_usd"] is None
