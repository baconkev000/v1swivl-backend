import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import BusinessProfile, BusinessProfileMembership, ThirdPartyApiErrorLog

User = get_user_model()


class _StripeLikeObject:
    def __init__(self, payload):
        self._payload = payload

    def _to_dict_recursive(self):
        return self._payload


def _mk_user_profile() -> tuple[object, BusinessProfile]:
    user = User.objects.create_user(
        username="bill_pm@example.com",
        email="bill_pm@example.com",
        password="pw",
    )
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Billing Co",
        stripe_customer_id="cus_test_pm",
        stripe_subscription_id="sub_test_pm",
        plan=BusinessProfile.PLAN_PRO,
    )
    return user, profile


@pytest.mark.django_db
def test_billing_summary_payment_method_from_subscription_default(monkeypatch, settings):
    pytest.importorskip("stripe")
    settings.STRIPE_SECRET_KEY = "sk_test_123"
    user, _profile = _mk_user_profile()

    monkeypatch.setattr(
        "accounts.views.stripe.Subscription.retrieve",
        lambda *_a, **_k: {
            "items": {
                "data": [
                    {
                        "price": {
                            "unit_amount": 4900,
                            "currency": "usd",
                            "recurring": {"interval": "month", "interval_count": 1},
                        }
                    }
                ]
            },
            "default_payment_method": {
                "card": {
                    "brand": "visa",
                    "last4": "4242",
                    "exp_month": 9,
                    "exp_year": 2029,
                    "funding": "credit",
                }
            },
        },
    )
    monkeypatch.setattr("accounts.views.stripe.Subscription.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.Customer.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.Invoice.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.PaymentMethod.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.PaymentIntent.retrieve", lambda *_a, **_k: {})

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/billing/summary/")
    assert res.status_code == 200
    body = res.json()
    assert body["payment_method"] == {
        "brand": "visa",
        "last4": "4242",
        "exp_month": 9,
        "exp_year": 2029,
        "funding": "credit",
    }
    assert "invoices" in body
    assert "plan_label" in body


@pytest.mark.django_db
def test_billing_summary_payment_method_falls_back_to_customer_default(monkeypatch, settings):
    pytest.importorskip("stripe")
    settings.STRIPE_SECRET_KEY = "sk_test_123"
    user, _profile = _mk_user_profile()

    monkeypatch.setattr(
        "accounts.views.stripe.Subscription.retrieve",
        lambda *_a, **_k: {
            "items": {
                "data": [
                    {
                        "price": {
                            "unit_amount": 4900,
                            "currency": "usd",
                            "recurring": {"interval": "month", "interval_count": 1},
                        }
                    }
                ]
            },
            "default_payment_method": None,
        },
    )
    monkeypatch.setattr("accounts.views.stripe.Subscription.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr(
        "accounts.views.stripe.Customer.retrieve",
        lambda *_a, **_k: {
            "invoice_settings": {
                "default_payment_method": {
                    "card": {
                        "brand": "mastercard",
                        "last4": "4444",
                        "exp_month": 12,
                        "exp_year": 2030,
                        "funding": "debit",
                    }
                }
            }
        },
    )
    monkeypatch.setattr("accounts.views.stripe.Invoice.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.PaymentMethod.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.PaymentIntent.retrieve", lambda *_a, **_k: {})

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/billing/summary/")
    assert res.status_code == 200
    body = res.json()
    assert body["payment_method"] == {
        "brand": "mastercard",
        "last4": "4444",
        "exp_month": 12,
        "exp_year": 2030,
        "funding": "debit",
    }


@pytest.mark.django_db
def test_billing_summary_payment_method_null_when_missing_everywhere(monkeypatch, settings):
    pytest.importorskip("stripe")
    settings.STRIPE_SECRET_KEY = "sk_test_123"
    user, _profile = _mk_user_profile()

    monkeypatch.setattr(
        "accounts.views.stripe.Subscription.retrieve",
        lambda *_a, **_k: {"items": {"data": []}, "default_payment_method": None},
    )
    monkeypatch.setattr("accounts.views.stripe.Subscription.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr(
        "accounts.views.stripe.Customer.retrieve",
        lambda *_a, **_k: {"invoice_settings": {"default_payment_method": None}},
    )
    monkeypatch.setattr(
        "accounts.views.stripe.Invoice.list",
        lambda *_a, **_k: {"data": [{"status": "paid", "payment_intent": None}]},
    )
    monkeypatch.setattr("accounts.views.stripe.PaymentMethod.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.PaymentIntent.retrieve", lambda *_a, **_k: {})

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/billing/summary/")
    assert res.status_code == 200
    body = res.json()
    assert body["payment_method"] is None


@pytest.mark.django_db
def test_billing_summary_reads_monthly_price_from_legacy_plan_shape(monkeypatch, settings):
    pytest.importorskip("stripe")
    settings.STRIPE_SECRET_KEY = "sk_test_123"
    user, _profile = _mk_user_profile()

    monkeypatch.setattr(
        "accounts.views.stripe.Subscription.retrieve",
        lambda *_a, **_k: {
            "items": {
                "data": [
                    {
                        "plan": {
                            "unit_amount": 12900,
                            "currency": "usd",
                            "recurring": {"interval": "month", "interval_count": 1},
                        }
                    }
                ]
            },
            "default_payment_method": None,
            "current_period_end": 1893456000,
        },
    )
    monkeypatch.setattr("accounts.views.stripe.Subscription.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.Customer.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.Invoice.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.PaymentMethod.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.PaymentIntent.retrieve", lambda *_a, **_k: {})

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/billing/summary/")
    assert res.status_code == 200
    body = res.json()
    assert body["price_month"] == "129.00"
    assert body["currency"] == "usd"
    assert body["renewal_date"] is not None


@pytest.mark.django_db
def test_billing_summary_uses_subscription_scoped_invoice_fallback(monkeypatch, settings):
    pytest.importorskip("stripe")
    settings.STRIPE_SECRET_KEY = "sk_test_123"
    user, _profile = _mk_user_profile()

    monkeypatch.setattr(
        "accounts.views.stripe.Subscription.retrieve",
        lambda *_a, **_k: {
            "items": {
                "data": [
                    {
                        "price": {
                            "unit_amount": 4900,
                            "currency": "usd",
                            "recurring": {"interval": "month", "interval_count": 1},
                        }
                    }
                ]
            },
            "default_payment_method": None,
        },
    )
    monkeypatch.setattr("accounts.views.stripe.Subscription.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.Customer.retrieve", lambda *_a, **_k: {})

    def _invoice_list(*_a, **kwargs):
        if kwargs.get("customer"):
            return {"data": []}
        if kwargs.get("subscription"):
            return {
                "data": [
                    {
                        "id": "in_123",
                        "number": "INV-123",
                        "created": 1704067200,
                        "amount_paid": 4900,
                        "currency": "usd",
                        "status": "paid",
                        "status_transitions": {"paid_at": 1704067300},
                    }
                ]
            }
        return {"data": []}

    monkeypatch.setattr("accounts.views.stripe.Invoice.list", _invoice_list)
    monkeypatch.setattr("accounts.views.stripe.PaymentMethod.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.PaymentIntent.retrieve", lambda *_a, **_k: {})

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/billing/summary/")
    assert res.status_code == 200
    body = res.json()
    assert len(body["invoices"]) == 1
    assert body["invoices"][0]["id"] == "INV-123"


@pytest.mark.django_db
def test_billing_summary_falls_back_to_owned_profile_with_customer_id(monkeypatch, settings):
    pytest.importorskip("stripe")
    settings.STRIPE_SECRET_KEY = "sk_test_123"

    owner = User.objects.create_user(username="owner_main@example.com", email="owner_main@example.com", password="pw")
    external_owner = User.objects.create_user(username="other@example.com", email="other@example.com", password="pw")
    owned = BusinessProfile.objects.create(
        user=owner,
        is_main=True,
        business_name="Owned Billing",
        stripe_customer_id="cus_owned_real",
        stripe_subscription_id="sub_owned_real",
        plan=BusinessProfile.PLAN_PRO,
    )
    external = BusinessProfile.objects.create(
        user=external_owner,
        is_main=True,
        business_name="External Workspace",
        stripe_customer_id="",
        stripe_subscription_id="",
        plan=BusinessProfile.PLAN_PRO,
    )
    BusinessProfileMembership.objects.create(
        business_profile=external,
        user=owner,
        role=BusinessProfileMembership.ROLE_ADMIN,
        is_owner=False,
        hidden_from_team_ui=False,
    )

    seen_customers: list[str] = []

    monkeypatch.setattr(
        "accounts.views.stripe.Subscription.retrieve",
        lambda *_a, **_k: {
            "items": {"data": [{"price": {"unit_amount": 4900, "currency": "usd", "recurring": {"interval": "month"}}}]},
            "default_payment_method": None,
        },
    )
    monkeypatch.setattr("accounts.views.stripe.Subscription.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.Customer.retrieve", lambda *_a, **_k: {})

    def _invoice_list(*_a, **kwargs):
        if kwargs.get("customer"):
            seen_customers.append(str(kwargs.get("customer")))
        return {"data": []}

    monkeypatch.setattr("accounts.views.stripe.Invoice.list", _invoice_list)
    monkeypatch.setattr("accounts.views.stripe.PaymentMethod.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.PaymentIntent.retrieve", lambda *_a, **_k: {})

    client = APIClient()
    client.force_authenticate(user=owner)
    res = client.get("/api/billing/summary/")
    assert res.status_code == 200
    body = res.json()
    assert body["plan_label"] == "Pro"
    assert seen_customers == ["cus_owned_real"]
    owned.refresh_from_db()
    assert owned.id != external.id


@pytest.mark.django_db
def test_billing_summary_records_third_party_error_on_stripe_failure(monkeypatch, settings):
    pytest.importorskip("stripe")
    settings.STRIPE_SECRET_KEY = "sk_test_123"
    user, _profile = _mk_user_profile()

    def _boom(*_a, **_k):
        raise RuntimeError("stripe unavailable")

    monkeypatch.setattr("accounts.views.stripe.Subscription.retrieve", _boom)
    monkeypatch.setattr("accounts.views.stripe.Subscription.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.Customer.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.Invoice.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.PaymentMethod.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.PaymentIntent.retrieve", lambda *_a, **_k: {})

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/billing/summary/")
    assert res.status_code == 200
    row = ThirdPartyApiErrorLog.objects.filter(provider="stripe").order_by("-id").first()
    assert row is not None
    assert row.operation == "stripe.subscription.retrieve.billing_summary"
    assert row.error_kind == ThirdPartyApiErrorLog.ErrorKind.UNKNOWN_EXCEPTION


@pytest.mark.django_db
def test_billing_summary_handles_stripe_sdk_object_payloads(monkeypatch, settings):
    pytest.importorskip("stripe")
    settings.STRIPE_SECRET_KEY = "sk_test_123"
    user, _profile = _mk_user_profile()

    monkeypatch.setattr(
        "accounts.views.stripe.Subscription.retrieve",
        lambda *_a, **_k: _StripeLikeObject(
            {
                "items": {
                    "data": [
                        {
                            "price": {
                                "unit_amount": 4900,
                                "currency": "usd",
                                "recurring": {"interval": "month", "interval_count": 1},
                            }
                        }
                    ]
                },
                "default_payment_method": _StripeLikeObject(
                    {
                        "card": {
                            "brand": "visa",
                            "last4": "4242",
                            "exp_month": 9,
                            "exp_year": 2030,
                            "funding": "credit",
                        }
                    }
                ),
                "current_period_end": 1893456000,
            }
        ),
    )
    monkeypatch.setattr("accounts.views.stripe.Subscription.list", lambda *_a, **_k: {"data": []})
    monkeypatch.setattr("accounts.views.stripe.Customer.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr(
        "accounts.views.stripe.Invoice.list",
        lambda *_a, **_k: _StripeLikeObject(
            {
                "data": [
                    {
                        "id": "in_1",
                        "number": "INV-1",
                        "created": 1704067200,
                        "amount_paid": 4900,
                        "currency": "usd",
                        "status": "paid",
                        "status_transitions": {"paid_at": 1704067300},
                    }
                ]
            }
        ),
    )
    monkeypatch.setattr("accounts.views.stripe.PaymentMethod.retrieve", lambda *_a, **_k: {})
    monkeypatch.setattr("accounts.views.stripe.PaymentIntent.retrieve", lambda *_a, **_k: {})

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/billing/summary/")
    assert res.status_code == 200
    body = res.json()
    assert body["price_month"] == "49.00"
    assert body["payment_method"] is not None
    assert body["payment_method"]["brand"] == "visa"
    assert len(body["invoices"]) == 1
