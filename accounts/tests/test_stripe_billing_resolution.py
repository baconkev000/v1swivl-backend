import logging

import pytest
from django.contrib.auth import get_user_model

from accounts.models import BusinessProfile
from accounts.stripe_billing import (
    AEO_PROMPT_PLAN_EXPANSION_POST_PAYMENT_COUNTDOWN_SECONDS,
)
from accounts.stripe_billing import StripeSyncResult
from accounts.stripe_billing import _resolve_profile_for_event
from accounts.stripe_billing import apply_subscription_payload_to_profile
from accounts.stripe_billing import sync_from_checkout_session
from accounts.stripe_billing import sync_from_invoice_paid
from accounts.stripe_billing import sync_from_subscription

User = get_user_model()


@pytest.fixture(autouse=True)
def _prevent_real_celery_from_stripe_on_commit(monkeypatch):
    """Stripe tests run on_commit handlers; avoid connecting to Redis/AMQP in CI."""
    monkeypatch.setattr(
        "accounts.tasks.post_payment_seo_snapshot_task.delay",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        lambda *a, **k: None,
    )


@pytest.fixture(autouse=True)
def _stub_stripe_subscription_retrieve_empty(monkeypatch):
    """Avoid real Stripe HTTP when webhooks pass subscription id without expanded items."""

    def _retrieve(_sid, expand=None):
        return {}

    monkeypatch.setattr("accounts.stripe_billing.stripe.Subscription.retrieve", _retrieve)


@pytest.mark.django_db
def test_resolver_prefers_client_reference_id_when_present():
    user = User.objects.create_user(username="idmatch@example.com", email="idmatch@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Acme")
    payload = {"object": {"client_reference_id": str(profile.id)}}
    resolved, path = _resolve_profile_for_event(payload)
    assert resolved is not None
    assert resolved.id == profile.id
    assert path == "client_reference_id"


@pytest.mark.django_db
def test_resolver_fallback_customer_id_match():
    user = User.objects.create_user(username="cusmatch@example.com", email="cusmatch@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Acme",
        stripe_customer_id="cus_123",
    )
    payload = {"object": {"customer": "cus_123"}}
    resolved, path = _resolve_profile_for_event(payload)
    assert resolved is not None
    assert resolved.id == profile.id
    assert path == "customer_id"


@pytest.mark.django_db
def test_resolver_fallback_email_match():
    user = User.objects.create_user(username="emailmatch@example.com", email="emailmatch@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Acme")
    payload = {"object": {"customer_details": {"email": "emailmatch@example.com"}}}
    resolved, path = _resolve_profile_for_event(payload)
    assert resolved is not None
    assert resolved.id == profile.id
    assert path == "email"


@pytest.mark.django_db
def test_checkout_session_logs_mismatch_warning(caplog):
    user_ref = User.objects.create_user(username="ref@example.com", email="ref@example.com", password="x")
    profile_ref = BusinessProfile.objects.create(user=user_ref, is_main=True, business_name="Ref")
    user_other = User.objects.create_user(username="other@example.com", email="other@example.com", password="x")
    BusinessProfile.objects.create(user=user_other, is_main=True, business_name="Other")
    payload = {
        "object": {
            "client_reference_id": str(profile_ref.id),
            "customer_details": {"email": "other@example.com"},
            "customer": "cus_zzz",
            "subscription": "sub_zzz",
        }
    }
    with caplog.at_level(logging.WARNING):
        result = sync_from_checkout_session(payload, event_id="evt_test")
    assert isinstance(result, StripeSyncResult)
    assert result.did_update is True
    assert "checkout session email mismatch" in caplog.text


@pytest.mark.django_db
def test_checkout_session_no_profile_match_returns_diagnostic():
    payload = {"object": {"customer": "cus_missing"}}
    result = sync_from_checkout_session(payload, event_id="evt_missing")
    assert isinstance(result, StripeSyncResult)
    assert result.handled is False
    assert result.did_update is False
    assert result.reason_code == "missing_profile_match"


@pytest.mark.django_db
def test_checkout_session_empty_update_payload_returns_false():
    user = User.objects.create_user(username="empty@example.com", email="empty@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Empty")
    payload = {"object": {"client_reference_id": str(profile.id)}}
    result = sync_from_checkout_session(payload, event_id="evt_empty")
    assert isinstance(result, StripeSyncResult)
    assert result.handled is False
    assert result.did_update is False
    assert result.reason_code == "empty_update_payload"


@pytest.mark.django_db
def test_new_business_profile_has_no_paid_plan_by_default():
    user = User.objects.create_user(username="noplan@example.com", email="noplan@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="N")
    profile.refresh_from_db()
    assert profile.plan == BusinessProfile.PLAN_NONE


@pytest.mark.django_db
def test_checkout_session_success_has_updated_fields():
    user = User.objects.create_user(username="ok@example.com", email="ok@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Ok")
    payload = {
        "object": {
            "client_reference_id": str(profile.id),
            "customer": "cus_ok",
            "subscription": "sub_ok",
        }
    }
    result = sync_from_checkout_session(payload, event_id="evt_ok")
    assert isinstance(result, StripeSyncResult)
    assert result.handled is True
    assert result.did_update is True
    assert len(result.updated_fields) > 0


@pytest.mark.django_db
def test_checkout_session_client_reference_promotes_profile_to_main():
    user = User.objects.create_user(username="promo@example.com", email="promo@example.com", password="x")
    main_profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Main")
    new_profile = BusinessProfile.objects.create(user=user, is_main=False, business_name="NewCo")
    payload = {
        "object": {
            "client_reference_id": str(new_profile.id),
            "customer": "cus_promo",
            "subscription": "sub_promo",
        }
    }
    result = sync_from_checkout_session(payload, event_id="evt_promo")
    assert result.handled is True
    new_profile.refresh_from_db()
    main_profile.refresh_from_db()
    assert new_profile.is_main is True
    assert main_profile.is_main is False


@pytest.mark.django_db
def test_checkout_session_sets_plan_from_expanded_subscription_price(settings):
    settings.STRIPE_PRICE_ID_STARTER_MONTHLY = "price_from_sub"
    user = User.objects.create_user(username="expsub@example.com", email="expsub@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Exp")
    payload = {
        "object": {
            "client_reference_id": str(profile.id),
            "customer": "cus_exp",
            "subscription": {
                "id": "sub_exp",
                "status": "active",
                "items": {"data": [{"price": {"id": "price_from_sub"}}]},
            },
        }
    }
    result = sync_from_checkout_session(payload, event_id="evt_exp")
    assert result.handled is True
    profile.refresh_from_db()
    assert profile.plan == BusinessProfile.PLAN_STARTER
    assert profile.stripe_subscription_status == "active"


@pytest.mark.django_db
def test_apply_subscription_pro_enqueues_expansion(monkeypatch, settings, django_capture_on_commit_callbacks):
    settings.STRIPE_PRICE_ID_PRO_MONTHLY = "price_maps_pro"
    aeo_calls: list[tuple] = []
    seo_pids: list[int] = []

    def _fake_apply_async(*args, **kwargs):
        aeo_calls.append((args, kwargs))

    def _fake_seo_delay(pid, *a, **k):
        seo_pids.append(int(pid))

    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        _fake_apply_async,
    )
    monkeypatch.setattr(
        "accounts.tasks.post_payment_seo_snapshot_task.delay",
        _fake_seo_delay,
    )
    user = User.objects.create_user(username="expq@example.com", email="expq@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Q",
        website_url="https://pro-paid.example.com",
    )
    from accounts.stripe_billing import apply_subscription_payload_to_profile

    with django_capture_on_commit_callbacks(execute=True):
        apply_subscription_payload_to_profile(
            profile,
            customer_id="cus_x",
            subscription_id="sub_x",
            status="active",
            payment_link_id="",
            price_id="price_maps_pro",
        )
    profile.refresh_from_db()
    assert profile.plan == BusinessProfile.PLAN_PRO
    assert seo_pids == [profile.id]
    assert len(aeo_calls) == 1
    opt = aeo_calls[0][1]
    assert opt.get("countdown") == AEO_PROMPT_PLAN_EXPANSION_POST_PAYMENT_COUNTDOWN_SECONDS
    assert (opt.get("args") or ())[0] == profile.id
    task_kw = opt.get("kwargs") or {}
    assert task_kw.get("expected_plan_slug") == BusinessProfile.PLAN_PRO
    assert task_kw.get("expansion_cap") == 75


@pytest.mark.django_db
def test_apply_subscription_advanced_enqueues_expansion_with_cap_100(
    monkeypatch, settings, django_capture_on_commit_callbacks
):
    settings.STRIPE_PRICE_ID_ADVANCED_MONTHLY = "price_adv_m"
    aeo_calls: list[tuple] = []
    seo_pids: list[int] = []

    def _fake_apply_async(*args, **kwargs):
        aeo_calls.append((args, kwargs))

    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        _fake_apply_async,
    )
    monkeypatch.setattr(
        "accounts.tasks.post_payment_seo_snapshot_task.delay",
        lambda pid, *a, **k: seo_pids.append(int(pid)),
    )
    user = User.objects.create_user(username="expadv@example.com", email="expadv@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Adv",
        website_url="https://adv.example.com",
    )
    with django_capture_on_commit_callbacks(execute=True):
        apply_subscription_payload_to_profile(
            profile,
            customer_id="cus_a",
            subscription_id="sub_a",
            status="active",
            price_id="price_adv_m",
        )
    profile.refresh_from_db()
    assert profile.plan == BusinessProfile.PLAN_ADVANCED
    assert seo_pids == [profile.id]
    opt = aeo_calls[0][1]
    assert opt.get("countdown") == AEO_PROMPT_PLAN_EXPANSION_POST_PAYMENT_COUNTDOWN_SECONDS
    task_kw = opt.get("kwargs") or {}
    assert task_kw.get("expected_plan_slug") == BusinessProfile.PLAN_ADVANCED
    assert task_kw.get("expansion_cap") == 150


@pytest.mark.django_db
def test_apply_subscription_starter_does_not_enqueue_expansion(
    monkeypatch, settings, django_capture_on_commit_callbacks
):
    settings.STRIPE_PRICE_ID_STARTER_MONTHLY = "price_st_only"
    aeo_calls: list[tuple] = []
    seo_pids: list[int] = []

    def _fake_apply_async(*args, **kwargs):
        aeo_calls.append((args, kwargs))

    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        _fake_apply_async,
    )
    monkeypatch.setattr(
        "accounts.tasks.post_payment_seo_snapshot_task.delay",
        lambda pid, *a, **k: seo_pids.append(int(pid)),
    )
    user = User.objects.create_user(username="expst@example.com", email="expst@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="St",
        website_url="https://starter.example.com",
    )
    with django_capture_on_commit_callbacks(execute=True):
        apply_subscription_payload_to_profile(
            profile,
            customer_id="cus_s",
            subscription_id="sub_s",
            status="active",
            price_id="price_st_only",
        )
    assert aeo_calls == []
    assert seo_pids == [profile.id]


@pytest.mark.django_db
def test_apply_subscription_no_plan_in_updates_does_not_enqueue_expansion(
    monkeypatch, settings, django_capture_on_commit_callbacks
):
    """Stripe fields only (no resolvable price / plan) — must not re-fire post-payment expansion."""
    calls: list[tuple] = []

    def _fake_apply_async(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        _fake_apply_async,
    )
    user = User.objects.create_user(username="exnoplan@example.com", email="exnoplan@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Np",
        plan=BusinessProfile.PLAN_ADVANCED,
    )
    with django_capture_on_commit_callbacks(execute=True):
        did, _fields = apply_subscription_payload_to_profile(
            profile,
            customer_id="cus_np",
            subscription_id="sub_np",
            status="active",
            price_id="",
            payment_link_id="",
        )
    assert did is True
    assert calls == []


@pytest.mark.django_db
def test_apply_subscription_empty_updates_does_not_enqueue(monkeypatch, django_capture_on_commit_callbacks):
    calls: list[tuple] = []

    def _fake_apply_async(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        _fake_apply_async,
    )
    user = User.objects.create_user(username="exempty@example.com", email="exempty@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="E")
    with django_capture_on_commit_callbacks(execute=True):
        did, _ = apply_subscription_payload_to_profile(profile)
    assert did is False
    assert calls == []


@pytest.mark.django_db
def test_checkout_session_maps_payment_link_to_plan(settings):
    settings.STRIPE_PAYMENT_LINK_PRO_MONTHLY = "plink_unit_pro_m"
    user = User.objects.create_user(username="plink@example.com", email="plink@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Pl")
    payload = {
        "object": {
            "client_reference_id": str(profile.id),
            "customer": "cus_pl",
            "subscription": "sub_pl",
            "payment_link": "plink_unit_pro_m",
        }
    }
    result = sync_from_checkout_session(payload, event_id="evt_pl")
    assert result.handled is True
    profile.refresh_from_db()
    assert profile.plan == BusinessProfile.PLAN_PRO


@pytest.mark.django_db
def test_subscription_canceled_clears_plan():
    user = User.objects.create_user(username="canceled@example.com", email="canceled@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="X",
        stripe_customer_id="cus_can",
        plan=BusinessProfile.PLAN_PRO,
        stripe_subscription_status="active",
    )
    payload = {
        "object": {
            "id": "sub_can",
            "customer": "cus_can",
            "status": "canceled",
            "cancel_at_period_end": False,
            "current_period_end": 1735689600,
            "items": {"data": [{"price": {"id": "price_any"}}]},
        }
    }
    result = sync_from_subscription(payload, event_id="evt_can")
    assert result.handled is True
    profile.refresh_from_db()
    assert profile.plan == BusinessProfile.PLAN_NONE
    assert profile.stripe_subscription_status == "canceled"


@pytest.mark.django_db
def test_apply_subscription_payload_terminal_status_clears_plan_without_price():
    user = User.objects.create_user(username="term@example.com", email="term@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="T",
        plan=BusinessProfile.PLAN_ADVANCED,
        stripe_customer_id="cus_t",
        stripe_subscription_id="sub_t",
    )
    did, fields = apply_subscription_payload_to_profile(
        profile,
        status="unpaid",
        subscription_id="sub_t",
    )
    assert did is True
    assert "plan" in fields
    profile.refresh_from_db()
    assert profile.plan == BusinessProfile.PLAN_NONE


@pytest.mark.django_db
def test_invoice_paid_updates_via_email_fallback_and_lines_price():
    user = User.objects.create_user(username="inv-email@example.com", email="inv-email@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="Inv Email")
    payload = {
        "object": {
            "customer": "cus_invoice_email",
            "subscription": "sub_invoice_email",
            "customer_details": {"email": "inv-email@example.com"},
            "lines": {"data": [{"price": {"id": "price_line_123"}}]},
        }
    }
    result = sync_from_invoice_paid(payload, event_id="evt_invoice_email")
    assert isinstance(result, StripeSyncResult)
    assert result.handled is True
    assert result.did_update is True
    assert "stripe_price_id" in result.updated_fields
    profile.refresh_from_db()
    assert profile.stripe_customer_id == "cus_invoice_email"
    assert profile.stripe_subscription_id == "sub_invoice_email"
    assert profile.stripe_price_id == "price_line_123"
    assert profile.stripe_subscription_status == "active"


@pytest.mark.django_db
def test_invoice_paid_sets_plan_when_price_matches_env(settings):
    settings.STRIPE_PRICE_ID_PRO_MONTHLY = "price_inv_pro"
    user = User.objects.create_user(username="inv-plan@example.com", email="inv-plan@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="InvP")
    payload = {
        "object": {
            "customer": "cus_inv_plan",
            "subscription": "sub_inv_plan",
            "customer_details": {"email": "inv-plan@example.com"},
            "lines": {"data": [{"price": {"id": "price_inv_pro"}}]},
        }
    }
    result = sync_from_invoice_paid(payload, event_id="evt_inv_plan")
    assert result.handled is True
    profile.refresh_from_db()
    assert profile.plan == BusinessProfile.PLAN_PRO


@pytest.mark.django_db
def test_subscription_updated_reads_items_price_id():
    user = User.objects.create_user(username="sub@example.com", email="sub@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Sub",
        stripe_customer_id="cus_sub_123",
    )
    payload = {
        "object": {
            "id": "sub_123",
            "customer": "cus_sub_123",
            "status": "active",
            "cancel_at_period_end": False,
            "current_period_end": 1735689600,
            "items": {"data": [{"price": {"id": "price_sub_123"}}]},
        }
    }
    result = sync_from_subscription(payload, event_id="evt_sub_updated")
    assert isinstance(result, StripeSyncResult)
    assert result.handled is True
    assert result.did_update is True
    assert "stripe_price_id" in result.updated_fields
    profile.refresh_from_db()
    assert profile.stripe_subscription_id == "sub_123"
    assert profile.stripe_price_id == "price_sub_123"
    assert profile.stripe_subscription_status == "active"


@pytest.mark.django_db
def test_invoice_paid_empty_update_payload_returns_reason():
    user = User.objects.create_user(username="inv-empty@example.com", email="inv-empty@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="InvEmpty")
    payload = {"object": {"customer_details": {"email": "inv-empty@example.com"}}}
    result = sync_from_invoice_paid(payload, event_id="evt_inv_empty")
    assert isinstance(result, StripeSyncResult)
    assert result.handled is False
    assert result.did_update is False
    assert result.matched_profile_id == profile.id
    assert result.reason_code == "empty_update_payload"


@pytest.mark.django_db
def test_checkout_session_resolves_plan_via_subscription_retrieve_when_price_omitted(monkeypatch, settings):
    settings.STRIPE_PRICE_ID_ADVANCED_MONTHLY = "price_from_api"

    def _retrieve(sid, expand=None):
        assert sid == "sub_checkout_scalar"
        assert expand == ["items.data.price"]
        return {"id": sid, "items": {"data": [{"price": {"id": "price_from_api"}}]}}

    monkeypatch.setattr("accounts.stripe_billing.stripe.Subscription.retrieve", _retrieve)
    user = User.objects.create_user(username="coapi@example.com", email="coapi@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="CoApi")
    payload = {
        "object": {
            "client_reference_id": str(profile.id),
            "customer": "cus_coapi",
            "subscription": "sub_checkout_scalar",
            "payment_link": "plink_unknown_not_in_env",
        }
    }
    result = sync_from_checkout_session(payload, event_id="evt_coapi")
    assert result.handled is True
    profile.refresh_from_db()
    assert profile.stripe_price_id == "price_from_api"
    assert profile.plan == BusinessProfile.PLAN_ADVANCED


@pytest.mark.django_db
def test_invoice_paid_resolves_plan_via_subscription_retrieve_when_lines_empty(monkeypatch, settings):
    settings.STRIPE_PRICE_ID_PRO_MONTHLY = "price_inv_api"

    def _retrieve(sid, expand=None):
        assert sid == "sub_inv_api"
        return {"id": sid, "items": {"data": [{"price": {"id": "price_inv_api"}}]}}

    monkeypatch.setattr("accounts.stripe_billing.stripe.Subscription.retrieve", _retrieve)
    user = User.objects.create_user(username="invapi@example.com", email="invapi@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="InvApi",
        stripe_customer_id="cus_inv_api",
    )
    payload = {
        "object": {
            "customer": "cus_inv_api",
            "subscription": "sub_inv_api",
            "customer_details": {"email": "invapi@example.com"},
        }
    }
    result = sync_from_invoice_paid(payload, event_id="evt_inv_api")
    assert result.handled is True
    profile.refresh_from_db()
    assert profile.stripe_price_id == "price_inv_api"
    assert profile.plan == BusinessProfile.PLAN_PRO


@pytest.mark.django_db
def test_apply_subscription_pro_skips_seo_without_website_still_schedules_aeo_expansion(
    monkeypatch, settings, django_capture_on_commit_callbacks
):
    settings.STRIPE_PRICE_ID_PRO_MONTHLY = "price_maps_pro"
    seo_pids: list[int] = []
    aeo_calls: list[tuple] = []

    monkeypatch.setattr(
        "accounts.tasks.post_payment_seo_snapshot_task.delay",
        lambda pid, *a, **k: seo_pids.append(int(pid)),
    )
    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        lambda *a, **kw: aeo_calls.append((a, kw)),
    )
    user = User.objects.create_user(username="pro_noweb@example.com", email="pro_noweb@example.com", password="x")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="NoWeb", website_url="")
    with django_capture_on_commit_callbacks(execute=True):
        apply_subscription_payload_to_profile(
            profile,
            customer_id="cus_nw",
            subscription_id="sub_nw",
            status="active",
            price_id="price_maps_pro",
        )
    assert seo_pids == []
    assert len(aeo_calls) == 1
    assert aeo_calls[0][1].get("countdown") == AEO_PROMPT_PLAN_EXPANSION_POST_PAYMENT_COUNTDOWN_SECONDS


@pytest.mark.django_db
def test_apply_subscription_incomplete_status_does_not_enqueue_seo(
    monkeypatch, settings, django_capture_on_commit_callbacks
):
    """Non-active status in this write must not trigger post-payment SEO (even with a website)."""
    settings.STRIPE_PRICE_ID_PRO_MONTHLY = "price_maps_pro"
    seo_pids: list[int] = []
    aeo_calls: list[tuple] = []

    monkeypatch.setattr(
        "accounts.tasks.post_payment_seo_snapshot_task.delay",
        lambda pid, *a, **k: seo_pids.append(int(pid)),
    )
    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        lambda *a, **kw: aeo_calls.append((a, kw)),
    )
    user = User.objects.create_user(username="inc@example.com", email="inc@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Inc",
        website_url="https://inc.example.com",
        plan=BusinessProfile.PLAN_STARTER,
        stripe_subscription_status="active",
    )
    with django_capture_on_commit_callbacks(execute=True):
        apply_subscription_payload_to_profile(
            profile,
            subscription_id="sub_inc",
            status="incomplete",
        )
    assert seo_pids == []
    assert aeo_calls == []


@pytest.mark.django_db
def test_post_payment_seo_webhook_enqueue_deduped_within_ttl(
    monkeypatch, settings, django_capture_on_commit_callbacks
):
    settings.STRIPE_PRICE_ID_STARTER_MONTHLY = "price_st_dedupe"
    seo_pids: list[int] = []
    monkeypatch.setattr(
        "accounts.tasks.post_payment_seo_snapshot_task.delay",
        lambda pid, *a, **k: seo_pids.append(int(pid)),
    )
    monkeypatch.setattr(
        "accounts.tasks.schedule_aeo_prompt_plan_expansion.apply_async",
        lambda *a, **k: None,
    )
    user = User.objects.create_user(username="dedupe@example.com", email="dedupe@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        website_url="https://dedupe.example.com",
    )
    with django_capture_on_commit_callbacks(execute=True):
        apply_subscription_payload_to_profile(
            profile,
            customer_id="cus_d1",
            subscription_id="sub_d1",
            status="active",
            price_id="price_st_dedupe",
        )
    with django_capture_on_commit_callbacks(execute=True):
        apply_subscription_payload_to_profile(
            profile,
            customer_id="cus_d2",
            subscription_id="sub_d2",
            status="active",
            price_id="price_st_dedupe",
        )
    assert seo_pids == [profile.id]


@pytest.mark.django_db
def test_post_payment_seo_snapshot_task_calls_run_full_with_abort_on_low_coverage(monkeypatch):
    from accounts.tasks import post_payment_seo_snapshot_task

    full_calls: list[tuple[int, dict]] = []

    def _fake_run_full(profile, **kwargs):
        full_calls.append((int(profile.id), dict(kwargs)))
        return {"ok": True, "persisted": False, "external_api_called": False}

    monkeypatch.setattr(
        "accounts.seo_snapshot_refresh.run_full_seo_snapshot_for_profile",
        _fake_run_full,
    )
    user = User.objects.create_user(username="seotask@example.com", email="seotask@example.com", password="x")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        website_url="https://seotask.example.com",
    )
    post_payment_seo_snapshot_task.run(profile.id)
    assert len(full_calls) == 1
    assert full_calls[0][0] == profile.id
    assert full_calls[0][1].get("abort_on_low_coverage") is True


def test_infer_sync_failure_reason_invoice_does_not_use_invoice_id_as_subscription():
    """Without subscription/customer/price, invoice id must not satisfy the subscription slot."""
    from accounts.stripe_billing import infer_sync_failure_reason

    payload = {
        "object": {
            "id": "in_123",
            "customer": "",
            "customer_details": {"email": "a@example.com"},
        }
    }
    assert infer_sync_failure_reason("invoice.paid", payload) == "no_stripe_ids_or_price"

