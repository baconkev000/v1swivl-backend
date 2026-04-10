import logging

import pytest
from django.contrib.auth import get_user_model

from accounts.models import BusinessProfile
from accounts.stripe_billing import _resolve_profile_for_event
from accounts.stripe_billing import sync_from_checkout_session

User = get_user_model()


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
    assert path == "customer"


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
        ok = sync_from_checkout_session(payload)
    assert ok is True
    assert "checkout session email mismatch" in caplog.text

