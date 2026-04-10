import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import BusinessProfile

User = get_user_model()


@pytest.mark.django_db
def test_checkout_identity_returns_main_profile():
    user = User.objects.create_user(username="checkoutid@example.com", email="checkoutid@example.com", password="x")
    BusinessProfile.objects.create(user=user, is_main=False, business_name="Old")
    main = BusinessProfile.objects.create(user=user, is_main=True, business_name="Main")
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/business-profile/checkout-identity/")
    assert res.status_code == 200
    data = res.json()
    assert data["profile_id"] == main.id
    assert data["user_id"] == user.id
    assert data["email"] == user.email
    assert data["is_main"] is True

