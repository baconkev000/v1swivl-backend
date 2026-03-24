import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from accounts.models import BusinessProfile


User = get_user_model()


@pytest.mark.django_db
def test_aeo_onboarding_prompt_plan_requires_auth():
    client = APIClient()
    resp = client.get("/api/aeo/onboarding-prompt-plan/")
    assert resp.status_code == 403


@pytest.mark.django_db
def test_aeo_onboarding_prompt_plan_returns_groups_and_combined():
    user = User.objects.create_user(
        username="u-aeo-onb",
        email="u-aeo-onb@example.com",
        password="pw",
    )
    BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Test Dental",
        business_address="123 Main St, Salt Lake City, UT",
        industry="dental",
        website_url="https://example.com",
    )

    client = APIClient()
    client.force_authenticate(user=user)
    resp = client.get("/api/aeo/onboarding-prompt-plan/?include_openai=0")
    assert resp.status_code == 200
    body = resp.json()
    assert "groups" in body
    assert "combined" in body
    assert "business" in body
    assert "meta" in body
    assert body["meta"].get("openai_status") == "disabled"
    assert "fixed" in body["groups"]
    assert "dynamic" in body["groups"]
    assert "openai_generated" in body["groups"]
    assert isinstance(body["groups"]["fixed"], list)
    assert isinstance(body["combined"], list)
    if body["groups"]["fixed"]:
        row = body["groups"]["fixed"][0]
        assert "prompt" in row
        assert "type" in row
        assert "weight" in row
        assert "dynamic" in row
    nf = len(body["groups"]["fixed"])
    nd = len(body["groups"]["dynamic"])
    nc = len(body["combined"])
    assert nc <= nf + nd
    if nf > 0 or nd > 0:
        assert nc >= 1
