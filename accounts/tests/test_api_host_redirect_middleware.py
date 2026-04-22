import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

User = get_user_model()


@pytest.mark.django_db
def test_non_staff_page_request_redirects_to_frontend(settings):
    settings.FRONTEND_BASE_URL = "https://app.amplerank.ai"
    client = APIClient()
    res = client.get("/")
    assert res.status_code in (301, 302)
    assert res["Location"].startswith("https://app.amplerank.ai/app?")


@pytest.mark.django_db
def test_non_staff_api_request_not_redirected(settings):
    settings.FRONTEND_BASE_URL = "https://app.amplerank.ai"
    client = APIClient()
    res = client.get("/api/auth/status/")
    assert res.status_code == 200
    body = res.json()
    assert body["authenticated"] is False


@pytest.mark.django_db
def test_auth_route_not_redirected(settings):
    settings.FRONTEND_BASE_URL = "https://app.amplerank.ai"
    client = APIClient()
    res = client.get("/auth/google/login/?next=/app")
    assert res.status_code in (301, 302)
    assert "app.amplerank.ai/app?" not in str(res.get("Location", ""))


@pytest.mark.django_db
def test_staff_can_access_backend_page(settings):
    settings.FRONTEND_BASE_URL = "https://app.amplerank.ai"
    staff = User.objects.create_user(
        username="staff_redirect@example.com",
        email="staff_redirect@example.com",
        password="pw",
        is_staff=True,
    )
    client = APIClient()
    assert client.login(username="staff_redirect@example.com", password="pw")
    res = client.get("/")
    assert res.status_code == 200
