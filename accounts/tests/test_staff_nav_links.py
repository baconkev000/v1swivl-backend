import pytest
from django.test import Client

from swivl.users.models import User


pytestmark = pytest.mark.django_db


def test_staff_user_sees_staff_analytics_links():
    user = User.objects.create_user(
        username="staffnav",
        email="staffnav@example.com",
        password="pw",
        is_staff=True,
    )
    c = Client()
    assert c.login(username="staffnav", password="pw")
    r = c.get("/")
    body = r.content.decode("utf-8")
    assert "API Usage" in body
    assert "AEO Ops Console" in body
    assert "/staff/aeo-pass-counts/" in body
    assert "Staff Analytics" not in body
    assert ">About<" not in body


def test_non_staff_user_does_not_see_staff_analytics_links():
    user = User.objects.create_user(
        username="usernav",
        email="usernav@example.com",
        password="pw",
        is_staff=False,
    )
    c = Client()
    assert c.login(username="usernav", password="pw")
    r = c.get("/")
    body = r.content.decode("utf-8")
    assert "API Usage" not in body
    assert "AEO Ops Console" not in body
    assert "/staff/aeo-pass-counts/" not in body
    assert ">About<" not in body

