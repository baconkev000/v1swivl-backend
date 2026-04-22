import pytest
from django.http import HttpResponseRedirect
from django.test import override_settings

from swivl.users.spa_oauth_redirect import is_frontend_absolute_url, maybe_wrap_redirect_for_spa_history


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/app", False),
        ("not-a-url", False),
        ("", False),
    ],
)
def test_is_frontend_absolute_url_without_setting(url, expected):
    with override_settings(FRONTEND_BASE_URL=""):
        assert is_frontend_absolute_url(url) is expected


@override_settings(FRONTEND_BASE_URL="https://app.example.com")
def test_wraps_redirect_to_configured_frontend():
    target = "https://app.example.com/app?tab=keywords"
    r = HttpResponseRedirect(target)
    out = maybe_wrap_redirect_for_spa_history(r)
    assert out.status_code == 200
    body = out.content.decode()
    assert "location.replace" in body
    assert target in body


@override_settings(FRONTEND_BASE_URL="https://app.example.com")
def test_leaves_api_host_redirect_untouched():
    r = HttpResponseRedirect("https://api.example.com/accounts/profile/")
    out = maybe_wrap_redirect_for_spa_history(r)
    assert out is r
