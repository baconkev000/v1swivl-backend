import pytest
from rest_framework import serializers

from accounts.serializers import BusinessProfileSerializer, _normalize_stored_website_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "https://example.com"),
        ("https://Example.com/foo/bar?x=1#h", "https://example.com"),
        ("www.acme.org/path", "https://acme.org"),
        ("https://user:pass@shop.example.com/checkout", "https://shop.example.com"),
        ("http://localhost:3000/app", "http://localhost:3000"),
        ("https://example.com/" + "a" * 300, "https://example.com"),
    ],
)
def test_normalize_stored_website_url_strips_path_and_query(raw, expected):
    assert _normalize_stored_website_url(raw) == expected


def test_normalize_stored_website_url_empty():
    assert _normalize_stored_website_url("") == ""
    assert _normalize_stored_website_url("   ") == ""


def test_business_profile_serializer_validate_website_url_delegates_to_normalize():
    ser = BusinessProfileSerializer()
    assert ser.validate_website_url("https://x.test/p?q=1") == "https://x.test"
    with pytest.raises(serializers.ValidationError):
        ser.validate_website_url("https:///")
