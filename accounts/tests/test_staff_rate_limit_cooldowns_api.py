import pytest
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import Client
from django.urls import reverse


User = get_user_model()


@pytest.mark.django_db
def test_staff_rate_limit_cooldowns_forbidden_for_non_staff():
    client = Client()
    user = User.objects.create_user(username="nonstaff", email="ns@example.com", password="pw")
    client.force_login(user)
    resp = client.get(reverse("staff-rate-limit-cooldowns"))
    assert resp.status_code == 403


@pytest.mark.django_db
def test_staff_rate_limit_cooldowns_returns_provider_statuses():
    client = Client()
    user = User.objects.create_user(
        username="staffcool",
        email="staffcool@example.com",
        password="pw",
        is_staff=True,
    )
    client.force_login(user)

    cache.set("aeo:openai:rate_limit_until_unix", 4102444800.0, timeout=60)  # future
    cache.set("aeo:gemini:rate_limit_until_unix", 0.0, timeout=60)  # expired
    cache.delete("aeo:perplexity:rate_limit_until_unix")  # unset
    cache.set("seo:dataforseo:rate_limit_until_unix", "bad", timeout=60)  # malformed

    resp = client.get(reverse("staff-rate-limit-cooldowns"))
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True
    providers = body.get("providers") or {}

    openai = providers.get("openai") or {}
    assert openai.get("cache_key") == "aeo:openai:rate_limit_until_unix"
    assert openai.get("active") is True
    assert float(openai.get("cooldown_seconds_remaining") or 0.0) > 0.0

    gemini = providers.get("gemini") or {}
    assert gemini.get("active") is False
    assert float(gemini.get("cooldown_seconds_remaining") or 0.0) == 0.0

    perplexity = providers.get("perplexity") or {}
    assert perplexity.get("cooldown_until_unix") is None
    assert perplexity.get("active") is False

    dataforseo = providers.get("dataforseo") or {}
    assert dataforseo.get("cooldown_until_unix") is None
    assert dataforseo.get("active") is False

    for key in (
        "aeo:openai:rate_limit_until_unix",
        "aeo:gemini:rate_limit_until_unix",
        "aeo:perplexity:rate_limit_until_unix",
        "seo:dataforseo:rate_limit_until_unix",
    ):
        cache.delete(key)
