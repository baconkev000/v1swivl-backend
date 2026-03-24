from __future__ import annotations

import typing
from urllib.parse import urlparse

from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.conf import settings

if typing.TYPE_CHECKING:
    from allauth.socialaccount.models import SocialLogin
    from django.http import HttpRequest

    from swivl.users.models import User


def _frontend_netlocs() -> set[str]:
    """Hostnames allowed for ?next= redirects back to the SPA."""
    raw = (getattr(settings, "FRONTEND_BASE_URL", "") or "").strip()
    if not raw:
        return set()
    if "://" not in raw:
        raw = f"https://{raw}"
    netloc = urlparse(raw).netloc.lower()
    if not netloc:
        return set()
    out = {netloc}
    if netloc.startswith("www."):
        out.add(netloc[4:])
    else:
        out.add(f"www.{netloc}")
    return out


class AccountAdapter(DefaultAccountAdapter):
    def is_open_for_signup(self, request: HttpRequest) -> bool:
        return getattr(settings, "ACCOUNT_ALLOW_REGISTRATION", True)

    def is_safe_url(self, url: str) -> bool:
        """
        Allow ?next= redirects to FRONTEND_BASE_URL even when host differs from the API host
        (e.g. api.* vs apex/www marketing site). Parent also trusts CSRF_TRUSTED_ORIGINS.
        """
        if url:
            try:
                parsed = urlparse(url)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    if parsed.netloc.lower() in _frontend_netlocs():
                        return True
            except Exception:
                pass
        return super().is_safe_url(url)

class SocialAccountAdapter(DefaultSocialAccountAdapter):
    def is_open_for_signup(
        self,
        request: HttpRequest,
        sociallogin: SocialLogin,
    ) -> bool:
        return getattr(settings, "ACCOUNT_ALLOW_REGISTRATION", True)

    def populate_user(
        self,
        request: HttpRequest,
        sociallogin: SocialLogin,
        data: dict[str, typing.Any],
    ) -> User:
        """
        Populates user information from social provider info.

        See: https://docs.allauth.org/en/latest/socialaccount/advanced.html#creating-and-populating-user-instances
        """
        user = super().populate_user(request, sociallogin, data)
        if not user.name:
            if name := data.get("name"):
                user.name = name
            elif first_name := data.get("first_name"):
                user.name = first_name
                if last_name := data.get("last_name"):
                    user.name += f" {last_name}"
        return user
