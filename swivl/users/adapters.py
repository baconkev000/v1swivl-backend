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

    def get_login_redirect_url(self, request: HttpRequest) -> str:
        """
        After Google/social login, send fully onboarded users to /app even if ?next= pointed
        at /onboarding. Incomplete users keep the normal ``next`` destination.
        """
        url = super().get_login_redirect_url(request)
        user = getattr(request, "user", None)
        if user is not None and user.is_authenticated:
            from accounts.onboarding_completion import user_has_completed_full_onboarding

            if user_has_completed_full_onboarding(user):
                base = (getattr(settings, "FRONTEND_BASE_URL", "") or "").strip().rstrip("/")
                if base and "://" not in base:
                    base = f"https://{base.lstrip('/')}"
                if base:
                    return f"{base}/app"
        return url

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

    def save_user(self, request: HttpRequest, sociallogin: SocialLogin, form=None):
        """
        After Google (or other social) signup/login, ensure a main BusinessProfile exists
        so onboarding can PATCH /api/business-profile/.
        """
        from accounts.business_profile_access import should_create_owned_main_business_profile_for_user
        from accounts.models import BusinessProfile, BusinessProfileMembership

        user = super().save_user(request, sociallogin, form)
        if should_create_owned_main_business_profile_for_user(user):
            bp = BusinessProfile.objects.create(user=user, is_main=True)
            BusinessProfileMembership.objects.get_or_create(
                business_profile=bp,
                user=user,
                defaults={
                    "role": BusinessProfileMembership.ROLE_ADMIN,
                    "is_owner": True,
                },
            )
        return user

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
