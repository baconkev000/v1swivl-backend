from __future__ import annotations

import logging
import typing
from urllib.parse import urlparse

from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.conf import settings

if typing.TYPE_CHECKING:
    from allauth.socialaccount.models import SocialLogin
    from django.http import HttpRequest

    from swivl.users.models import User

logger = logging.getLogger(__name__)


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

    def pre_social_login(self, request: HttpRequest, sociallogin: SocialLogin) -> None:
        from accounts.user_identity_reconciliation import reconcile_user_identity_for_email

        raw_email = (
            (getattr(getattr(sociallogin, "user", None), "email", "") or "").strip()
            or str(getattr(getattr(sociallogin, "account", None), "extra_data", {}).get("email") or "").strip()
        )
        if not raw_email:
            return
        result = reconcile_user_identity_for_email(
            raw_email,
            preferred_user=getattr(sociallogin, "user", None),
            reason="social_pre_login",
        )
        canonical = result.user
        if canonical is None or not getattr(canonical, "id", None):
            return
        if getattr(sociallogin, "user", None) is None or getattr(sociallogin.user, "id", None) != canonical.id:
            sociallogin.user = canonical
        logger.info(
            "[social_pre_login] email=%s canonical_user_id=%s reconciled=%s merged_user_ids=%s memberships=%s owned_profiles=%s",
            raw_email.strip().lower(),
            canonical.id,
            result.reconciled,
            ",".join(str(x) for x in result.merged_user_ids),
            result.membership_count,
            result.owned_profile_count,
        )

    def save_user(self, request: HttpRequest, sociallogin: SocialLogin, form=None):
        """
        After Google (or other social) signup/login, ensure a main BusinessProfile exists
        so onboarding can PATCH /api/business-profile/.
        """
        from allauth.socialaccount.models import SocialAccount
        from accounts.business_profile_access import should_create_owned_main_business_profile_for_user
        from accounts.models import BusinessProfile, BusinessProfileMembership
        from accounts.user_identity_reconciliation import reconcile_user_identity_for_email

        raw_email = (
            (getattr(getattr(sociallogin, "user", None), "email", "") or "").strip()
            or str(getattr(getattr(sociallogin, "account", None), "extra_data", {}).get("email") or "").strip()
        )
        reconcile = reconcile_user_identity_for_email(
            raw_email,
            preferred_user=getattr(sociallogin, "user", None),
            reason="social_save_user",
        )
        canonical = reconcile.user
        linked_existing = False
        if canonical is not None and getattr(canonical, "id", None):
            provider = str(getattr(sociallogin.account, "provider", "") or "").strip()
            uid = str(getattr(sociallogin.account, "uid", "") or "").strip()
            linked_existing = SocialAccount.objects.filter(
                user=canonical,
                provider=provider,
                uid=uid,
            ).exists()
            if not linked_existing and provider and uid:
                sociallogin.connect(request, canonical)
                linked_existing = True
            user = canonical
        else:
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
        membership_count = BusinessProfileMembership.objects.filter(user=user).count()
        owned_profile_count = BusinessProfile.objects.filter(user=user).count()
        logger.info(
            "[social_save_user] user_id=%s email=%s reconciled=%s merged_user_ids=%s linked_existing_social=%s memberships=%s owned_profiles=%s",
            user.id,
            (user.email or "").strip().lower(),
            reconcile.reconciled,
            ",".join(str(x) for x in reconcile.merged_user_ids),
            linked_existing,
            membership_count,
            owned_profile_count,
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
