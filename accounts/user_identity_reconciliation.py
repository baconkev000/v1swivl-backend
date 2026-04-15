from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from django.contrib.auth import authenticate, get_user_model
from django.db import transaction

from accounts.models import BusinessProfile, BusinessProfileMembership

logger = logging.getLogger(__name__)
User = get_user_model()


@dataclass
class IdentityReconcileResult:
    user: Any | None
    reconciled: bool
    merged_user_ids: list[int]
    membership_count: int
    owned_profile_count: int


def _norm_email(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    return s if "@" in s else ""


def _user_score(user, *, preferred_user_id: int | None) -> tuple[int, int]:
    external_count = BusinessProfileMembership.objects.filter(user=user).exclude(
        business_profile__user=user
    ).count()
    membership_count = BusinessProfileMembership.objects.filter(user=user).count()
    owned_count = BusinessProfile.objects.filter(user=user).count()
    social_count = user.socialaccount_set.count()
    has_password = 1 if user.has_usable_password() else 0
    preferred = 1 if preferred_user_id is not None and int(user.id) == int(preferred_user_id) else 0
    # External membership dominates because team users should open shared workspace.
    rank = (
        external_count * 1000
        + owned_count * 100
        + membership_count * 50
        + social_count * 20
        + has_password * 10
        + preferred * 5
    )
    return rank, -int(user.id)


def _pick_canonical_user(candidates: list[Any], *, preferred_user_id: int | None) -> Any | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda u: _user_score(u, preferred_user_id=preferred_user_id), reverse=True)[0]


def _merge_memberships(from_user, to_user) -> None:
    rows = list(BusinessProfileMembership.objects.filter(user=from_user).order_by("id"))
    for row in rows:
        existing = BusinessProfileMembership.objects.filter(
            business_profile=row.business_profile,
            user=to_user,
        ).first()
        if existing is not None:
            updates: list[str] = []
            if row.role == BusinessProfileMembership.ROLE_ADMIN and existing.role != BusinessProfileMembership.ROLE_ADMIN:
                existing.role = BusinessProfileMembership.ROLE_ADMIN
                updates.append("role")
            if row.is_owner and not existing.is_owner:
                existing.is_owner = True
                updates.append("is_owner")
            # If any membership is visible, keep visible.
            merged_hidden = bool(existing.hidden_from_team_ui and row.hidden_from_team_ui)
            if merged_hidden != bool(existing.hidden_from_team_ui):
                existing.hidden_from_team_ui = merged_hidden
                updates.append("hidden_from_team_ui")
            if updates:
                existing.save(update_fields=updates + ["updated_at"])
            row.delete()
            continue
        row.user = to_user
        row.save(update_fields=["user", "updated_at"])


def _merge_social_accounts(from_user, to_user) -> None:
    from allauth.socialaccount.models import SocialAccount

    rows = list(SocialAccount.objects.filter(user=from_user).order_by("id"))
    for row in rows:
        conflict = (
            SocialAccount.objects.filter(provider=row.provider, uid=row.uid)
            .exclude(id=row.id)
            .first()
        )
        if conflict is not None:
            if conflict.user_id == to_user.id:
                row.delete()
            continue
        row.user = to_user
        row.save(update_fields=["user"])


def _merge_email_addresses(from_user, to_user) -> None:
    from allauth.account.models import EmailAddress

    rows = list(EmailAddress.objects.filter(user=from_user).order_by("id"))
    for row in rows:
        existing = EmailAddress.objects.filter(user=to_user, email__iexact=row.email).first()
        if existing is not None:
            updates: list[str] = []
            if row.verified and not existing.verified:
                existing.verified = True
                updates.append("verified")
            if row.primary and not existing.primary:
                existing.primary = True
                updates.append("primary")
            if updates:
                existing.save(update_fields=updates)
            row.delete()
            continue
        row.user = to_user
        row.save(update_fields=["user"])


def _merge_login_fields(from_user, to_user) -> None:
    updates: list[str] = []
    if not to_user.has_usable_password() and from_user.has_usable_password():
        to_user.password = from_user.password
        updates.append("password")
    if not str(getattr(to_user, "name", "") or "").strip() and str(getattr(from_user, "name", "") or "").strip():
        to_user.name = from_user.name
        updates.append("name")
    if updates:
        to_user.save(update_fields=updates)


@transaction.atomic
def reconcile_user_identity_for_email(
    email: str | None,
    *,
    preferred_user: Any | None = None,
    reason: str = "",
) -> IdentityReconcileResult:
    normalized = _norm_email(email)
    if not normalized:
        return IdentityReconcileResult(
            user=preferred_user,
            reconciled=False,
            merged_user_ids=[],
            membership_count=0,
            owned_profile_count=0,
        )

    candidates = list(
        User.objects.select_for_update()
        .filter(email__iexact=normalized)
        .order_by("id")
    )
    if not candidates:
        return IdentityReconcileResult(
            user=preferred_user,
            reconciled=False,
            merged_user_ids=[],
            membership_count=0,
            owned_profile_count=0,
        )

    preferred_user_id = int(preferred_user.id) if getattr(preferred_user, "id", None) else None
    canonical = _pick_canonical_user(candidates, preferred_user_id=preferred_user_id)
    if canonical is None:
        canonical = candidates[0]

    merged_user_ids: list[int] = []
    for candidate in candidates:
        if candidate.id == canonical.id:
            continue
        _merge_memberships(candidate, canonical)
        _merge_social_accounts(candidate, canonical)
        _merge_email_addresses(candidate, canonical)
        _merge_login_fields(candidate, canonical)
        merged_user_ids.append(int(candidate.id))

    membership_count = BusinessProfileMembership.objects.filter(user=canonical).count()
    owned_profile_count = BusinessProfile.objects.filter(user=canonical).count()
    if merged_user_ids:
        logger.info(
            "[identity_reconcile] reason=%s email=%s canonical_user_id=%s merged_user_ids=%s memberships=%s owned_profiles=%s",
            reason or "unspecified",
            normalized,
            canonical.id,
            ",".join(str(x) for x in merged_user_ids),
            membership_count,
            owned_profile_count,
        )
    return IdentityReconcileResult(
        user=canonical,
        reconciled=bool(merged_user_ids),
        merged_user_ids=merged_user_ids,
        membership_count=membership_count,
        owned_profile_count=owned_profile_count,
    )


def authenticate_by_email_candidates(email: str, password: str) -> Any | None:
    normalized = _norm_email(email)
    if not normalized or not password:
        return None
    candidates = list(User.objects.filter(email__iexact=normalized).order_by("id"))
    for user in candidates:
        try:
            if user.check_password(password):
                return user
        except Exception:
            continue
    # Fallback for legacy username/password accounts.
    return authenticate(username=email, password=password)
