from __future__ import annotations

"""
Utility functions for Meta Ads integrations.

OAuth and token storage are handled in views.py; this module exposes
connection status and will host Marketing API helpers for campaigns,
ad sets, ads, creatives, audiences, insights, and (optionally) page posts.
"""

from dataclasses import dataclass

from django.utils import timezone


@dataclass
class MetaAdsStatus:
    """
    Simple status payload indicating whether the current user has a usable
    Meta Ads connection.
    """

    connected: bool = False


def get_meta_ads_status_for_user(user_id: int) -> MetaAdsStatus:
    """
    Return whether the given user has an active Meta Ads connection
    (non-empty access token, optionally still valid if expires_at is set).
    """
    from .models import MetaAdsConnection

    try:
        conn = MetaAdsConnection.objects.get(user_id=user_id)
    except MetaAdsConnection.DoesNotExist:
        return MetaAdsStatus(connected=False)

    if not (conn.access_token and conn.access_token.strip()):
        return MetaAdsStatus(connected=False)

    # Optional: treat as disconnected if token is long expired (e.g. 7 days past)
    if conn.expires_at:
        from datetime import timedelta
        if conn.expires_at < timezone.now() - timedelta(days=7):
            return MetaAdsStatus(connected=False)

    return MetaAdsStatus(connected=True)

