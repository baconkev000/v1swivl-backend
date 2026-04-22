"""
Wrap django-allauth OAuth2 callback redirects to the SPA so the browser can use
``history.replaceState`` semantics (via ``location.replace``) instead of stacking
another full navigation on top of the callback URL (which made Back land on
``/accounts/.../callback/``).
"""

from __future__ import annotations

import json
from html import escape
from urllib.parse import urlparse

from django.http import HttpResponse

from swivl.users.adapters import _frontend_netlocs


def is_frontend_absolute_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    return parsed.netloc.lower() in _frontend_netlocs()


def replace_redirect_html(next_url: str) -> str:
    """Minimal HTML document: ``location.replace`` so the callback history entry is not kept."""
    safe_js = json.dumps(next_url)
    safe_href = escape(next_url, quote=True)
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Redirecting…</title></head>"
        f"<body><script>location.replace({safe_js});</script>"
        f'<noscript><a href="{safe_href}">Continue</a></noscript></body></html>'
    )


def maybe_wrap_redirect_for_spa_history(response) -> HttpResponse:
    code = getattr(response, "status_code", None)
    if code not in (301, 302, 303, 307, 308):
        return response
    loc = None
    headers = getattr(response, "headers", None)
    if headers is not None:
        loc = headers.get("Location")
    if not loc:
        try:
            loc = response["Location"]
        except (KeyError, TypeError):
            pass
    if not loc:
        loc = getattr(response, "url", None)
    if not loc or not isinstance(loc, str):
        return response
    if not is_frontend_absolute_url(loc):
        return response
    return HttpResponse(
        replace_redirect_html(loc),
        content_type="text/html; charset=utf-8",
        status=200,
    )
