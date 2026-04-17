from __future__ import annotations

from urllib.parse import quote

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect

# Public/auth paths needed for OAuth and callback flows.
_ALLOWED_PREFIXES = (
    "/api/",
    "/auth/google/login/",
    "/auth/microsoft/login/",
    "/accounts/",
    "/static/",
    "/media/",
    "/__debug__/",
)


def _django_admin_path_prefix() -> str:
    """Normalize settings.ADMIN_URL (e.g. ``admin/``) to a ``/``-prefixed prefix for path checks."""
    raw = str(getattr(settings, "ADMIN_URL", "admin/") or "admin/").strip()
    if not raw.startswith("/"):
        raw = "/" + raw
    if not raw.endswith("/"):
        raw = raw + "/"
    return raw


def _path_is_django_admin(path: str) -> bool:
    prefix = _django_admin_path_prefix()
    if path.startswith(prefix):
        return True
    # Match ``/admin`` when ADMIN_URL is ``admin/`` (no trailing slash on request).
    return path.rstrip("/") == prefix.rstrip("/")


def _frontend_redirect_target(request: HttpRequest) -> str:
    base = str(getattr(settings, "FRONTEND_BASE_URL", "http://localhost:3000") or "").rstrip("/")
    if not base:
        base = "http://localhost:3000"
    # Preserve deep links as ``next`` so frontend can decide what to do.
    next_path = request.get_full_path() or "/"
    return f"{base}/app?next={quote(next_path, safe='/?=&:%#')}"


class RedirectNonStaffApiHostPagesMiddleware:
    """
    Redirect non-staff browser page requests on the API host back to the frontend.

    Keeps all `/api/*` endpoints and auth/callback routes available for the app flow.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        path = request.path or "/"
        if request.method in {"GET", "HEAD"} and not any(path.startswith(p) for p in _ALLOWED_PREFIXES):
            if _path_is_django_admin(path):
                return self.get_response(request)
            user = getattr(request, "user", None)
            is_staff = bool(getattr(user, "is_authenticated", False) and getattr(user, "is_staff", False))
            if not is_staff:
                return redirect(_frontend_redirect_target(request))
        return self.get_response(request)
