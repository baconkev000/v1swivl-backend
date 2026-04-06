"""
Lightweight URL reachability checks for AEO wrong-domain classification.

Used when a competitor row attributes a non-canonical URL to the tracked business.
SSRF-safe: only http/https, resolved IPs must be globally routable unless host is allowlisted.

Settings:
    AEO_DOMAIN_VERIFY_TIMEOUT_S — default 3.0
    AEO_DOMAIN_VERIFY_ENABLED — default True
    AEO_DOMAIN_VERIFY_USER_AGENT — optional override
    AEO_DOMAIN_VERIFY_MAX_REDIRECTS — default 5
    AEO_DOMAIN_VERIFY_ALLOWLIST — hosts that bypass RFC1918/reserved IP blocking (tests only)
"""

from __future__ import annotations

import concurrent.futures
import ipaddress
import json
import logging
import re
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from django.conf import settings
from django.utils import timezone

from ..models import AEOExtractionSnapshot

logger = logging.getLogger(__name__)


@dataclass
class UrlReachabilityResult:
    dns_ok: bool
    http_ok: bool
    final_url: str
    status_code: int | None
    error: str


def _verify_timeout_s() -> float:
    try:
        return max(0.5, min(30.0, float(getattr(settings, "AEO_DOMAIN_VERIFY_TIMEOUT_S", 3.0))))
    except (TypeError, ValueError):
        return 3.0


def _max_redirects() -> int:
    try:
        return max(0, min(15, int(getattr(settings, "AEO_DOMAIN_VERIFY_MAX_REDIRECTS", 5))))
    except (TypeError, ValueError):
        return 5


def _user_agent() -> str:
    ua = getattr(settings, "AEO_DOMAIN_VERIFY_USER_AGENT", "") or ""
    s = str(ua).strip()
    if s:
        return s[:256]
    return "SwivlAEO/1.0 (+domain-verify)"


def _allowlist_hosts() -> frozenset[str]:
    raw = getattr(settings, "AEO_DOMAIN_VERIFY_ALLOWLIST", ())
    if isinstance(raw, str):
        raw = (raw,)
    out: set[str] = set()
    for h in raw:
        t = str(h).strip().lower().rstrip(".")
        if t:
            out.add(t)
    return frozenset(out)


def _hostname_allowlisted(hostname: str) -> bool:
    h = (hostname or "").strip().lower().rstrip(".")
    if not h:
        return False
    for a in _allowlist_hosts():
        if h == a or h.endswith("." + a):
            return True
    return False


def _normalize_request_url(url_or_domain: str) -> str | None:
    s = (url_or_domain or "").strip().strip("<>").strip()
    if not s:
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        host_part = s.split("/")[0].split("?")[0]
        if not host_part:
            return None
        s = "https://" + host_part
    try:
        p = urlparse(s)
    except ValueError:
        return None
    if p.scheme not in ("http", "https") or not p.netloc:
        return None
    host = p.hostname
    if not host:
        return None
    netloc = host.lower()
    if p.port and p.port not in (80, 443):
        netloc = f"{netloc}:{p.port}"
    path = p.path or "/"
    if path != "/" and not path.endswith("/"):
        path = path.rstrip("/")
    return f"{p.scheme}://{netloc}{path if path != '/' else '/'}"


def _resolve_hostname_ips(hostname: str, timeout_s: float) -> list[str]:
    def _inner() -> list[str]:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        ips: list[str] = []
        for info in infos:
            sockaddr = info[4]
            if sockaddr and sockaddr[0]:
                ips.append(sockaddr[0])
        return list(dict.fromkeys(ips))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_inner)
        return fut.result(timeout=timeout_s)


def _ips_allowed_for_request(hostname: str, ips: list[str]) -> bool:
    if _hostname_allowlisted(hostname):
        return True
    if not ips:
        return False
    for raw in ips:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


class _LimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        self._count += 1
        if self._count > self._limit:
            raise urllib.error.URLError("too many redirects")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify_url_reachable(
    url_or_domain: str,
    *,
    timeout_s: float | None = None,
    user_agent: str | None = None,
    max_redirects: int | None = None,
) -> UrlReachabilityResult:
    """
    Resolve DNS and probe HTTP (HEAD, then GET on failure). No response bodies stored.

    Returns dns_ok/http_ok separately so callers can classify broken vs live.
    """
    t = timeout_s if timeout_s is not None else _verify_timeout_s()
    ua = user_agent if user_agent is not None else _user_agent()
    redirects = max_redirects if max_redirects is not None else _max_redirects()

    norm = _normalize_request_url(url_or_domain)
    if not norm:
        return UrlReachabilityResult(
            dns_ok=False,
            http_ok=False,
            final_url="",
            status_code=None,
            error="invalid_or_unsupported_url",
        )

    try:
        parsed = urlparse(norm)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    except Exception:
        return UrlReachabilityResult(
            dns_ok=False,
            http_ok=False,
            final_url=norm,
            status_code=None,
            error="parse_error",
        )

    if not hostname:
        return UrlReachabilityResult(
            dns_ok=False,
            http_ok=False,
            final_url=norm,
            status_code=None,
            error="missing_host",
        )

    try:
        ips = _resolve_hostname_ips(hostname, timeout_s=t)
    except (OSError, socket.gaierror, concurrent.futures.TimeoutError) as exc:
        return UrlReachabilityResult(
            dns_ok=False,
            http_ok=False,
            final_url=norm,
            status_code=None,
            error=f"dns_error:{type(exc).__name__}",
        )

    if not _ips_allowed_for_request(hostname, ips):
        return UrlReachabilityResult(
            dns_ok=True,
            http_ok=False,
            final_url=norm,
            status_code=None,
            error="ssrf_blocked_non_global_ip",
        )

    opener = urllib.request.build_opener(_LimitedRedirectHandler(redirects))
    opener.addheaders = [("User-Agent", ua)]

    def _do(method: str) -> tuple[int | None, str, str]:
        req = urllib.request.Request(norm, method=method)
        try:
            with opener.open(req, timeout=t) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
                final = resp.geturl() or norm
                if method == "GET":
                    try:
                        resp.read(1024)
                    except Exception:
                        pass
                return int(code), final, ""
        except urllib.error.HTTPError as e:
            return int(e.code), e.url or norm, ""
        except (urllib.error.URLError, ssl.SSLError, TimeoutError, OSError) as e:
            return None, norm, f"{type(e).__name__}:{e}"

    code, final_url, err = _do("HEAD")
    if code is None or code >= 400:
        code2, final2, err2 = _do("GET")
        if code2 is not None and code2 < 400:
            return UrlReachabilityResult(
                dns_ok=True,
                http_ok=True,
                final_url=final2,
                status_code=code2,
                error="",
            )
        detail = err2 or err or "http_error"
        return UrlReachabilityResult(
            dns_ok=True,
            http_ok=False,
            final_url=final2,
            status_code=code2,
            error=detail[:500],
        )

    if 200 <= code < 400:
        return UrlReachabilityResult(
            dns_ok=True,
            http_ok=True,
            final_url=final_url,
            status_code=code,
            error="",
        )

    return UrlReachabilityResult(
        dns_ok=True,
        http_ok=False,
        final_url=final_url,
        status_code=code,
        error=err or f"http_status_{code}",
    )


def _cited_display_for_wrong_url(url_or_domain: str) -> str:
    norm = _normalize_request_url(url_or_domain)
    if not norm:
        return (url_or_domain or "").strip()[:512]
    try:
        p = urlparse(norm)
        host = (p.hostname or "").lower().rstrip(".")
        return host or norm[:512]
    except Exception:
        return norm[:512]


def resolve_brand_url_status_fields(
    *,
    brand_mentioned: bool,
    tracked_root: str,
    attributed_wrong_url: str | None,
) -> dict[str, Any]:
    """
    Produce DB column values for brand_mentioned_url_status and related fields.

    Phase 4 visibility formulas intentionally ignore these; they exist for UI/diagnostics.
    """
    tr = (tracked_root or "").strip().lower().rstrip(".")

    if brand_mentioned and tr:
        return {
            "brand_mentioned_url_status": AEOExtractionSnapshot.URL_STATUS_MATCHED,
            "cited_domain_or_url": tr,
            "url_verification_notes": {},
            "verified_at": None,
        }

    wrong = (attributed_wrong_url or "").strip()
    if not wrong or not tr:
        return {
            "brand_mentioned_url_status": AEOExtractionSnapshot.URL_STATUS_NOT_MENTIONED,
            "cited_domain_or_url": "",
            "url_verification_notes": {},
            "verified_at": None,
        }

    cited_disp = _cited_display_for_wrong_url(wrong)

    if not getattr(settings, "AEO_DOMAIN_VERIFY_ENABLED", True):
        return {
            "brand_mentioned_url_status": AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN,
            "cited_domain_or_url": cited_disp,
            "url_verification_notes": {"verification_disabled": True},
            "verified_at": None,
        }

    try:
        result = verify_url_reachable(wrong)
    except Exception as exc:
        logger.exception("verify_url_reachable failed unexpectedly: %s", exc)
        return {
            "brand_mentioned_url_status": AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN,
            "cited_domain_or_url": cited_disp,
            "url_verification_notes": {
                "dns_ok": False,
                "http_ok": False,
                "error": f"verify_exception:{type(exc).__name__}",
            },
            "verified_at": timezone.now(),
        }

    notes = {
        "dns_ok": result.dns_ok,
        "http_ok": result.http_ok,
        "status_code": result.status_code,
        "final_url": (result.final_url or "")[:512],
        "error": (result.error or "")[:500],
    }
    live = bool(result.dns_ok and result.http_ok)
    status = (
        AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_LIVE
        if live
        else AEOExtractionSnapshot.URL_STATUS_MENTIONED_URL_WRONG_BROKEN
    )
    return {
        "brand_mentioned_url_status": status,
        "cited_domain_or_url": cited_disp,
        "url_verification_notes": notes,
        "verified_at": timezone.now(),
    }


def verification_notes_json_safe(notes: dict[str, Any]) -> dict[str, Any]:
    """Ensure JSON-serializable dict for JSONField (defensive)."""
    try:
        json.dumps(notes)
        return notes
    except TypeError:
        return {"serialization_error": True, "repr": repr(notes)[:500]}
