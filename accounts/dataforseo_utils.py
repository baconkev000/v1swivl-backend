from __future__ import annotations

"""
Helpers for calling DataForSEO Labs APIs for the SEO Agent.

Primary endpoints used:
- POST /v3/dataforseo_labs/google/ranked_keywords/live      → current visibility
- POST /v3/dataforseo_labs/google/domain_intersection/live  → keyword gap vs competitors

Usage logging: pass ``business_profile=`` into helpers (or set ``usage_profile_context``) so
``ThirdPartyApiRequestLog.business_profile_id`` is populated for profile-scoped work. Account-agnostic
helpers (e.g. niche-only question seeds) may log with a null profile.
"""

from typing import Any, Dict, List, Optional, Tuple

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
import re
import random
import time

import hashlib
import json
import math
import requests
from pathlib import Path
from django.conf import settings
from django.core.cache import cache

from .models import AEOOverviewSnapshot, BusinessProfile, SEOOverviewSnapshot
from .constants import AEO_SNAPSHOT_TTL
from .seo_metrics_service import (
    build_seo_snapshot_api_metadata,
    effective_rank_for_aggregate_metrics,
    local_verification_affects_visibility,
)
from . import debug_log as _debug

logger = logging.getLogger(__name__)
COMPETITOR_LOOKUP_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

# domain_intersection/live: cache full merged gap list (stable key over sorted competitors + loc/lang/limit).
DOMAIN_INTERSECTION_CACHE_VERSION = "v1"

# Lazy import to avoid circular dependency (openai_utils imports get_or_refresh_seo_score_for_user)
def _get_llm_keyword_candidates(profile: Optional[BusinessProfile], homepage_meta: Optional[str]) -> List[str]:
    try:
        from .openai_utils import generate_seo_keyword_candidates
        return generate_seo_keyword_candidates(profile, homepage_meta)
    except Exception:
        logger.exception("[SEO score] LLM keyword generation failed")
        return []

BASE_URL = "https://api.dataforseo.com"
DEBUG_LOG_PATH = "debug-098bfd.log"

# Debug-mode instrumentation (runtime evidence) for ranking/null issues.
# Use a relative path so it works in both local + containerized deployments
# where the module path resolution may not map 1:1 to the host workspace.
_BA84AE_LOG_PATH = Path("debug-ba84ae.log")


def _dbg_ba84ae_log(
    *,
    hypothesisId: str,
    location: str,
    message: str,
    data: Dict[str, Any] | None = None,
    runId: str = "pre-fix",
) -> None:
    """Write a single NDJSON debug log line for this debug session."""
    try:
        logger.info(
            "[ba84ae_debug] %s %s: %s",
            hypothesisId,
            runId,
            message,
        )
        payload: Dict[str, Any] = {
            "sessionId": "ba84ae",
            "id": f"ba84ae_{hypothesisId}_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "runId": runId,
            "hypothesisId": hypothesisId,
        }
        _BA84AE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_BA84AE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        # Never break production flows due to debug logging failures,
        # but do emit the error to standard logs so we can diagnose why
        # debug-ba84ae.log isn't being created.
        try:
            logger.exception("[ba84ae_debug] failed to write %s: %s", _BA84AE_LOG_PATH, e)
        except Exception:
            pass
        return

# Industries where broad local directories (Yelp, Angi, etc.) are relevant competitors.
# For other niches (e.g. SaaS, resume tools), we do not use these defaults.
_LOCAL_INDUSTRY_KEYWORDS = frozenset({
    "home services", "plumbing", "restaurant", "legal", "contractor",
    "cleaning", "landscaping", "roofing", "electrical", "hvac", "real estate",
})

# Competitor quality filter:
# We exclude common aggregator/social/map/listing domains (Yelp/Facebook/etc.)
# because they skew domain_intersection toward non-direct competitors.
_EXCLUDED_COMPETITOR_DOMAIN_ROOTS = frozenset({
    # Social / platforms
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "pinterest.com",
    "linkedin.com",
    # Directories / review aggregators
    "yelp.com",
    "thumbtack.com",
    "angieslist.com",
    "tripadvisor.com",
    "yellowpages.com",
    "mapquest.com",
    # Other common aggregators/maps
    "opentable.com",
    "foursquare.com",
    "waze.com",
})

_ADDRESS_TOKEN_EXCLUDE = frozenset({
    "usa",
    "us",
    "united",
    "states",
    "america",
    "county",
    "state",
    "province",
    "city",
    "town",
    "region",
    "district",
})


def _normalize_domain_value(domain_or_url: str | None) -> Optional[str]:
    """Normalize competitor override values into a root domain."""
    if not domain_or_url:
        return None
    # Accept either raw domain ("example.com") or URL ("https://example.com/path")
    normalized = normalize_domain(str(domain_or_url).strip())
    return normalized


def get_profile_location_code(
    profile: Optional[BusinessProfile],
    default_code: int,
) -> Tuple[int, bool, str]:
    """
    Resolve the effective DataForSEO location code for a profile.

    Returns:
        (resolved_location_code, used_fallback_default, location_label)
    """
    fallback_code = int(default_code)
    if not profile:
        return fallback_code, True, ""

    # Organic mode intentionally bypasses profile-level location targeting and
    # uses the app default location behavior (current baseline behavior).
    location_mode = str(getattr(profile, "seo_location_mode", "organic") or "organic").strip().lower()
    if location_mode == "organic":
        return fallback_code, False, "Organic"

    raw_code = getattr(profile, "seo_location_code", None)
    raw_label = str(getattr(profile, "seo_location_label", "") or "").strip()
    if raw_code is None:
        return fallback_code, True, raw_label
    try:
        resolved = int(raw_code)
    except (TypeError, ValueError):
        return fallback_code, True, raw_label
    if resolved <= 0:
        return fallback_code, True, raw_label
    return resolved, False, raw_label


def resolve_snapshot_location_context(
    *,
    profile: Optional[BusinessProfile],
    default_location_code: int,
) -> Dict[str, Any]:
    """
    Build mode-aware snapshot context so organic/local snapshots remain isolated.
    """
    location_mode = _get_seo_location_mode(profile)
    resolved_location_code, _used_fallback, resolved_location_label = get_profile_location_code(
        profile,
        int(default_location_code),
    )
    if location_mode != "local":
        return {
            "mode": "organic",
            "code": 0,
            "label": "Organic",
        }
    label = str(resolved_location_label or "").strip()
    if not label:
        label = resolve_local_verification_location(profile) or ""
    return {
        "mode": "local",
        "code": int(resolved_location_code or 0),
        "label": label,
    }


def _parse_domain_csv(raw: str | None) -> List[str]:
    """Parse comma-separated domains into normalized unique root domains."""
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    out: List[str] = []
    seen: set[str] = set()
    for p in parts:
        if not p:
            continue
        d = _normalize_domain_value(p)
        if not d:
            continue
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def _is_excluded_competitor_domain(domain: str) -> bool:
    d = (domain or "").lower().strip()
    if not d:
        return True
    if d in _EXCLUDED_COMPETITOR_DOMAIN_ROOTS:
        return True
    # Defensive: treat subdomains of excluded roots as excluded too.
    for root in _EXCLUDED_COMPETITOR_DOMAIN_ROOTS:
        if root and (d == root or d.endswith("." + root)):
            return True
    return False


def _extract_tokens_from_text(text: str, *, min_len: int = 3, max_tokens: int = 20) -> List[str]:
    """
    Extract lowercase word tokens usable for heuristic domain scoring.
    """
    if not text:
        return []
    t = text.lower()
    tokens = re.split(r"[^a-z0-9]+", t)
    out: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok or len(tok) < min_len:
            continue
        if tok in _ADDRESS_TOKEN_EXCLUDE:
            continue
        if tok.isdigit():
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= max_tokens:
            break
    return out


def _format_company_name_or_url(*, domain: str | None, url: str | None) -> Optional[str]:
    """
    Prefer a readable company name from domain; fallback to URL.
    """
    d = (domain or "").strip().lower()
    if d:
        # Remove common prefixes/suffixes and title-case the brand-like part.
        if d.startswith("www."):
            d = d[4:]
        host = d.split("/")[0].split(":")[0]
        labels = [p for p in host.split(".") if p]
        if labels:
            core = labels[0]
            core = core.replace("-", " ").replace("_", " ").strip()
            if core:
                words = [w for w in core.split() if w]
                if words:
                    return " ".join(w.capitalize() for w in words)

    u = (url or "").strip()
    if u:
        return u
    return None


def _get_profile_for_user(user) -> Optional[BusinessProfile]:
    try:
        return (
            BusinessProfile.objects.filter(user=user, is_main=True).first()
            or BusinessProfile.objects.filter(user=user).first()
        )
    except Exception:
        logger.exception("[SEO score] Failed to load BusinessProfile for competitor selection")
        return None


def _score_competitor_domain(
    *,
    competitor_domain: str,
    target_domain: str,
    niche_tokens: List[str],
    geo_tokens: List[str],
) -> float:
    """
    Heuristic scoring for competitor priority when multiple filtered candidates exist.
    Keeps DataForSEO relevance order as a baseline; boosts same-niche & geo hints.
    """
    score = 0.0
    d = (competitor_domain or "").lower()

    if d.endswith(".com") and target_domain.endswith(".com"):
        score += 0.1

    if niche_tokens and any(t in d for t in niche_tokens):
        score += 2.0

    if geo_tokens and any(t in d for t in geo_tokens):
        score += 1.0

    return score


def get_competitors_for_domain_intersection(
    *,
    domain: str,
    location_code: int,
    language_code: str,
    user=None,
    profile: Optional[BusinessProfile] = None,
    limit: int = 5,
    min_competitors: int = 3,
) -> Dict[str, Any]:
    """
    Select competitor domains for domain_intersection/live using:
    - per-profile overrides (preferred)
    - DataForSEO competitors_domain/live auto-competitors (filtered + prioritized)
    - manual fallback list from settings (if provided)

    Returns both raw and filtered lists for debug logging.
    """
    _profile = profile or _get_profile_for_user(user)
    industry_lower = ((_profile.industry or "").strip()).lower() if _profile else ""
    business_address = ((_profile.business_address or "").strip()).lower() if _profile else ""

    override_domains = _parse_domain_csv(getattr(_profile, "seo_competitor_domains_override", "") or "")

    raw_competitor_candidates: List[str] = []
    if not override_domains:
        # Only fetch paid auto-competitors when no explicit profile override is configured.
        # Use a larger initial limit to allow for filtering down to "limit".
        raw_competitor_candidates = _get_competitor_domains(
            domain,
            location_code=location_code,
            language_code=language_code,
            limit=min(max(limit * 2, limit + 2), 10),
            business_profile=_profile,
        )

    # Filter out generic aggregators/social/map/listing domains.
    filtered_auto: List[str] = []
    for c in raw_competitor_candidates:
        if not c:
            continue
        if c.lower().strip() == domain.lower().strip():
            continue
        if _is_excluded_competitor_domain(c):
            continue
        filtered_auto.append(c.lower().strip())

    # Heuristic prioritization: same niche + possible geo hints inside the domain.
    niche_tokens = _extract_tokens_from_text(industry_lower, min_len=3, max_tokens=20)
    if any(k in industry_lower for k in ("dental", "dentist", "orthodont", "implan")):
        # Expand dental-specific tokens (improves sorting on dental practice domains).
        for t in ["dental", "dentist", "orthodont", "implant", "implants", "clinic", "care"]:
            if t not in niche_tokens:
                niche_tokens.append(t)
    geo_tokens = _extract_tokens_from_text(business_address, min_len=3, max_tokens=10)

    # Preserve original order as a fallback; score only as a tiebreak.
    scored_auto: List[tuple[int, str, float]] = [
        (
            i,
            c,
            _score_competitor_domain(
                competitor_domain=c,
                target_domain=domain,
                niche_tokens=niche_tokens,
                geo_tokens=geo_tokens,
            ),
        )
        for i, c in enumerate(filtered_auto)
    ]
    scored_auto.sort(key=lambda t: (-t[2], t[0]))
    filtered_auto_prioritized = [c for _i, c, _s in scored_auto][:limit]

    manual_raw = getattr(settings, "DATAFORSEO_MANUAL_COMPETITORS", "") or ""
    manual_competitors = _parse_domain_csv(manual_raw)
    manual_competitors = [c for c in manual_competitors if not _is_excluded_competitor_domain(c)][:limit]

    competitor_domains_used: List[str] = []
    competitor_source = "none"

    if override_domains:
        competitor_domains_used = override_domains[:limit]
        competitor_source = "profile_override"
    else:
        if len(filtered_auto_prioritized) >= min_competitors:
            competitor_domains_used = filtered_auto_prioritized
            competitor_source = "filtered_auto"
        elif manual_competitors:
            # Too weak: prefer manual competitor list instead of generic directories.
            competitor_domains_used = manual_competitors
            competitor_source = "manual_fallback"
        else:
            competitor_domains_used = []
            competitor_source = "insufficient_auto_after_filter"

    # Debug logs: raw vs filtered.
    try:
        logger.info(
            "[SEO competitors] domain=%s raw_competitors=%s filtered_auto=%s used=%s source=%s",
            domain,
            raw_competitor_candidates,
            filtered_auto_prioritized,
            competitor_domains_used,
            competitor_source,
        )
    except Exception:
        pass

    _dbg_ba84ae_log(
        hypothesisId="H7_competitor_filter",
        location="accounts/dataforseo_utils.py:get_competitors_for_domain_intersection",
        message="competitor selection evidence (raw->filtered->used)",
        data={
            "domain": domain,
            "raw_competitors": raw_competitor_candidates,
            "filtered_auto_prioritized": filtered_auto_prioritized,
            "competitor_domains_used": competitor_domains_used,
            "competitor_source": competitor_source,
            "override_domains_count": len(override_domains),
            "manual_competitors_count": len(manual_competitors),
        },
        runId="pre-fix",
    )

    return {
        "raw_competitors": raw_competitor_candidates,
        "filtered_competitors_used": competitor_domains_used,
        "competitor_source": competitor_source,
    }

# Seed keywords come only from the LLM (generate_seo_keyword_candidates) and are
# validated via DataForSEO search volume; no fixed industry/generic seed data.


# Stopwords: phrases must not start/end with these, and we do not form n-grams from them.
_DESC_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "as", "it", "its",
    "that", "this", "takes", "take", "while", "also", "from", "into",
    "user", "users", "their", "them", "they", "we", "our", "your",
})

# Verb/noun forms to normalize toward search-intent (verb -> noun or service term).
_DESC_NORMALIZE = {
    "tailors": "tailoring", "tailor": "tailoring", "tailored": "tailoring",
    "improving": "improvement", "improve": "improvement", "improves": "improvement",
    "optimizing": "optimization", "optimize": "optimization", "optimizes": "optimization",
    "building": "builder", "builds": "builder", "build": "builder",
    "checking": "checker", "checks": "checker", "check": "checker",
}

# Concept-style phrases (e.g. ATS resume checker, resume optimization tool) are generated by the LLM
# in generate_seo_keyword_candidates(); no fixed concept map is used here.


def _significant_words(text: str) -> List[str]:
    """Tokenize and return only words that are not stopwords and look substantive."""
    words = [w for w in text.split() if w.isalnum() or w.replace("-", "").isalnum()]
    out = []
    for w in words:
        low = w.lower()
        if low in _DESC_STOPWORDS:
            continue
        if len(low) >= 2 or low in ("ats", "ai", "cv"):
            out.append(w)
    return out


def _is_search_like(phrase: str, allow_one_stopword: bool = False) -> bool:
    """Reject fragments that do not resemble a natural search query."""
    if not phrase or len(phrase) < 4 or len(phrase) > 50:
        return False
    low = phrase.lower()
    words = low.split()
    if not words:
        return False
    if words[0] in _DESC_STOPWORDS or words[-1] in _DESC_STOPWORDS:
        return False
    stop_count = sum(1 for w in words if w in _DESC_STOPWORDS)
    if not allow_one_stopword and stop_count > 0:
        return False
    if allow_one_stopword and (stop_count > 1 or (stop_count == 1 and len(words) < 3)):
        return False  # allow at most one internal stopword in 3+ word phrases (e.g. "resume for job description")
    if len(words) < 2:
        return False
    if any(frag in low for frag in (" that ", " that", " a ", " and ", " to the ", " it ")):
        return False
    return True


def _normalize_phrase(phrase: str) -> str:
    """Convert phrase toward search-intent form (e.g. 'resume tailors' -> 'resume tailoring')."""
    words = phrase.split()
    out = []
    for w in words:
        low = w.lower()
        out.append(_DESC_NORMALIZE.get(low, low))
    normalized = " ".join(out)
    # Prefer "ATS X" over "X ATS" for search intent (e.g. "improvement ATS" -> "ATS improvement")
    parts = normalized.split()
    if len(parts) == 2 and parts[1].lower() == "ats":
        normalized = f"{parts[1]} {parts[0]}"
    return normalized


def _extract_phrases_from_description(description: str, max_phrases: int = 5) -> List[str]:
    """
    Extract search-intent keyword phrases from business description using only
    significant-word consecutive n-grams. No fixed concept maps; concept-style
    phrases (e.g. ATS resume checker, resume optimization tool) come from the LLM.
    """
    if not description or not description.strip():
        return []
    text = description.strip()
    significant = _significant_words(text)
    if len(significant) < 2:
        return []

    seen: set = set()
    out: List[str] = []

    for n in (3, 2):
        for i in range(len(significant) - n + 1):
            if len(out) >= max_phrases + 5:
                break
            phrase_words = significant[i : i + n]
            raw = " ".join(phrase_words).strip()[:50]
            candidate = _normalize_phrase(raw)
            candidate = candidate.strip()[:50]
            if not _is_search_like(candidate):
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)

    def score_phrase(s: str) -> tuple:
        w = s.split()
        return (len(w), -len(s), s.lower())

    out_sorted = sorted(set(out), key=score_phrase, reverse=True)
    return out_sorted[:max_phrases]


def _get_business_seed_phrases(user) -> List[tuple[str, int]]:
    """
    Build niche-specific seed keywords from BusinessProfile: industry, description,
    and business name. Returns list of (phrase, default_search_volume) so that
    top_keywords reflect what the business actually offers (e.g. resume builder,
    online resume creator) rather than generic terms.
    """
    profile = None
    try:
        profile = (
            BusinessProfile.objects.filter(user=user, is_main=True).first()
            or BusinessProfile.objects.filter(user=user).first()
        )
    except Exception:
        return []
    if not profile:
        return []

    seeds: List[tuple[str, int]] = []
    industry = (profile.industry or "").strip()
    description = (profile.description or "").strip()
    business_name = (profile.business_name or "").strip()

    if industry and len(industry) >= 2 and len(industry) <= 60:
        seeds.append((industry, 500))

    for phrase in _extract_phrases_from_description(description, max_phrases=5):
        if phrase and phrase not in {s[0] for s in seeds}:
            seeds.append((phrase, 400))

    # Only add business name if it's descriptive (e.g. "Oak Tree Plumbing"); skip single-word brands like "Rezzii"
    if business_name and len(business_name) >= 2 and len(business_name) <= 60:
        name_lower = business_name.lower()
        if name_lower not in {s[0].lower() for s in seeds}:
            words = business_name.split()
            if len(words) > 1 or len(business_name) > 8:
                seeds.append((business_name, 300))

    return seeds


def _get_auth() -> Optional[tuple[str, str]]:
    """
    Return (login, password) tuple for DataForSEO HTTP Basic auth.

    Expected settings or env:
    - DATAFORSEO_LOGIN
    - DATAFORSEO_PASSWORD
    """
    login = getattr(settings, "DATAFORSEO_LOGIN", None)
    password = getattr(settings, "DATAFORSEO_PASSWORD", None)

    if not login or not password:
        logger.warning(
            "[DataForSEO] Missing DATAFORSEO_LOGIN/DATAFORSEO_PASSWORD; skipping API call.",
        )
        return None
    return str(login), str(password)


def _get_search_volumes_for_keywords(
    keywords: List[str],
    location_code: int,
    *,
    business_profile: Optional[BusinessProfile] = None,
) -> Dict[str, int]:
    """
    Fetch monthly search volume for a list of keyword phrases via DataForSEO
    keywords_data/google/search_volume/live. Returns a dict keyed by keyword (lowercase)
    with search_volume > 0 only. Used to validate LLM-generated candidates.
    """
    if not keywords:
        return {}
    # DataForSEO recommends not sending language_code for this endpoint (can return null)
    payload = [
        {
            "location_code": int(location_code),
            "keywords": [str(k).strip() for k in keywords if str(k).strip()][:700],
        },
    ]
    data = _post(
        "/v3/keywords_data/google/search_volume/live",
        payload,
        business_profile=business_profile,
    )
    if not data:
        return {}
    result: Dict[str, int] = {}
    try:
        tasks = data.get("tasks") or []
        if not tasks:
            return {}
        for task in tasks:
            for item in task.get("result") or []:
                kw = (item.get("keyword") or "").strip()
                if not kw:
                    continue
                sv = item.get("search_volume")
                try:
                    vol = int(sv) if sv is not None else 0
                except (TypeError, ValueError):
                    vol = 0
                if vol > 0:
                    result[kw.lower()] = vol
        return result
    except Exception:
        logger.exception(
            "[DataForSEO] Error parsing search_volume/live response for keywords_count=%s",
            len(keywords),
        )
        return {}


def _post(
    path: str,
    payload: List[Dict[str, Any]],
    *,
    business_profile: Optional[BusinessProfile] = None,
) -> Optional[Dict[str, Any]]:
    """Low-level POST helper with basic error handling."""
    auth = _get_auth()
    if auth is None:
        return None

    url = f"{BASE_URL}{path}"
    try:
        resp = requests.post(
            url,
            json=payload,
            auth=auth,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - network failure
        logger.exception("[DataForSEO] POST %s failed: %s", path, exc)
        from accounts.models import ThirdPartyApiErrorLog, ThirdPartyApiProvider
        from accounts.third_party_usage import record_third_party_api_error

        record_third_party_api_error(
            provider=ThirdPartyApiProvider.DATAFORSEO,
            operation=path,
            error_kind=ThirdPartyApiErrorLog.ErrorKind.CONNECTION_ERROR,
            message=str(exc)[:1024],
            detail=str(exc),
            http_status=None,
            business_profile=business_profile,
        )
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "accounts/dataforseo_utils.py:_post",
                    "message": "HTTP error when calling DataForSEO",
                    "data": {"path": path, "error": str(exc)},
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return None

    parsed_json: Optional[Dict[str, Any]] = None
    if resp.content:
        try:
            parsed_json = resp.json()
        except ValueError:
            parsed_json = None

    operation_label = path if resp.status_code == 200 else f"{path} HTTP {resp.status_code}"
    from accounts.models import ThirdPartyApiErrorLog, ThirdPartyApiProvider
    from accounts.third_party_usage import record_dataforseo_request, record_third_party_api_error

    record_dataforseo_request(
        operation=operation_label,
        response_json=parsed_json,
        business_profile=business_profile,
    )

    if resp.status_code != 200:
        # DataForSEO returns 200 with status_code inside body for logical errors;
        # non-200 here is a transport-level issue.
        record_third_party_api_error(
            provider=ThirdPartyApiProvider.DATAFORSEO,
            operation=path,
            error_kind=ThirdPartyApiErrorLog.ErrorKind.HTTP_ERROR,
            message=f"HTTP {resp.status_code}",
            detail=(resp.text or "")[:4000],
            http_status=resp.status_code,
            business_profile=business_profile,
        )
        logger.warning(
            "[DataForSEO] POST %s HTTP %s: %s",
            path,
            resp.status_code,
            resp.text[:500],
        )
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "accounts/dataforseo_utils.py:_post",
                    "message": "Non-200 from DataForSEO",
                    "data": {
                        "path": path,
                        "status_code": resp.status_code,
                        "body_preview": resp.text[:200],
                    },
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return None

    if parsed_json is None:
        record_third_party_api_error(
            provider=ThirdPartyApiProvider.DATAFORSEO,
            operation=path,
            error_kind=ThirdPartyApiErrorLog.ErrorKind.PARSE_ERROR,
            message="Non-JSON response body",
            detail=(resp.text or "")[:4000],
            http_status=resp.status_code,
            business_profile=business_profile,
        )
        logger.warning("[DataForSEO] POST %s returned non-JSON body.", path)
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "accounts/dataforseo_utils.py:_post",
                    "message": "Non-JSON response from DataForSEO",
                    "data": {"path": path, "body_preview": resp.text[:200]},
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return None

    data = parsed_json
    # DataForSEO wraps results in tasks / result; callers will unpack further.
    # #region agent log
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "sessionId": "098bfd",
                "runId": "pre-fix",
                "hypothesisId": "H1",
                "location": "accounts/dataforseo_utils.py:_post",
                "message": "DataForSEO response summary",
                "data": {
                    "path": path,
                    "has_tasks": bool((data or {}).get("tasks")),
                },
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion
    return data


def _ctr_for_position(position: int) -> float:
    """
    Simple CTR curve approximating typical SERP click-through rates.
    Position 1 ≈ 0.28, 2 ≈ 0.15, 3 ≈ 0.10, 4 ≈ 0.07, 5 ≈ 0.05, 6–10 decreasing.
    """
    if position <= 0:
        return 0.0
    if position == 1:
        return 0.28
    if position == 2:
        return 0.15
    if position == 3:
        return 0.10
    if position == 4:
        return 0.07
    if position == 5:
        return 0.05
    if 6 <= position <= 10:
        # Linearly decay from 0.04 at 6 → 0.01 at 10
        return max(0.01, 0.04 - (position - 6) * 0.0075)
    if 11 <= position <= 20:
        return 0.005
    return 0.002


def _appearance_weight_for_position(position: int) -> float:
    """
    Estimate appearance coverage contribution by rank band.
    This is intentionally different from CTR, representing likely SERP presence.
    """
    if position <= 0:
        return 0.0
    if position <= 3:
        return 1.0
    if position <= 10:
        return 0.85
    if position <= 20:
        return 0.45
    if position <= 50:
        return 0.2
    if position <= 100:
        return 0.08
    return 0.02


def _estimate_missed_searches_monthly(search_volume: float, rank: Optional[int]) -> int:
    """
    Estimate how many monthly searches are *not* expected to click the business site.

    Uses the same CTR curve as the overall visibility/SEO score, so per-keyword
    missed-search math stays consistent with the snapshot-level "missed_searches_monthly".
    """
    try:
        sv = max(float(search_volume or 0.0), 0.0)
    except (TypeError, ValueError):
        sv = 0.0
    if sv <= 0:
        return 0

    if rank is None or rank <= 0:
        # If we don't have a rank, assume 0% CTR for that snippet.
        return int(round(sv))

    ctr = _ctr_for_position(int(rank))
    estimated_clicks = sv * ctr
    return int(round(max(0.0, sv - estimated_clicks)))


def _extract_keyword_difficulty(keyword_info: Dict[str, Any]) -> Optional[float]:
    """
    Extract a difficulty / competition score from DataForSEO keyword_info.
    Handles either 0–1 or 0–100 scales and normalises to 0–100.
    """
    if not keyword_info:
        return None

    for key in ("competition", "competition_level", "difficulty", "keyword_difficulty"):
        if key in keyword_info and keyword_info[key] is not None:
            try:
                val = float(keyword_info[key])
            except (TypeError, ValueError):
                continue
            # If looks like 0–1, scale to 0–100
            if 0.0 <= val <= 1.0:
                return val * 100.0
            return max(0.0, min(100.0, val))

    return None


def _get_competitor_average_traffic(
    target_domain: str,
    *,
    location_code: int,
    language_code: str,
    business_profile: Optional[BusinessProfile] = None,
) -> float:
    """
    Call competitors_domain/live to estimate the average organic traffic/visibility
    of the top competitors for the given domain.
    """
    if bool(getattr(settings, "DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS", False)):
        logger.info(
            "[DataForSEO] competitor lookup bypassed (kill switch) path=_get_competitor_average_traffic domain=%s",
            target_domain,
        )
        return 0.0

    cache_key = (
        f"seo:competitors_domain:avg_traffic:v2:"
        f"{target_domain.lower().strip()}:{int(location_code)}:{language_code}"
    )
    cached = cache.get(cache_key)
    if cached is not None:
        try:
            return float(cached)
        except (TypeError, ValueError):
            pass

    payload = [
        {
            "target": target_domain,
            "location_code": int(location_code),
            "language_code": language_code,
        },
    ]

    data = _post(
        "/v3/dataforseo_labs/google/competitors_domain/live",
        payload,
        business_profile=business_profile,
    )
    if not data:
        cache.set(cache_key, 0.0, timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
        return 0.0

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            cache.set(cache_key, 0.0, timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
            return 0.0
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            cache.set(cache_key, 0.0, timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
            return 0.0
        result = results[0]
        items = result.get("items") or []
        if not items:
            cache.set(cache_key, 0.0, timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
            return 0.0

        competitor_scores: List[float] = []
        for item in items:
            metrics = item.get("organic_metrics") or item
            raw_vis = (
                metrics.get("estimated_traffic")
                or metrics.get("visibility")
                or metrics.get("sum_search_volume")
                or 0
            )
            try:
                vis = float(raw_vis)
            except (TypeError, ValueError):
                vis = 0.0
            if vis > 0:
                competitor_scores.append(vis)

        if not competitor_scores:
            cache.set(cache_key, 0.0, timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
            return 0.0

        avg_traffic = sum(competitor_scores) / len(competitor_scores)
        cache.set(cache_key, float(avg_traffic), timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
        return avg_traffic
    except Exception:
        logger.exception(
            "[DataForSEO] competitors_domain parsing failed for target_domain=%s",
            target_domain,
        )
        cache.set(cache_key, 0.0, timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
        return 0.0


def _get_competitor_domains(
    target_domain: str,
    *,
    location_code: int,
    language_code: str,
    limit: int = 5,
    business_profile: Optional[BusinessProfile] = None,
) -> List[str]:
    """
    Call competitors_domain/live and return 3–5 main competitor domain names
    (excluding the target). Used to feed get_keyword_gap_keywords for
    high-intent opportunity keywords.
    """
    if bool(getattr(settings, "DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS", False)):
        logger.info(
            "[DataForSEO] competitor lookup bypassed (kill switch) path=_get_competitor_domains domain=%s",
            target_domain,
        )
        return []

    cache_key = (
        f"seo:competitors_domain:domains:v2:"
        f"{target_domain.lower().strip()}:{int(location_code)}:{language_code}:{int(limit)}"
    )
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        return [str(x).strip().lower() for x in cached if str(x).strip()]

    payload = [
        {
            "target": target_domain,
            "location_code": int(location_code),
            "language_code": language_code,
            "limit": min(limit + 5, 100),
        },
    ]
    data = _post(
        "/v3/dataforseo_labs/google/competitors_domain/live",
        payload,
        business_profile=business_profile,
    )
    if not data:
        cache.set(cache_key, [], timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
        return []

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            cache.set(cache_key, [], timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
            return []
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            cache.set(cache_key, [], timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
            return []
        result = results[0]
        items = result.get("items") or []
        if not items:
            cache.set(cache_key, [], timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
            return []
        target_lower = target_domain.lower().strip()
        domains: List[str] = []
        for item in items:
            domain = (item.get("domain") or "").strip().lower()
            if not domain or domain == target_lower:
                continue
            if domain in domains:
                continue
            domains.append(domain)
            if len(domains) >= limit:
                break
        logger.info(
            "[DataForSEO] competitors_domain domains for target=%s -> %s",
            target_domain,
            domains[:limit],
        )
        final_domains = domains[:limit]
        cache.set(cache_key, final_domains, timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
        return final_domains
    except Exception:
        logger.exception(
            "[DataForSEO] competitors_domain domain list parsing failed for target_domain=%s",
            target_domain,
        )
        cache.set(cache_key, [], timeout=COMPETITOR_LOOKUP_CACHE_TTL_SECONDS)
        return []


def get_ranked_keywords_visibility(
    target_domain: str,
    *,
    location_code: int,
    language_code: str = "en",
    limit: int = 100,
    business_profile: Optional[BusinessProfile] = None,
) -> Optional[Dict[str, Any]]:
    """
    Call ranked_keywords/live to get visibility & ranking metrics for a domain.

    Returns a dict with at least:
    - visibility: float
    - keywords_count: int
    - top3_positions: int
    """
    # #region agent log
    from . import debug_log as _debug
    _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:entry", "DataForSEO call", {"target_domain": target_domain, "location_code": location_code}, "H3")
    # #endregion
    payload = [
        {
            "target": target_domain,
            "location_code": int(location_code),
            "language_code": language_code,
            "limit": int(limit),
        },
    ]

    data = _post(
        "/v3/dataforseo_labs/google/ranked_keywords/live",
        payload,
        business_profile=business_profile,
    )
    # #region agent log
    _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:after_post", "API response", {"has_data": data is not None}, "H4")
    # #endregion
    # #region agent log
    _debug.log(
        "dataforseo_utils.py:get_ranked_keywords_visibility:response_shape",
        "Ranked keywords response shape",
        {
            "data_type": type(data).__name__ if data is not None else "NoneType",
            "top_keys": list((data or {}).keys())[:8] if isinstance(data, dict) else [],
            "tasks_count": len((data or {}).get("tasks") or []) if isinstance(data, dict) else -1,
        },
        "H1",
    )
    # #endregion
    if not data:
        return None

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            # #region agent log
            _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:no_tasks", "no tasks in response", {"target_domain": target_domain}, "H5")
            # #endregion
            logger.warning("[DataForSEO] ranked_keywords: no tasks in response for %s", target_domain)
            return None
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            # #region agent log
            _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:no_result", "no result for domain", {"target_domain": target_domain}, "H5")
            # #endregion
            logger.warning("[DataForSEO] ranked_keywords: no result for %s", target_domain)
            return None
        result = results[0]
        items = result.get("items") or []

        keywords_count = len(items)

        top3 = sum(
            1 for item in items
            if item.get("rank_absolute") and item["rank_absolute"] <= 3
        )

        # crude visibility proxy
        visibility = sum(
            (item.get("search_volume") or 0)
            for item in items
        )

        logger.info(
            "[DataForSEO] ranked_keywords target=%s visibility=%.2f keywords_count=%s top3=%s",
            target_domain,
            visibility,
            keywords_count,
            top3,
        )

        # #region agent log
        _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:success", "returning visibility dict", {"keywords_count": keywords_count, "top3_positions": top3}, "H5")
        # #endregion
        return {
            "visibility": visibility,
            "keywords_count": keywords_count,
            "top3_positions": top3,
        }
    except Exception as exc:  # pragma: no cover - defensive
        # #region agent log
        _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:exception", "parsing exception", {"exc_type": type(exc).__name__, "exc_msg": str(exc)[:200]}, "H4")
        # #endregion
        logger.exception(
            "[DataForSEO] ranked_keywords parsing failed for %s: %s",
            target_domain,
            exc,
        )
        return None


def get_question_intent_keywords(
    *,
    target_domain: Optional[str] = None,
    niche: Optional[str] = None,
    location_code: Optional[int] = None,
    language_code: str = "en",
    limit: int = 20,
    business_profile: Optional[BusinessProfile] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve question-intent keywords from DataForSEO for a business domain or niche.

    - Uses ranked_keywords/live when target_domain is provided.
    - Uses keywords_data search_volume/live with seeded question phrases when niche is provided.
    - Keeps only question-intent prefixes: how, what, why, when, best.
    - Returns at most 20 rows in shape: {"keyword": str, "search_volume": int}
    - Returns [] on any failure.
    """
    try:
        max_rows = max(1, min(int(limit or 20), 20))
        resolved_location_code = int(
            location_code if location_code is not None else getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840)
        )
        question_prefixes = ("how ", "what ", "why ", "when ", "best ")

        def _is_question_intent_keyword(value: Any) -> bool:
            kw = str(value or "").strip().lower()
            return any(kw.startswith(prefix) for prefix in question_prefixes)

        collected: Dict[str, int] = {}

        normalized_domain = normalize_domain(target_domain) if target_domain else None
        if normalized_domain:
            payload = [
                {
                    "target": normalized_domain,
                    "location_code": resolved_location_code,
                    "language_code": language_code,
                    "limit": 200,
                },
            ]
            data = _post(
                "/v3/dataforseo_labs/google/ranked_keywords/live",
                payload,
                business_profile=business_profile,
            )
            if data:
                tasks = data.get("tasks") or []
                for task in tasks:
                    for result in (task.get("result") or []):
                        for item in (result.get("items") or []):
                            keyword_data = item.get("keyword_data") or {}
                            kw = (keyword_data.get("keyword") or item.get("keyword") or "").strip()
                            if not kw or not _is_question_intent_keyword(kw):
                                continue

                            kw_info = keyword_data.get("keyword_info") or {}
                            raw_sv = (
                                kw_info.get("search_volume")
                                or kw_info.get("search_volume_global")
                                or kw_info.get("sum_search_volume")
                                or item.get("search_volume")
                                or item.get("sum_search_volume")
                                or 0
                            )
                            try:
                                sv = int(raw_sv) if raw_sv is not None else 0
                            except (TypeError, ValueError):
                                sv = 0
                            if sv <= 0:
                                continue
                            prev = collected.get(kw.lower(), 0)
                            if sv > prev:
                                collected[kw.lower()] = sv

        niche_clean = (niche or "").strip()
        if niche_clean:
            seeded_keywords = [f"{prefix.strip()} {niche_clean}" for prefix in question_prefixes]
            seed_volumes = _get_search_volumes_for_keywords(
                seeded_keywords,
                resolved_location_code,
                business_profile=business_profile,
            )
            for kw_seed in seeded_keywords:
                kw_seed_clean = kw_seed.strip()
                if not _is_question_intent_keyword(kw_seed_clean):
                    continue
                sv = int(seed_volumes.get(kw_seed_clean.lower(), 0) or 0)
                if sv <= 0:
                    continue
                prev = collected.get(kw_seed_clean.lower(), 0)
                if sv > prev:
                    collected[kw_seed_clean.lower()] = sv

        def _extract_paa_questions_for_terms(terms: List[str]) -> List[str]:
            """
            Pull question variants from SERP People Also Ask blocks.
            Best-effort only: returns [] on any failure.
            """
            if not terms:
                return []
            found: List[str] = []
            # Keep this lightweight to control API usage.
            for term in terms[:5]:
                payload = [
                    {
                        "keyword": term,
                        "location_code": resolved_location_code,
                        "language_code": language_code,
                        "device": "desktop",
                        "os": "windows",
                        "depth": 20,
                    },
                ]
                data = _post(
                    "/v3/serp/google/organic/live/advanced",
                    payload,
                    business_profile=business_profile,
                )
                if not data:
                    continue
                try:
                    tasks = data.get("tasks") or []
                    for task in tasks:
                        for result in (task.get("result") or []):
                            for item in (result.get("items") or []):
                                item_type = str(item.get("type") or "").lower()
                                if "people_also_ask" not in item_type and "related_questions" not in item_type:
                                    continue
                                nested_items = item.get("items") or []
                                for nested in nested_items if isinstance(nested_items, list) else []:
                                    q = str(
                                        nested.get("question")
                                        or nested.get("title")
                                        or nested.get("text")
                                        or ""
                                    ).strip()
                                    if not q:
                                        continue
                                    ql = q.lower()
                                    if _is_question_intent_keyword(ql) and ql not in found:
                                        found.append(ql)
                except Exception:
                    continue
            return found

        # Enrich demand set with true SERP/PAA questions from top terms.
        if collected:
            top_seed_terms = [
                kw for kw, _sv in sorted(collected.items(), key=lambda kv: -int(kv[1]))[:5]
            ]
            paa_questions = _extract_paa_questions_for_terms(top_seed_terms)
            if paa_questions:
                paa_volumes = _get_search_volumes_for_keywords(
                    paa_questions,
                    resolved_location_code,
                    business_profile=business_profile,
                )
                for q in paa_questions:
                    sv = int(paa_volumes.get(q.lower(), 0) or 0)
                    if sv <= 0:
                        continue
                    prev = collected.get(q.lower(), 0)
                    if sv > prev:
                        collected[q.lower()] = sv

        if not collected:
            return []

        sorted_rows = sorted(
            ({"keyword": kw, "search_volume": int(sv)} for kw, sv in collected.items()),
            key=lambda row: (-int(row.get("search_volume") or 0), str(row.get("keyword") or "")),
        )
        return sorted_rows[:max_rows]
    except Exception:
        logger.exception(
            "[DataForSEO] get_question_intent_keywords failed for domain=%s niche=%s",
            target_domain,
            niche,
        )
        return []


def _extract_text_fragments_for_question_coverage(value: Any) -> List[str]:
    """
    Recursively extract text-like fragments from nested DataForSEO page payloads.
    """
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        v = value.strip()
        if v:
            out.append(v)
        return out
    if isinstance(value, (int, float, bool)):
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_extract_text_fragments_for_question_coverage(item))
        return out
    if isinstance(value, dict):
        # Prefer these fields first if available.
        preferred_keys = (
            "url",
            "title",
            "description",
            "meta_title",
            "meta_description",
            "h1",
            "h2",
            "text",
            "snippet",
            "content",
        )
        seen = set()
        for key in preferred_keys:
            if key in value:
                seen.add(key)
                out.extend(_extract_text_fragments_for_question_coverage(value.get(key)))
        for key, nested in value.items():
            if key in seen:
                continue
            out.extend(_extract_text_fragments_for_question_coverage(nested))
        return out
    return out


def _crawl_pages_for_aeo(
    *,
    target_domain: Optional[str],
    location_code: Optional[int] = None,
    language_code: str = "en",
    max_pages: int = 20,
    crawl_scope: Optional[str] = None,
    timeout_seconds: int = 120,
    business_profile: Optional[BusinessProfile] = None,
) -> Dict[str, Any]:
    """
    Crawl site pages via DataForSEO On-Page API with bounded polling.
    Returns status metadata with pages.
    """
    normalized_domain = normalize_domain(target_domain) if target_domain else None
    # #region agent log
    _debug.log(
        "dataforseo_utils.py:_crawl_pages_for_aeo:entry",
        "Starting crawl pages for AEO",
        {
            "target_domain": target_domain or "",
            "normalized_domain": normalized_domain or "",
            "max_pages": int(max_pages or 0),
        },
        "H3",
    )
    # #endregion
    logger.info(
        "[AEO debug eb0539] H3 crawl entry target_domain=%s normalized_domain=%s max_pages=%s",
        target_domain,
        normalized_domain,
        max_pages,
    )
    if not normalized_domain:
        # #region agent log
        _debug.log(
            "dataforseo_utils.py:_crawl_pages_for_aeo:no_domain",
            "No normalized domain; returning empty pages",
            {},
            "H3",
        )
        # #endregion
        logger.info("[AEO debug eb0539] H3 crawl early return: no normalized domain")
        return {"pages": [], "aeo_status": "error", "exit_reason": "no_domain", "task_id": None}

    resolved_location_code = int(
        location_code if location_code is not None else getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840)
    )
    scope = (crawl_scope or "default").strip().lower() or "default"
    lock_key = f"aeo_crawl_lock:{scope}:{normalized_domain}:{resolved_location_code}:{language_code}"
    task_key = f"aeo_crawl_task:{scope}:{normalized_domain}:{resolved_location_code}:{language_code}"
    lock_ttl = max(60, min(120, int(timeout_seconds)))
    task_id: Optional[str] = None
    lock_acquired = cache.add(lock_key, "1", lock_ttl)

    if lock_acquired:
        task_data = _post(
            "/v3/on_page/task_post",
            [
                {
                    "target": normalized_domain,
                    "max_crawl_pages": max_pages,
                    "load_resources": False,
                    "enable_javascript": True,
                },
            ],
            business_profile=business_profile,
        )
        if not task_data:
            cache.delete(lock_key)
            return {"pages": [], "aeo_status": "error", "exit_reason": "task_post_empty", "task_id": None}
        try:
            task_id = ((task_data.get("tasks") or [])[0] or {}).get("id")
        except Exception:
            task_id = None
        if not task_id:
            cache.delete(lock_key)
            return {"pages": [], "aeo_status": "error", "exit_reason": "missing_task_id", "task_id": None}
        cache.set(task_key, str(task_id), lock_ttl)
    else:
        cached_task_id = cache.get(task_key)
        if isinstance(cached_task_id, str) and cached_task_id.strip():
            task_id = cached_task_id.strip()
            logger.info("[AEO crawl] Reusing existing task domain=%s task_id=%s", normalized_domain, task_id)
        else:
            return {"pages": [], "aeo_status": "processing", "exit_reason": "lock_in_progress", "task_id": None}

    started_at = time.monotonic()
    attempt = 0
    sleep_seconds = 2.0
    while True:
        attempt += 1
        elapsed = float(time.monotonic() - started_at)
        if elapsed >= float(timeout_seconds):
            return {"pages": [], "aeo_status": "timed_out", "exit_reason": "timeout", "task_id": task_id}

        pages_data = _post(
            "/v3/on_page/pages",
            [
                {
                    "id": task_id,
                    "location_code": resolved_location_code,
                    "language_code": language_code,
                    "limit": max_pages,
                    "offset": 0,
                },
            ],
            business_profile=business_profile,
        )
        if not pages_data:
            return {"pages": [], "aeo_status": "error", "exit_reason": "api_error", "task_id": task_id}

        pages: List[Dict[str, Any]] = []
        task_status_code = 0
        task_status_message = ""
        try:
            tasks = pages_data.get("tasks") or []
            if tasks:
                task_status_code = int((tasks[0] or {}).get("status_code") or 0)
                task_status_message = str((tasks[0] or {}).get("status_message") or "")
            for task in tasks:
                for page_result in (task.get("result") or []):
                    pages.extend((page_result.get("items") or [])[:max_pages])
        except Exception:
            return {"pages": [], "aeo_status": "error", "exit_reason": "parse_error", "task_id": task_id}

        if pages:
            logger.info(
                "[AEO crawl] attempt=%s elapsed=%.2fs next_sleep=0.00 status=%s pages=%s exit_reason=finished_with_pages",
                attempt,
                elapsed,
                task_status_code,
                len(pages),
            )
            return {"pages": pages[:max_pages], "aeo_status": "ready", "exit_reason": "finished_with_pages", "task_id": task_id}

        if task_status_code >= 40000:
            homepage_page = _fetch_homepage_page_for_aeo(normalized_domain)
            if homepage_page:
                logger.info(
                    "[AEO crawl] attempt=%s elapsed=%.2fs next_sleep=0.00 status=%s pages=1 exit_reason=fallback_used",
                    attempt,
                    elapsed,
                    task_status_code,
                )
                return {"pages": [homepage_page], "aeo_status": "ready", "exit_reason": "fallback_used", "task_id": task_id}
            logger.warning(
                "[AEO crawl] terminal_error domain=%s task_id=%s status=%s message=%s",
                normalized_domain,
                task_id,
                task_status_code,
                task_status_message,
            )
            return {"pages": [], "aeo_status": "error", "exit_reason": "api_error", "task_id": task_id}

        next_sleep = min(15.0, sleep_seconds * random.uniform(0.8, 1.2))
        logger.info(
            "[AEO crawl] attempt=%s elapsed=%.2fs next_sleep=%.2fs status=%s pages=%s exit_reason=processing",
            attempt,
            elapsed,
            next_sleep,
            task_status_code,
            len(pages),
        )
        _debug.log(
            "dataforseo_utils.py:_crawl_pages_for_aeo:poll",
            "Polling on_page/pages for crawl completion",
            {
                "attempt": int(attempt),
                "elapsed_time_seconds": round(elapsed, 3),
                "next_sleep_seconds": round(next_sleep, 3),
                "crawl_progress_status": int(task_status_code),
                "pages_found_count": int(len(pages)),
                "exit_reason": "processing",
            },
            "H4",
        )
        remaining = float(timeout_seconds) - elapsed
        time.sleep(min(next_sleep, max(0.05, remaining)))
        sleep_seconds = min(15.0, sleep_seconds * 1.7)


def crawl_pages_for_onboarding(
    *,
    target_domain: Optional[str],
    location_code: Optional[int] = None,
    language_code: str = "en",
    max_pages: int = 10,
    timeout_seconds: int = 150,
    business_profile: Optional[BusinessProfile] = None,
) -> Dict[str, Any]:
    """
    On-Page crawl for onboarding (bounded pages, separate cache locks from AEO).

    Returns:
        pages: list of raw DataForSEO page dicts (up to max_pages)
        exit_reason, task_id, crawl_status: ready | error | timed_out | processing
    """
    normalized_domain = normalize_domain(target_domain) if target_domain else None
    out_base: Dict[str, Any] = {
        "pages": [],
        "exit_reason": "",
        "task_id": None,
        "crawl_status": "error",
    }
    if not normalized_domain:
        out_base["exit_reason"] = "no_domain"
        return out_base

    resolved_location_code = int(
        location_code if location_code is not None else getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840)
    )
    lock_key = f"onboarding_onpage_lock:{normalized_domain}"
    task_key = f"onboarding_onpage_task:{normalized_domain}"
    lock_ttl = max(60, min(180, int(timeout_seconds)))
    task_id: Optional[str] = None
    lock_acquired = cache.add(lock_key, "1", lock_ttl)

    if lock_acquired:
        task_data = _post(
            "/v3/on_page/task_post",
            [
                {
                    "target": normalized_domain,
                    "max_crawl_pages": max(1, min(int(max_pages), 10)),
                    "load_resources": False,
                    "enable_javascript": True,
                },
            ],
            business_profile=business_profile,
        )
        if not task_data:
            cache.delete(lock_key)
            out_base["exit_reason"] = "task_post_empty"
            return out_base
        try:
            task_id = ((task_data.get("tasks") or [])[0] or {}).get("id")
        except Exception:
            task_id = None
        if not task_id:
            cache.delete(lock_key)
            out_base["exit_reason"] = "missing_task_id"
            return out_base
        cache.set(task_key, str(task_id), lock_ttl)
    else:
        cached_task_id = cache.get(task_key)
        if isinstance(cached_task_id, str) and cached_task_id.strip():
            task_id = cached_task_id.strip()
            logger.info(
                "[onboarding onpage] Reusing task domain=%s task_id=%s",
                normalized_domain,
                task_id,
            )
        else:
            out_base["exit_reason"] = "lock_in_progress"
            out_base["crawl_status"] = "processing"
            return out_base

    started_at = time.monotonic()
    attempt = 0
    sleep_seconds = 2.0
    cap = max(1, min(int(max_pages), 10))
    while True:
        attempt += 1
        elapsed = float(time.monotonic() - started_at)
        if elapsed >= float(timeout_seconds):
            out_base["exit_reason"] = "timeout"
            out_base["task_id"] = task_id
            out_base["crawl_status"] = "timed_out"
            return out_base

        pages_data = _post(
            "/v3/on_page/pages",
            [
                {
                    "id": task_id,
                    "location_code": resolved_location_code,
                    "language_code": language_code,
                    "limit": cap,
                    "offset": 0,
                    "order_by": ["page_rank,desc"],
                },
            ],
            business_profile=business_profile,
        )
        if not pages_data:
            out_base["exit_reason"] = "api_error"
            out_base["task_id"] = task_id
            return out_base

        pages: List[Dict[str, Any]] = []
        task_status_code = 0
        task_status_message = ""
        try:
            tasks = pages_data.get("tasks") or []
            if tasks:
                task_status_code = int((tasks[0] or {}).get("status_code") or 0)
                task_status_message = str((tasks[0] or {}).get("status_message") or "")
            for task in tasks:
                for page_result in (task.get("result") or []):
                    pages.extend((page_result.get("items") or [])[:cap])
        except Exception:
            out_base["exit_reason"] = "parse_error"
            out_base["task_id"] = task_id
            return out_base

        if pages:
            logger.info(
                "[onboarding onpage] done domain=%s pages=%s task_id=%s",
                normalized_domain,
                len(pages[:cap]),
                task_id,
            )
            if lock_acquired:
                cache.delete(lock_key)
            return {
                "pages": pages[:cap],
                "exit_reason": "finished_with_pages",
                "task_id": task_id,
                "crawl_status": "ready",
            }

        if task_status_code >= 40000:
            if lock_acquired:
                cache.delete(lock_key)
            homepage_page = _fetch_homepage_page_for_aeo(normalized_domain)
            if homepage_page:
                return {
                    "pages": [homepage_page],
                    "exit_reason": "fallback_used",
                    "task_id": task_id,
                    "crawl_status": "ready",
                }
            logger.warning(
                "[onboarding onpage] terminal_error domain=%s task_id=%s status=%s msg=%s",
                normalized_domain,
                task_id,
                task_status_code,
                task_status_message,
            )
            out_base["exit_reason"] = "api_error"
            out_base["task_id"] = task_id
            return out_base

        next_sleep = min(15.0, sleep_seconds * random.uniform(0.8, 1.2))
        remaining = float(timeout_seconds) - elapsed
        time.sleep(min(next_sleep, max(0.05, remaining)))
        sleep_seconds = min(15.0, sleep_seconds * 1.7)


def _fetch_homepage_page_for_aeo(normalized_domain: str) -> Dict[str, Any] | None:
    """
    Minimal fallback when DataForSEO On-Page returns no page rows.
    Returns a page-like dict consumed by AEO scoring helpers.
    """
    if not normalized_domain:
        return None
    for scheme in ("https://", "http://"):
        url = f"{scheme}{normalized_domain}/"
        try:
            resp = requests.get(
                url,
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0 (compatible; swivl-aeo-debug/1.0)"},
            )
            if resp.status_code >= 400:
                continue
            text = resp.text or ""
            if not text.strip():
                continue
            title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
            title = re.sub(r"\s+", " ", (title_match.group(1) if title_match else "")).strip()
            return {
                "url": url,
                "title": title,
                "content": text,
                "text": text,
            }
        except Exception:
            continue
    return None


def _collect_heading_texts(page: Dict[str, Any]) -> List[str]:
    headings: List[str] = []
    content_value = page.get("content")
    content_dict = content_value if isinstance(content_value, dict) else {}
    page_timing_value = page.get("page_timing")
    page_timing_dict = page_timing_value if isinstance(page_timing_value, dict) else {}
    raw_headings = (
        page.get("headings")
        or content_dict.get("headings")
        or page_timing_dict.get("headings")
        or []
    )
    for row in raw_headings if isinstance(raw_headings, list) else []:
        if isinstance(row, dict):
            txt = str(row.get("text") or row.get("value") or "").strip()
            if txt:
                headings.append(txt)
        elif isinstance(row, str):
            txt = row.strip()
            if txt:
                headings.append(txt)
    return headings


def _contains_faq_schema(value: Any) -> bool:
    """
    Detect FAQPage schema in nested payload values.
    """
    if value is None:
        return False
    if isinstance(value, dict):
        node_type = str(value.get("@type") or value.get("type") or "").strip().lower()
        if node_type == "faqpage":
            return True
        for nested in value.values():
            if _contains_faq_schema(nested):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_faq_schema(v) for v in value)
    if isinstance(value, str):
        v = value.lower()
        return '"@type":"faqpage"' in v.replace(" ", "") or '"@type": "faqpage"' in v
    return False


def compute_faq_readiness_for_pages(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    FAQ readiness:
    - Detect FAQ sections in HTML/text
    - Detect FAQ schema presence
    """
    if not pages:
        return {
            "faq_readiness_score": 0,
            "faq_blocks_found": 0,
            "faq_schema_present": False,
        }

    faq_blocks_found = 0
    faq_schema_present = False
    faq_section_re = re.compile(r"\b(faq|faqs|frequently asked questions)\b", re.IGNORECASE)

    for page in pages:
        if _contains_faq_schema(page):
            faq_schema_present = True

        page_text = " ".join(_extract_text_fragments_for_question_coverage(page))
        if faq_section_re.search(page_text):
            faq_blocks_found += 1
            continue

        headings = _collect_heading_texts(page)
        if any(faq_section_re.search(h) for h in headings):
            faq_blocks_found += 1

    # Isolated scoring: FAQ section presence + FAQ schema presence.
    score = 0
    if faq_blocks_found > 0:
        score += 60
    if faq_schema_present:
        score += 40
    score = max(0, min(100, score))
    # #region agent log
    _debug.log(
        "dataforseo_utils.py:compute_faq_readiness_for_pages:summary",
        "FAQ readiness summary",
        {
            "pages_count": len(pages),
            "faq_blocks_found": int(faq_blocks_found),
            "faq_schema_present": bool(faq_schema_present),
            "score": int(score),
        },
        "H7",
    )
    # #endregion
    logger.info(
        "[AEO debug eb0539] H7 faq summary pages=%s faq_blocks=%s faq_schema=%s score=%s",
        len(pages),
        int(faq_blocks_found),
        bool(faq_schema_present),
        int(score),
    )
    return {
        "faq_readiness_score": int(score),
        "faq_blocks_found": int(faq_blocks_found),
        "faq_schema_present": bool(faq_schema_present),
    }


def _extract_candidate_answer_paragraphs(value: Any) -> List[str]:
    """
    Extract paragraph-like text blocks for snippet readiness checks.
    """
    texts = _extract_text_fragments_for_question_coverage(value)
    out: List[str] = []
    for t in texts:
        words = re.findall(r"\b[\w'-]+\b", t)
        if len(words) >= 20:  # exclude tiny fragments
            out.append(t.strip())
    return out


def compute_snippet_readiness_for_pages(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Snippet readiness:
    - Detect question-style headings
    - Detect short answer paragraph below heading
    - Reward paragraph lengths near 40-60 words
    """
    if not pages:
        return {
            "snippet_readiness_score": 0,
            "answer_blocks_found": 0,
        }

    question_heading_re = re.compile(
        r"^(how|what|why|when|best|can|should|is|are|does|do)\b|.+\?$",
        re.IGNORECASE,
    )
    answer_blocks_found = 0
    paragraphs_near_target = 0

    for page in pages:
        headings = _collect_heading_texts(page)
        question_headings = [h for h in headings if question_heading_re.search(h.strip())]
        if not question_headings:
            continue

        paragraphs = _extract_candidate_answer_paragraphs(page)
        if not paragraphs:
            continue

        # Heuristic: if page has question-style heading and paragraph text exists, count as answer block.
        answer_blocks_found += 1
        for p in paragraphs:
            wc = len(re.findall(r"\b[\w'-]+\b", p))
            if 40 <= wc <= 60:
                paragraphs_near_target += 1
                break

    # Isolated scoring for snippet readiness.
    total_pages = len(pages)
    answer_block_ratio = (answer_blocks_found / total_pages) if total_pages > 0 else 0.0
    near_target_bonus = 20 if paragraphs_near_target > 0 else 0
    score = int(round(min(100.0, (answer_block_ratio * 80.0) + near_target_bonus)))
    # #region agent log
    _debug.log(
        "dataforseo_utils.py:compute_snippet_readiness_for_pages:summary",
        "Snippet readiness summary",
        {
            "pages_count": len(pages),
            "answer_blocks_found": int(answer_blocks_found),
            "paragraphs_near_target": int(paragraphs_near_target),
            "score": int(score),
        },
        "H8",
    )
    # #endregion
    logger.info(
        "[AEO debug eb0539] H8 snippet summary pages=%s answer_blocks=%s near_target=%s score=%s",
        len(pages),
        int(answer_blocks_found),
        int(paragraphs_near_target),
        int(score),
    )
    return {
        "snippet_readiness_score": score,
        "answer_blocks_found": int(answer_blocks_found),
    }


def get_question_coverage_for_site(
    *,
    target_domain: Optional[str],
    niche: Optional[str] = None,
    location_code: Optional[int] = None,
    language_code: str = "en",
    pages: Optional[List[Dict[str, Any]]] = None,
    keywords_rows: Optional[List[Dict[str, Any]]] = None,
    business_profile: Optional[BusinessProfile] = None,
) -> Dict[str, Any]:
    """
    Compute question coverage score by matching question-intent keywords against crawled page content.

    Returns:
    {
      "question_coverage_score": int,
      "questions_found": [str],
      "questions_missing": [str],
    }
    """
    result: Dict[str, Any] = {
        "question_coverage_score": 0,
        "questions_found": [],
        "questions_missing": [],
    }
    try:
        normalized_domain = normalize_domain(target_domain) if target_domain else None
        resolved_keywords_rows = keywords_rows if keywords_rows is not None else get_question_intent_keywords(
            target_domain=normalized_domain,
            niche=niche,
            location_code=location_code,
            language_code=language_code,
            limit=20,
            business_profile=business_profile,
        )
        if not resolved_keywords_rows:
            return result

        weighted_keywords: List[tuple[str, int]] = []
        for row in resolved_keywords_rows:
            kw = str((row or {}).get("keyword") or "").strip().lower()
            if not kw:
                continue
            try:
                sv = int((row or {}).get("search_volume") or 0)
            except Exception:
                sv = 0
            # Keep a floor weight so lower-volume questions still contribute.
            weighted_keywords.append((kw, max(1, sv)))
        if not weighted_keywords:
            return result
        # #region agent log
        _debug.log(
            "dataforseo_utils.py:get_question_coverage_for_site:keywords",
            "Resolved weighted question keywords",
            {
                "keywords_count": len(weighted_keywords),
                "sample_keywords": [kw for kw, _sv in weighted_keywords[:5]],
            },
            "H9",
        )
        # #endregion
        logger.info(
            "[AEO debug eb0539] H9 keywords count=%s sample=%s",
            len(weighted_keywords),
            [kw for kw, _sv in weighted_keywords[:5]],
        )

        if not normalized_domain:
            result["questions_missing"] = [kw for kw, _sv in weighted_keywords]
            return result

        resolved_pages_raw = pages if pages is not None else _crawl_pages_for_aeo(
            target_domain=normalized_domain,
            location_code=location_code,
            language_code=language_code,
            max_pages=20,
            business_profile=business_profile,
        )
        if isinstance(resolved_pages_raw, dict):
            resolved_pages = list(resolved_pages_raw.get("pages") or [])
        else:
            resolved_pages = list(resolved_pages_raw or [])

        if not resolved_pages:
            result["questions_missing"] = [kw for kw, _sv in weighted_keywords]
            return result

        text_fragments: List[str] = []
        for page in resolved_pages:
            text_fragments.extend(_extract_text_fragments_for_question_coverage(page))
        page_text = " ".join(text_fragments).lower()
        if not page_text.strip():
            result["questions_missing"] = [kw for kw, _sv in weighted_keywords]
            return result
        # #region agent log
        _debug.log(
            "dataforseo_utils.py:get_question_coverage_for_site:text",
            "Prepared page text for matching",
            {
                "text_length": len(page_text),
                "fragments_count": len(text_fragments),
                "contains_faq_token": ("faq" in page_text),
            },
            "H9",
        )
        # #endregion
        logger.info(
            "[AEO debug eb0539] H9 page text length=%s fragments=%s contains_faq=%s",
            len(page_text),
            len(text_fragments),
            ("faq" in page_text),
        )

        questions_found: List[str] = []
        questions_missing: List[str] = []
        found_volume = 0
        total_volume = 0
        for kw, sv in weighted_keywords:
            total_volume += int(sv)
            # Require full phrase boundaries to avoid partial/garbage matches.
            pattern = re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.IGNORECASE)
            if pattern.search(page_text):
                questions_found.append(kw)
                found_volume += int(sv)
            else:
                questions_missing.append(kw)

        score = int(round((found_volume / total_volume) * 100)) if total_volume > 0 else 0
        # #region agent log
        _debug.log(
            "dataforseo_utils.py:get_question_coverage_for_site:match_summary",
            "Question coverage match summary",
            {
                "found_count": len(questions_found),
                "missing_count": len(questions_missing),
                "score": int(score),
                "sample_missing": questions_missing[:5],
            },
            "H9",
        )
        # #endregion
        logger.info(
            "[AEO debug eb0539] H9 match summary found=%s missing=%s score=%s",
            len(questions_found),
            len(questions_missing),
            int(score),
        )
        result["question_coverage_score"] = score
        result["questions_found"] = questions_found
        result["questions_missing"] = questions_missing
        return result
    except Exception:
        logger.exception(
            "[DataForSEO] get_question_coverage_for_site failed for domain=%s niche=%s",
            target_domain,
            niche,
        )
        return result


def get_aeo_content_readiness_for_site(
    *,
    target_domain: Optional[str],
    niche: Optional[str] = None,
    cache_key_seed: Optional[str] = None,
    force_refresh: bool = False,
    location_code: Optional[int] = None,
    language_code: str = "en",
    profile_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Aggregate isolated AEO checks:
    - question coverage
    - FAQ readiness
    - snippet readiness
    """
    default_result: Dict[str, Any] = {
        "aeo_status": "ready",
        "aeo_last_computed_at": None,
        "question_coverage_score": 0,
        "questions_found": [],
        "questions_missing": [],
        "faq_readiness_score": 0,
        "faq_blocks_found": 0,
        "faq_schema_present": False,
        "snippet_readiness_score": 0,
        "answer_blocks_found": 0,
        "aeo_score": 0,
    }
    try:
        usage_profile: Optional[BusinessProfile] = None
        if profile_id:
            usage_profile = BusinessProfile.objects.filter(pk=profile_id).first()
        normalized_domain = normalize_domain(target_domain) if target_domain else ""
        normalized_niche = (niche or "").strip().lower()
        cache_seed = (cache_key_seed or "").strip().lower()
        cache_key = (
            f"aeo_readiness:{cache_seed}:{normalized_domain}:{normalized_niche}:"
            f"{int(location_code if location_code is not None else getattr(settings, 'DATAFORSEO_LOCATION_CODE', 2840))}:{language_code}"
        )
        cache_ttl_seconds = int(AEO_SNAPSHOT_TTL.total_seconds())
        if not force_refresh:
            cached = cache.get(cache_key)
            if isinstance(cached, dict):
                # #region agent log
                _debug.log(
                    "dataforseo_utils.py:get_aeo_content_readiness_for_site:cache_hit",
                    "AEO readiness cache hit",
                    {
                        "cache_key_seed": cache_seed,
                        "normalized_domain": normalized_domain,
                        "force_refresh": bool(force_refresh),
                    },
                    "H2",
                )
                # #endregion
                logger.info(
                    "[AEO debug eb0539] H2 cache hit domain=%s niche=%s key=%s",
                    normalized_domain,
                    normalized_niche,
                    cache_key,
                )
                return {**default_result, **cached}

        question_keywords = get_question_intent_keywords(
            target_domain=target_domain,
            niche=niche,
            location_code=location_code,
            language_code=language_code,
            limit=20,
            business_profile=usage_profile,
        )
        crawl_result = _crawl_pages_for_aeo(
            target_domain=normalized_domain,
            location_code=location_code,
            language_code=language_code,
            max_pages=20,
            crawl_scope=cache_seed or f"profile:{profile_id or 'unknown'}",
            timeout_seconds=120,
            business_profile=usage_profile,
        )
        if isinstance(crawl_result, list):
            pages = list(crawl_result)
            aeo_status = "ready"
            exit_reason = "legacy_pages_list"
        else:
            pages = list((crawl_result or {}).get("pages") or [])
            aeo_status = str((crawl_result or {}).get("aeo_status") or "ready")
            exit_reason = str((crawl_result or {}).get("exit_reason") or "")
        now_iso = datetime.now(timezone.utc).isoformat()

        if aeo_status in {"processing", "timed_out"}:
            snapshot = None
            if profile_id:
                try:
                    snapshot = AEOOverviewSnapshot.objects.filter(
                        profile_id=profile_id,
                        domain=normalized_domain or "",
                        location_code=int(
                            location_code
                            if location_code is not None
                            else getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840)
                        ),
                    ).first()
                except Exception:
                    snapshot = None
            if snapshot:
                snapshot_result = {
                    **default_result,
                    "aeo_status": aeo_status,
                    "aeo_last_computed_at": snapshot.refreshed_at.isoformat() if snapshot.refreshed_at else now_iso,
                    "question_coverage_score": int(snapshot.question_coverage_score or 0),
                    "questions_found": list(snapshot.questions_found or []),
                    "questions_missing": list(snapshot.questions_missing or []),
                    "faq_readiness_score": int(snapshot.faq_readiness_score or 0),
                    "faq_blocks_found": int(snapshot.faq_blocks_found or 0),
                    "faq_schema_present": bool(snapshot.faq_schema_present),
                    "snippet_readiness_score": int(snapshot.snippet_readiness_score or 0),
                    "answer_blocks_found": int(snapshot.answer_blocks_found or 0),
                }
                logger.info(
                    "[AEO crawl] status=%s exit_reason=%s using_snapshot profile_id=%s domain=%s",
                    aeo_status,
                    exit_reason,
                    profile_id,
                    normalized_domain,
                )
                return snapshot_result
            degraded = {
                **default_result,
                "aeo_status": aeo_status,
                "aeo_last_computed_at": None,
            }
            logger.warning(
                "[AEO crawl] status=%s exit_reason=%s no_snapshot profile_id=%s domain=%s",
                aeo_status,
                exit_reason,
                profile_id,
                normalized_domain,
            )
            return degraded

        question_data = get_question_coverage_for_site(
            target_domain=target_domain,
            niche=niche,
            location_code=location_code,
            language_code=language_code,
            pages=pages,
            keywords_rows=question_keywords,
            business_profile=usage_profile,
        )
        faq_data = compute_faq_readiness_for_pages(pages)
        snippet_data = compute_snippet_readiness_for_pages(pages)
        result = {
            **default_result,
            "aeo_status": "ready",
            "aeo_last_computed_at": now_iso,
            **(question_data or {}),
            **(faq_data or {}),
            **(snippet_data or {}),
        }
        # Canonical AEO score from the crawl-derived readiness metrics.
        q_score = int(result.get("question_coverage_score") or 0)
        faq_score = int(result.get("faq_readiness_score") or 0)
        snip_score = int(result.get("snippet_readiness_score") or 0)
        result["aeo_score"] = max(0, min(100, int(round((q_score * 0.4) + (faq_score * 0.3) + (snip_score * 0.3)))))
        # #region agent log
        _debug.log(
            "dataforseo_utils.py:get_aeo_content_readiness_for_site:computed",
            "Computed AEO readiness result",
            {
                "normalized_domain": normalized_domain,
                "pages_count": len(pages or []),
                "question_coverage_score": int(result.get("question_coverage_score") or 0),
                "faq_readiness_score": int(result.get("faq_readiness_score") or 0),
                "snippet_readiness_score": int(result.get("snippet_readiness_score") or 0),
                "questions_found_count": len(result.get("questions_found") or []),
                "questions_missing_count": len(result.get("questions_missing") or []),
                "aeo_status": result.get("aeo_status"),
                "exit_reason": exit_reason,
            },
            "H5",
        )
        # #endregion
        logger.info(
            "[AEO debug eb0539] H5 computed domain=%s pages=%s q=%s faq=%s snip=%s found=%s missing=%s",
            normalized_domain,
            len(pages or []),
            int(result.get("question_coverage_score") or 0),
            int(result.get("faq_readiness_score") or 0),
            int(result.get("snippet_readiness_score") or 0),
            len(result.get("questions_found") or []),
            len(result.get("questions_missing") or []),
        )
        cache.set(cache_key, result, cache_ttl_seconds)
        return result
    except Exception:
        logger.exception(
            "[DataForSEO] get_aeo_content_readiness_for_site failed for domain=%s niche=%s",
            target_domain,
            niche,
        )
        return default_result


def get_keyword_gap_keywords(
    target_domain: str,
    competitor_domains: List[str],
    *,
    location_code: int,
    language_code: str = "en",
    limit: int = 100,
    business_profile: Optional[BusinessProfile] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Call domain_intersection/live to compute keyword gaps vs competitors.

    For now we return a simplified list of items suitable for the frontend SEO keywords table:
    - keyword
    - search_volume

    Successful merged results are cached (~30 days by default) so refresh / Celery re-enrichment
    does not re-bill DataForSEO per competitor. Use ``force_refresh=True`` or settings
    ``DATAFORSEO_DOMAIN_INTERSECTION_FORCE_REFRESH`` to bypass.
    """
    cleaned_competitors = sorted({c.strip().lower() for c in competitor_domains if c.strip()})
    if not cleaned_competitors:
        logger.info(
            "[DataForSEO] domain_intersection skipped for %s: no competitors configured",
            target_domain,
        )
        return []

    norm_target = (target_domain or "").strip().lower()
    intersections_flag = True
    cache_ttl = int(getattr(settings, "DATAFORSEO_DOMAIN_INTERSECTION_CACHE_TTL", 30 * 24 * 3600))
    key_raw = (
        f"{DOMAIN_INTERSECTION_CACHE_VERSION}|{norm_target}|"
        f"{'|'.join(cleaned_competitors)}|{int(location_code)}|{language_code}|"
        f"{int(intersections_flag)}|{int(limit)}"
    )
    cache_key = "dfs:di:" + hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    global_force = bool(getattr(settings, "DATAFORSEO_DOMAIN_INTERSECTION_FORCE_REFRESH", False))

    if not force_refresh and not global_force:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(
                "[DataForSEO] domain_intersection cache_hit key_suffix=%s target=%s competitors_n=%s limit=%s",
                cache_key[-12:],
                norm_target,
                len(cleaned_competitors),
                int(limit),
            )
            return list(cached)

    logger.info(
        "[DataForSEO] domain_intersection cache_miss key_suffix=%s target=%s competitors_n=%s limit=%s force_refresh=%s",
        cache_key[-12:],
        norm_target,
        len(cleaned_competitors),
        int(limit),
        bool(force_refresh or global_force),
    )

    # The domain_intersection endpoint expects target1/target2 rather than a
    # "targets" array. Call it once per competitor and merge the results.
    # We also collect a short list of competitor URLs that are currently
    # ranking for each keyword so the frontend can show where competitors
    # dominate the SERP.
    aggregated: Dict[str, Dict[str, Any]] = {}

    for competitor in cleaned_competitors:
        payload = [
            {
                "target1": target_domain,
                "target2": competitor,
                "location_code": int(location_code),
                "language_code": language_code,
                # intersections=true ensures we only get queries where both
                # domains have ranking URLs in the SERP.
                "intersections": True,
                "limit": int(limit),
            },
        ]

        data = _post(
            "/v3/dataforseo_labs/google/domain_intersection/live",
            payload,
            business_profile=business_profile,
        )
        if not data:
            continue
        try:
            tasks = data.get("tasks") or []
            if not tasks:
                logger.warning(
                    "[DataForSEO] domain_intersection: no tasks in response for %s vs %s",
                    target_domain,
                    competitor,
                )
                continue

            for task in tasks:
                results = task.get("result") or []
                if not results:
                    logger.warning(
                        "[DataForSEO] domain_intersection: no result for %s vs %s",
                        target_domain,
                        competitor,
                    )
                    continue

                for result in results:
                    items = result.get("items") or []
                    for item in items:
                        keyword_data = item.get("keyword_data") or {}
                        kw = keyword_data.get("keyword") or item.get("keyword")
                        if not kw:
                            continue

                        keyword_info = keyword_data.get("keyword_info") or {}
                        search_volume = (
                            keyword_info.get("search_volume")
                            or keyword_info.get("search_volume_global")
                            or keyword_info.get("sum_search_volume")
                            or item.get("search_volume")
                            or item.get("sum_search_volume")
                        )

                        # For each keyword we want:
                        # - our (target1) rank (first_domain_serp_element)
                        # - competitor rank (second_domain_serp_element)
                        # The UI needs "You rank #X" plus missed-search estimates.
                        your_serp_el = item.get("first_domain_serp_element") or {}
                        comp_serp_el = item.get("second_domain_serp_element") or {}

                        your_rank = your_serp_el.get("rank_absolute")
                        your_url = your_serp_el.get("url") or ""
                        your_domain = (
                            your_serp_el.get("main_domain")
                            or your_serp_el.get("domain")
                            or target_domain
                        )

                        comp_domain = (
                            comp_serp_el.get("main_domain")
                            or comp_serp_el.get("domain")
                            or competitor
                        )
                        comp_url = comp_serp_el.get("url") or ""
                        comp_rank = comp_serp_el.get("rank_absolute")

                        try:
                            your_rank_int: Optional[int] = int(your_rank) if your_rank is not None else None
                        except (TypeError, ValueError):
                            your_rank_int = None

                        if your_rank_int is not None and your_rank_int <= 0:
                            your_rank_int = None

                        existing = aggregated.get(kw)
                        if existing is None:
                            competitors_list: List[Dict[str, Any]] = []
                            if comp_url:
                                competitors_list.append(
                                    {
                                        "domain": comp_domain,
                                        "url": comp_url,
                                        "rank": comp_rank,
                                    },
                                )
                            top_competitor = _format_company_name_or_url(
                                domain=comp_domain,
                                url=comp_url,
                            )
                            top_competitor_domain = comp_domain or None
                            top_competitor_rank = comp_rank
                            aggregated[kw] = {
                                "keyword": kw,
                                "search_volume": search_volume,
                                "your_domain": your_domain,
                                "your_url": your_url,
                                "your_rank": your_rank_int,
                                "competitors": competitors_list,
                                "top_competitor": top_competitor,
                                "top_competitor_domain": top_competitor_domain,
                                "top_competitor_rank": top_competitor_rank,
                            }
                        else:
                            # Keep the highest search volume we have seen.
                            prev_sv = existing.get("search_volume") or 0
                            curr_sv = search_volume or 0
                            if curr_sv > prev_sv:
                                existing["search_volume"] = search_volume

                            # Keep the best (lowest) rank we have seen for the target.
                            prev_your_rank = existing.get("your_rank")
                            if your_rank_int is not None:
                                if prev_your_rank is None or (
                                    isinstance(prev_your_rank, int) and your_rank_int < prev_your_rank
                                ):
                                    existing["your_rank"] = your_rank_int
                                    if your_url:
                                        existing["your_url"] = your_url

                            # Merge competitor URLs, keeping at most 3 per keyword,
                            # preferring better (lower) ranks.
                            if comp_url:
                                comp_entry = {
                                    "domain": comp_domain,
                                    "url": comp_url,
                                    "rank": comp_rank,
                                }
                                existing_list: List[Dict[str, Any]] = existing.setdefault("competitors", [])

                                # Avoid exact duplicates (same domain + URL).
                                if not any(
                                    c.get("domain") == comp_entry["domain"] and c.get("url") == comp_entry["url"]
                                    for c in existing_list
                                ):
                                    existing_list.append(comp_entry)

                                # Keep only top 3 by rank (ascending), with None ranks sorted last.
                                existing_list.sort(
                                    key=lambda c: (c.get("rank") or 10_000_000),
                                )
                                if len(existing_list) > 3:
                                    del existing_list[3:]

                            # Refresh top competitor after merge/sort.
                            best_comp = None
                            try:
                                comps = existing.get("competitors") or []
                                if comps:
                                    best_comp = comps[0]
                            except Exception:
                                best_comp = None
                            if best_comp:
                                existing["top_competitor"] = _format_company_name_or_url(
                                    domain=best_comp.get("domain"),
                                    url=best_comp.get("url"),
                                )
                                existing["top_competitor_domain"] = best_comp.get("domain")
                                existing["top_competitor_rank"] = best_comp.get("rank")

        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "[DataForSEO] domain_intersection parsing failed for %s vs %s: %s",
                target_domain,
                competitor,
                exc,
            )
            continue

    gap_keywords = list(aggregated.values())

    logger.info(
        "[DataForSEO] domain_intersection target=%s competitors=%s keyword_count=%s",
        target_domain,
        ",".join(cleaned_competitors),
        len(gap_keywords),
    )
    try:
        cache.set(cache_key, gap_keywords, cache_ttl)
    except Exception:
        logger.exception(
            "[DataForSEO] domain_intersection cache_set failed key_suffix=%s",
            cache_key[-12:],
        )
    return gap_keywords


def compute_professional_seo_score(
    *,
    estimated_traffic: float,
    keywords_count: int,
    top3_positions: int,
    top10_positions: int,
    avg_keyword_difficulty: Optional[float],
    competitor_avg_traffic: float,
) -> int:
    """
    Collapse core SEO metrics into a single 0–100 "professional grade" score using:
    - estimated organic traffic (via CTR curve)
    - keyword breadth
    - ranking quality (share in top 3 / top 10)
    - keyword difficulty strength
    - competitive market share vs. top competitors

    All components are soft-scaled so smaller sites can still make meaningful progress.
    """
    try:
        traffic = max(float(estimated_traffic), 0.0)
        k = max(int(keywords_count), 0)
        t = max(int(top3_positions), 0)
        t10 = max(int(top10_positions), 0)
    except (TypeError, ValueError):
        traffic, k, t, t10 = 0.0, 0, 0, 0

    import math

    # 1) Traffic visibility: log-scaled estimated organic traffic.
    # 0 → 0, 100 → ~30, 1k → ~55, 10k → ~75, 100k → ~90, 1M+ → ~100
    if traffic <= 0:
        visibility_score = 0.0
    else:
        visibility_score = min(100.0, (math.log10(traffic + 10.0) / 6.0) * 100.0)

    # 2) Keyword breadth: log-scaled number of ranking keywords.
    # 1 → ~15, 10 → ~35, 100 → ~60, 1k → ~85, 10k → ~100
    if k <= 0:
        breadth_score = 0.0
    else:
        breadth_score = min(100.0, (math.log10(k + 1.0) / 4.0) * 100.0)

    # 3) Ranking quality: combination of share in top 3 and share in top 10.
    if k <= 0:
        ranking_score = 0.0
    else:
        top3_share = min(t / k, 1.0)
        top10_share = min(t10 / k, 1.0)
        ranking_score = (top3_share * 0.7 + top10_share * 0.3) * 100.0

    # 4) Keyword difficulty strength: reward sites ranking on harder keywords.
    # We treat difficulty as 0–100 where higher means more competitive queries.
    if avg_keyword_difficulty is None:
        difficulty_score = 0.0
    else:
        d = max(0.0, min(100.0, float(avg_keyword_difficulty)))
        # Slightly compress extremes so very hard portfolios don't instantly max out.
        difficulty_score = 10.0 + (d * 0.8)
        difficulty_score = max(0.0, min(100.0, difficulty_score))

    # 5) Competitive market share: share of traffic vs. average competitor.
    if traffic <= 0 or competitor_avg_traffic <= 0:
        market_share_score = 0.0
    else:
        # ratio > 1 means above-average vs. top competitors.
        ratio = traffic / max(competitor_avg_traffic, 1e-6)
        # Soft saturation: 0.5 → ~33, 1 → ~66, 2 → ~85, 3+ → ~95
        market_share_score = min(
            95.0,
            (math.log10(ratio + 1.0) / math.log10(4.0)) * 100.0,
        )

    # Weighted blend:
    # - visibility: 40%
    # - breadth: 20%
    # - ranking quality: 15%
    # - keyword difficulty: 15%
    # - competitive strength: 10%
    final = (
        visibility_score * 0.40
        + breadth_score * 0.20
        + ranking_score * 0.15
        + difficulty_score * 0.15
        + market_share_score * 0.10
    )

    return max(0, min(100, int(round(final))))


# -----------------------------------------------------------------------------
# SEO pipeline helpers (orchestration only in get_or_refresh_seo_score_for_user)
# -----------------------------------------------------------------------------


def normalize_domain(site_url: Optional[str]) -> Optional[str]:
    """Extract and normalize domain from site_url; strip www. Return None if empty."""
    if not site_url or not str(site_url).strip():
        return None
    parsed = urlparse(site_url)
    domain = (parsed.netloc or parsed.path or "").strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain if domain else None


def seo_snapshot_context_for_profile(
    profile: Optional[BusinessProfile],
) -> tuple[str, int]:
    """
    Resolve (cached_location_mode, cached_location_code) for SEOOverviewSnapshot rows.

    Matches the snapshot scoping used in views and get_cached_seo_snapshot.
    """
    if profile is None:
        return "organic", 0
    mode = str(getattr(profile, "seo_location_mode", "organic") or "organic").strip().lower()
    if mode != "local":
        return "organic", 0
    default_location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    resolved_code, _fallback, _label = get_profile_location_code(
        profile, default_location_code
    )
    return "local", int(resolved_code or 0)


def get_cached_seo_snapshot(
    business_profile: Optional[BusinessProfile],
    domain: str,
    period_start,
    cache_ttl: timedelta,
    now_utc: datetime,
    location_mode: str = "organic",
    location_code: int = 0,
) -> Optional[Any]:
    """
    Return a fresh SEOOverviewSnapshot for this business profile/period if cache is valid
    (refreshed within cache_ttl and cached_domain matches). Otherwise return None.
    """
    if not business_profile:
        return None
    try:
        snapshot = SEOOverviewSnapshot.objects.filter(
            business_profile=business_profile,
            period_start=period_start,
            cached_location_mode=str(location_mode or "organic"),
            cached_location_code=int(location_code or 0),
        ).first()
        if not snapshot or not snapshot.refreshed_at or not snapshot.cached_domain:
            return None
        age = now_utc - snapshot.refreshed_at
        if age > cache_ttl:
            return None
        if (snapshot.cached_domain or "").strip().lower() != domain:
            return None
        return snapshot
    except Exception:
        logger.exception(
            "[SEO score] Error reading keyword cache for profile_id=%s",
            getattr(business_profile, "id", None),
        )
        return None


def fetch_ranked_keyword_items(
    domain: str,
    location_code: int,
    language_code: str,
    user=None,
    *,
    limit: int = 100,
    business_profile: Optional[BusinessProfile] = None,
) -> List[Dict[str, Any]]:
    """
    Call DataForSEO Labs ranked_keywords/live and return parsed items list.
    Returns [] on any failure. Logs request and first-item preview.
    """
    payload = [
        {
            "target": domain,
            "location_code": int(location_code),
            "language_code": language_code,
            "limit": int(limit),
        },
    ]
    logger.info(
        "[SEO score] ranked_keywords request user_id=%s domain=%s payload=%s",
        getattr(user, "id", None),
        domain,
        payload,
    )
    ranked_data = _post(
        "/v3/dataforseo_labs/google/ranked_keywords/live",
        payload,
        business_profile=business_profile,
    )
    # #region agent log
    _debug.log(
        "dataforseo_utils.py:fetch_ranked_keyword_items:response_shape",
        "Ranked keywords raw response shape",
        {
            "domain": domain,
            "ranked_data_type": type(ranked_data).__name__ if ranked_data is not None else "NoneType",
            "top_keys": list((ranked_data or {}).keys())[:8] if isinstance(ranked_data, dict) else [],
            "tasks_count": len((ranked_data or {}).get("tasks") or []) if isinstance(ranked_data, dict) else -1,
        },
        "S1",
    )
    # #endregion
    _dbg_ba84ae_log(
        hypothesisId="H0_ranked_keywords_request_response",
        location="accounts/dataforseo_utils.py:fetch_ranked_keyword_items",
        message="ranked_keywords response received",
        data={
            "domain": domain,
            "has_ranked_data": ranked_data is not None,
            "top_level_has_tasks": bool((ranked_data or {}).get("tasks")),
        },
        runId="pre-fix",
    )
    if not ranked_data:
        logger.warning(
            "[SEO score] ranked_keywords returned no data for user_id=%s domain=%s",
            getattr(user, "id", None),
            domain,
        )
        _dbg_ba84ae_log(
            hypothesisId="H0_ranked_keywords_no_data",
            location="accounts/dataforseo_utils.py:fetch_ranked_keyword_items",
            message="ranked_keywords: ranked_data falsy/None; returning []",
            data={"domain": domain},
            runId="pre-fix",
        )
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "098bfd",
                    "runId": "pre-fix",
                    "hypothesisId": "H3",
                    "location": "accounts/dataforseo_utils.py:fetch_ranked_keyword_items",
                    "message": "No ranked_data; using fallback SEO score",
                    "data": {"user_id": getattr(user, "id", None), "domain": domain},
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }) + "\n")
        except Exception:
            pass
        return []
    try:
        tasks = ranked_data.get("tasks") or []
        if not tasks:
            # #region agent log
            _debug.log(
                "dataforseo_utils.py:fetch_ranked_keyword_items:no_tasks",
                "Ranked keywords has no tasks",
                {"domain": domain, "raw_preview": str(ranked_data)[:300]},
                "S2",
            )
            # #endregion
            raise ValueError("no tasks")
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            raise ValueError("no results")
        result = results[0]
        items = result.get("items") or []
        # #region agent log
        _debug.log(
            "dataforseo_utils.py:fetch_ranked_keyword_items:parsed_items",
            "Parsed ranked keyword items",
            {
                "domain": domain,
                "items_count": len(items),
                "task_status_code": task.get("status_code"),
                "task_status_message": task.get("status_message"),
            },
            "S3",
        )
        # #endregion
        logger.info(
            "[SEO score] ranked_keywords parsed user_id=%s domain=%s task_status=%s items=%s",
            getattr(user, "id", None),
            domain,
            task.get("status_code"),
            len(items),
        )
        if items:
            preview = {
                "rank_absolute": items[0].get("rank_absolute"),
                "rank_group": items[0].get("rank_group"),
                "search_volume": (
                    ((items[0].get("keyword_data") or {}).get("keyword_info") or {}).get("search_volume")
                    or items[0].get("search_volume")
                ),
                "keyword": (items[0].get("keyword_data") or {}).get("keyword") or items[0].get("keyword"),
            }
            logger.info(
                "[SEO score] ranked_keywords first_item_preview user_id=%s domain=%s preview=%s",
                getattr(user, "id", None),
                domain,
                preview,
            )

            # Debug evidence: are rank fields present in the raw DataForSEO response?
            _dbg_ba84ae_log(
                hypothesisId="H1_rank_fields_present_in_ranked_keywords",
                location="accounts/dataforseo_utils.py:fetch_ranked_keyword_items",
                message="raw ranked_keywords rank fields sample",
                data={
                    "domain": domain,
                    "total_items": len(items),
                    "first_item": {
                        "rank_absolute": items[0].get("rank_absolute"),
                        "rank_group": items[0].get("rank_group"),
                        "keyword": (items[0].get("keyword_data") or {}).get("keyword")
                        or items[0].get("keyword"),
                        "search_volume": (
                            ((items[0].get("keyword_data") or {}).get("keyword_info") or {}).get("search_volume")
                            or items[0].get("search_volume")
                        ),
                    },
                },
                runId="pre-fix",
            )
        return items
    except Exception as exc:
        _dbg_ba84ae_log(
            hypothesisId="H0_ranked_keywords_parse_exception",
            location="accounts/dataforseo_utils.py:fetch_ranked_keyword_items",
            message="ranked_keywords parsing exception; returning []",
            data={"domain": domain, "exc_type": type(exc).__name__, "exc_msg": str(exc)[:200]},
            runId="pre-fix",
        )
        logger.exception(
            "[SEO score] Error parsing ranked_keywords for user_id=%s domain=%s raw_preview=%s",
            getattr(user, "id", None),
            domain,
            str(ranked_data)[:200],
        )
        return []


def display_rank_for_keyword_sort(row: Dict[str, Any]) -> Optional[int]:
    """Best positive display rank: organic ``rank``, else ``local_verified_rank`` (matches Keywords UI)."""
    r = row.get("rank")
    lv = row.get("local_verified_rank")
    if isinstance(r, int) and r > 0:
        return int(r)
    if isinstance(lv, int) and lv > 0:
        return int(lv)
    return None


def sort_top_keywords_for_display(
    rows: List[Dict[str, Any]],
    *,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Snapshot/API/UI order: ranked rows first (lowest position number first), then unranked
    by search_volume descending. ``max_rows`` optionally caps persistence (safety bound).
    """
    def sort_key(row: Dict[str, Any]) -> tuple[Any, ...]:
        pos = display_rank_for_keyword_sort(row)
        sv = row.get("search_volume") or 0
        try:
            sv_f = float(sv)
        except (TypeError, ValueError):
            sv_f = 0.0
        kw = str(row.get("keyword") or "").lower()
        if pos is not None:
            return (0, pos, -sv_f, kw)
        return (1, -sv_f, kw)

    out = sorted(list(rows or []), key=sort_key)
    if max_rows is not None and max_rows > 0 and len(out) > max_rows:
        return out[:max_rows]
    return out


def compute_ranked_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    From ranked_keywords API items, compute estimated_traffic, keywords_ranking,
    top3_positions, top10_positions, avg_difficulty, total_search_volume (items only),
    and top_keywords list (ranked + unranked opportunities from items).
    """
    MIN_SEARCH_VOLUME = 10
    estimated_traffic = 0.0
    keywords_ranking = len(items)
    top3_positions = 0
    top10_positions = 0
    difficulties: List[float] = []
    total_search_volume = 0.0
    top_keywords: List[Dict[str, Any]] = []
    rank_int_none_count = 0
    rank_int_positive_count = 0
    rank_int_other_count = 0

    for item in items:
        rank = item.get("rank_absolute") or item.get("rank_group")
        try:
            rank_int = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank_int = None

        if rank_int is None:
            rank_int_none_count += 1
        elif rank_int > 0:
            rank_int_positive_count += 1
        else:
            rank_int_other_count += 1

        keyword_data = item.get("keyword_data") or {}
        kw_info = keyword_data.get("keyword_info") or {}
        search_volume = (
            kw_info.get("search_volume")
            or kw_info.get("search_volume_global")
            or kw_info.get("sum_search_volume")
            or item.get("search_volume")
            or item.get("sum_search_volume")
            or 0
        )
        try:
            sv = float(search_volume)
        except (TypeError, ValueError):
            sv = 0.0

        if sv > 0:
            total_search_volume += sv

        keyword_text = keyword_data.get("keyword") or item.get("keyword")
        if keyword_text and sv > 0:
            if rank_int is not None and rank_int > 0:
                top_keywords.append(
                    {
                        "keyword": str(keyword_text),
                        "search_volume": int(sv),
                        "rank": int(rank_int),
                        "local_verified_rank": None,
                        "rank_source": "baseline",
                        "keyword_origin": "ranked",
                        "missed_searches_monthly": _estimate_missed_searches_monthly(sv, rank_int),
                        "top_competitor": None,
                        "top_competitor_domain": None,
                        "top_competitor_rank": None,
                        "competitors": [],
                    }
                )
            elif rank_int is None and sv >= MIN_SEARCH_VOLUME:
                top_keywords.append(
                    {
                        "keyword": str(keyword_text),
                        "search_volume": int(sv),
                        "rank": None,
                        "local_verified_rank": None,
                        "rank_source": "baseline",
                        "keyword_origin": "ranked",
                        "missed_searches_monthly": _estimate_missed_searches_monthly(sv, None),
                        "top_competitor": None,
                        "top_competitor_domain": None,
                        "top_competitor_rank": None,
                        "competitors": [],
                    }
                )

        if rank_int is not None and rank_int > 0 and sv > 0:
            ctr = _ctr_for_position(rank_int)
            estimated_traffic += sv * ctr
            if rank_int <= 3:
                top3_positions += 1
            if rank_int <= 10:
                top10_positions += 1

        diff = _extract_keyword_difficulty(kw_info)
        if diff is not None:
            difficulties.append(diff)

    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else None

    # Debug evidence: how often does DataForSEO give us rank values?
    _dbg_ba84ae_log(
        hypothesisId="H1_compute_ranked_metrics_rank_int_distribution",
        location="accounts/dataforseo_utils.py:compute_ranked_metrics",
        message="rank_int distribution for ranked_keywords items",
        data={
            "total_items": len(items),
            "rank_int_none_count": rank_int_none_count,
            "rank_int_positive_count": rank_int_positive_count,
            "rank_int_other_count": rank_int_other_count,
            "top_keywords_total": len(top_keywords),
            "top_keywords_with_rank": sum(1 for k in top_keywords if k.get("rank") is not None),
            "top_keywords_rank_positive": sum(1 for k in top_keywords if isinstance(k.get("rank"), int) and (k.get("rank") or 0) > 0),
        },
        runId="pre-fix",
    )
    return {
        "estimated_traffic": estimated_traffic,
        "keywords_ranking": keywords_ranking,
        "top3_positions": top3_positions,
        "top10_positions": top10_positions,
        "avg_difficulty": avg_difficulty,
        "total_search_volume": total_search_volume,
        "top_keywords": top_keywords,
    }


def enrich_with_gap_keywords(
    domain: str,
    location_code: int,
    language_code: str,
    user,
    top_keywords: List[Dict[str, Any]],
    force_refresh: bool = False,
) -> None:
    """Append gap keywords (vs competitors) to top_keywords in place. Preserves logging on error."""
    rank_non_null_before = sum(1 for k in top_keywords if k.get("rank") is not None)
    try:
        _profile = (
            BusinessProfile.objects.filter(user=user, is_main=True).first()
            or BusinessProfile.objects.filter(user=user).first()
        )
    except Exception:
        _profile = None
    competitor_selection = get_competitors_for_domain_intersection(
        domain=domain,
        location_code=location_code,
        language_code=language_code,
        user=user,
        profile=_profile,
        limit=5,
        min_competitors=int(getattr(settings, "DATAFORSEO_MIN_COMPETITORS_FOR_GAP", 2)),
    )
    competitor_domains = competitor_selection.get("filtered_competitors_used") or []

    existing_keys = {str(k.get("keyword", "")).lower() for k in top_keywords if k.get("keyword")}
    existing_by_key: Dict[str, Dict[str, Any]] = {
        str(k.get("keyword", "")).lower(): k for k in top_keywords if k.get("keyword")
    }

    if not competitor_domains:
        rank_non_null_after = sum(1 for k in top_keywords if k.get("rank") is not None)
        logger.warning(
            "[SEO score] No quality competitors for domain_intersection. domain=%s rank_non_null_before=%s rank_non_null_after=%s",
            domain,
            rank_non_null_before,
            rank_non_null_after,
        )
        _dbg_ba84ae_log(
            hypothesisId="H8_competitor_filter_no_competitors",
            location="accounts/dataforseo_utils.py:enrich_with_gap_keywords",
            message="no competitor domains after quality filter/override; skipping domain_intersection enrichment",
            data={
                "domain": domain,
                "rank_non_null_before": rank_non_null_before,
                "rank_non_null_after": rank_non_null_after,
                "competitor_source": competitor_selection.get("competitor_source"),
            },
            runId="pre-fix",
        )
        return
    try:
        gap_items = get_keyword_gap_keywords(
            domain,
            competitor_domains,
            location_code=location_code,
            language_code=language_code,
            limit=50,
            business_profile=_profile,
            force_refresh=force_refresh,
        )
        gap_items_processed = 0
        gap_items_with_rank = 0
        gap_items_without_rank = 0
        gap_items_existing_key_with_rank = 0
        gap_items_existing_key_rank_updated = 0
        for item in gap_items:
            kw = item.get("keyword")
            if not kw:
                continue
            key_lower = str(kw).lower()
            existing_entry = existing_by_key.get(key_lower)
            sv_gap = item.get("search_volume") or 0
            try:
                sv_gap_f = float(sv_gap)
            except (TypeError, ValueError):
                sv_gap_f = 0.0
            if sv_gap_f <= 0:
                continue

            # domain_intersection includes our own rank (first_domain_serp_element)
            # when intersections=true; use it so the UI can show "You rank #X".
            your_rank = item.get("your_rank")
            try:
                your_rank_int: Optional[int] = int(your_rank) if your_rank is not None else None
            except (TypeError, ValueError):
                your_rank_int = None
            if your_rank_int is not None and your_rank_int <= 0:
                your_rank_int = None

            missed = _estimate_missed_searches_monthly(sv_gap_f, your_rank_int)

            gap_items_processed += 1
            if your_rank_int is not None and your_rank_int > 0:
                gap_items_with_rank += 1
            else:
                gap_items_without_rank += 1
            if existing_entry is not None:
                # If the keyword already exists but we previously had rank=null,
                # overwrite it with the gap-provided rank (when available).
                if your_rank_int is not None and your_rank_int > 0:
                    gap_items_existing_key_with_rank += 1
                    try:
                        existing_rank = existing_entry.get("rank")
                        existing_rank_int = int(existing_rank) if existing_rank is not None else None
                    except (TypeError, ValueError):
                        existing_rank_int = None

                    if existing_rank_int is None or (existing_rank_int is not None and existing_rank_int > your_rank_int):
                        existing_entry["rank"] = your_rank_int
                        existing_entry["missed_searches_monthly"] = missed
                        if sv_gap_f > (existing_entry.get("search_volume") or 0):
                            existing_entry["search_volume"] = int(sv_gap_f)
                        top_comp = item.get("top_competitor")
                        if top_comp:
                            existing_entry["top_competitor"] = top_comp
                        existing_entry["top_competitor_domain"] = item.get("top_competitor_domain")
                        existing_entry["top_competitor_rank"] = item.get("top_competitor_rank")
                        gap_competitors = item.get("competitors")
                        if isinstance(gap_competitors, list):
                            existing_entry["competitors"] = gap_competitors[:3]
                        existing_entry["keyword_origin"] = "gap"
                        gap_items_existing_key_rank_updated += 1
                continue

            top_keywords.append(
                {
                    "keyword": str(kw),
                    "search_volume": int(sv_gap_f),
                    "rank": your_rank_int,
                    "local_verified_rank": None,
                    "rank_source": "baseline",
                    "keyword_origin": "gap",
                    "missed_searches_monthly": missed,
                    "top_competitor": item.get("top_competitor"),
                    "top_competitor_domain": item.get("top_competitor_domain"),
                    "top_competitor_rank": item.get("top_competitor_rank"),
                    "competitors": (item.get("competitors") or [])[:3],
                }
            )
            existing_keys.add(key_lower)

        _dbg_ba84ae_log(
            hypothesisId="H4_domain_intersection_your_rank_parsing",
            location="accounts/dataforseo_utils.py:enrich_with_gap_keywords",
            message="gap keyword rank availability (target1 SERP rank)",
            data={
                "domain": domain,
                "competitor_domains_count": len(competitor_domains),
                "gap_items_total": len(gap_items),
                "gap_items_processed": gap_items_processed,
                "gap_items_with_rank": gap_items_with_rank,
                "gap_items_without_rank": gap_items_without_rank,
                "gap_items_existing_key_with_rank": gap_items_existing_key_with_rank,
                "gap_items_existing_key_rank_updated": gap_items_existing_key_rank_updated,
            },
            runId="pre-fix",
        )
    except Exception:
        logger.exception(
            "[SEO score] Error while enriching top_keywords with gap keywords for domain=%s",
            domain,
        )
    finally:
        rank_non_null_after = sum(1 for k in top_keywords if k.get("rank") is not None)
        try:
            _dbg_ba84ae_log(
                hypothesisId="H9_rank_non_null_before_after_enrichment",
                location="accounts/dataforseo_utils.py:enrich_with_gap_keywords",
                message="keyword rank non-null coverage before/after gap enrichment",
                data={
                    "domain": domain,
                    "rank_non_null_before": rank_non_null_before,
                    "rank_non_null_after": rank_non_null_after,
                },
                runId="pre-fix",
            )
        except Exception:
            pass
        logger.info(
            "[SEO score] Gap enrichment rank non-null coverage domain=%s before=%s after=%s",
            domain,
            rank_non_null_before,
            rank_non_null_after,
        )


def _best_rank_from_ranked_items(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Build keyword -> best rank map from ranked_keywords/live items.
    """
    ranked_map: Dict[str, int] = {}
    for item in items:
        keyword_data = item.get("keyword_data") or {}
        keyword_text = (keyword_data.get("keyword") or item.get("keyword") or "").strip()
        if not keyword_text:
            continue
        raw_rank = item.get("rank_absolute") or item.get("rank_group")
        try:
            rank_int = int(raw_rank) if raw_rank is not None else None
        except (TypeError, ValueError):
            rank_int = None
        if rank_int is None or rank_int <= 0:
            continue
        key = keyword_text.lower()
        prev = ranked_map.get(key)
        if prev is None or rank_int < prev:
            ranked_map[key] = rank_int
    return ranked_map


def _resolve_profile_for_rank_enrichment(user: Any) -> Optional[BusinessProfile]:
    """Resolve the main business profile for the given user, if available."""
    try:
        from .business_profile_access import resolve_main_business_profile_for_user

        return resolve_main_business_profile_for_user(user)
    except Exception:
        return None


def _get_seo_location_mode(profile: Optional[BusinessProfile]) -> str:
    """
    Resolve SEO location mode from profile.
    Supports legacy naming (`seo_location_depth`) as an equivalent input.
    """
    if not profile:
        return "organic"
    raw_mode = getattr(profile, "seo_location_mode", None)
    if raw_mode is None:
        raw_mode = getattr(profile, "seo_location_depth", None)
    mode = str(raw_mode or "organic").strip().lower()
    return "local" if mode == "local" else "organic"


def _extract_city_state_from_business_address(address: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort parser for US city/state from a free-text business address.
    Expected examples:
    - "123 Main St, Austin, TX 78701"
    - "Austin, Texas"
    """
    if not address or not address.strip():
        return None, None
    parts = [p.strip() for p in str(address).split(",") if p and p.strip()]
    if not parts:
        return None, None

    state_re = re.compile(
        r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)\b",
        re.IGNORECASE,
    )

    # Prefer common "..., City, ST ZIP" pattern.
    for idx in range(len(parts) - 1, -1, -1):
        chunk = parts[idx]
        state_match = state_re.search(chunk)
        if not state_match:
            continue
        state = state_match.group(1).upper()
        city = parts[idx - 1].strip() if idx - 1 >= 0 else None
        if city and any(ch.isalpha() for ch in city):
            return city, state
        return None, state

    # Fallback: if trailing token looks like state name text.
    trailing = parts[-1].strip()
    if trailing and any(ch.isalpha() for ch in trailing):
        city = parts[-2].strip() if len(parts) >= 2 else None
        return city if city else None, trailing
    return None, None


def resolve_local_verification_location(profile: Optional[BusinessProfile]) -> Optional[str]:
    """
    Build a DataForSEO `location_name` string for local SERP verification.
    Priority:
    - City + State -> "City,State,United States"
    - State only   -> "State,United States"
    """
    if not profile:
        return None
    city, state = _extract_city_state_from_business_address(
        str(getattr(profile, "business_address", "") or "")
    )
    if city and state:
        return f"{city},{state},United States"
    if state:
        return f"{state},United States"
    return None


def select_keywords_for_local_verification(
    top_keywords: List[Dict[str, Any]],
    *,
    min_keywords: int = 5,
    max_keywords: int = 8,
) -> List[str]:
    """Pick the highest-demand keywords for local rank verification."""
    max_n = max(min_keywords, max_keywords)
    candidates: List[Tuple[str, int]] = []
    seen: set[str] = set()
    for row in top_keywords:
        kw = str((row or {}).get("keyword") or "").strip()
        if not kw:
            continue
        kw_key = kw.lower()
        if kw_key in seen:
            continue
        seen.add(kw_key)
        try:
            vol = int((row or {}).get("search_volume") or 0)
        except (TypeError, ValueError):
            vol = 0
        candidates.append((kw, max(0, vol)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [kw for kw, _ in candidates[:max_n]]
    if len(selected) < min_keywords:
        return selected
    return selected[:max_keywords]


def run_local_keyword_rank_verification(
    *,
    domain: str,
    keywords: List[str],
    location_name: str,
    language_code: str = "en",
    business_profile: Optional[BusinessProfile] = None,
) -> Dict[str, int]:
    """
    Verify keyword ranks via SERP organic/live/advanced for a specific location.
    Returns keyword(lower) -> best rank_absolute.
    """
    if not domain or not keywords or not location_name:
        return {}
    normalized_target = normalize_domain(domain) or str(domain).strip().lower()
    out: Dict[str, int] = {}

    for kw in keywords:
        keyword = str(kw or "").strip()
        if not keyword:
            continue
        try:
            payload = [
                {
                    "keyword": keyword,
                    "location_name": location_name,
                    "language_code": language_code,
                    "device": "desktop",
                    "os": "windows",
                    "depth": 100,
                }
            ]
            data = _post(
                "/v3/serp/google/organic/live/advanced",
                payload,
                business_profile=business_profile,
            )
            if not data:
                continue
            best_rank: Optional[int] = None
            tasks = data.get("tasks") or []
            for task in tasks:
                for result in (task.get("result") or []):
                    for item in (result.get("items") or []):
                        item_type = str(item.get("type") or "").lower()
                        if item_type and item_type != "organic":
                            continue
                        item_domain = str(item.get("domain") or "").strip().lower()
                        if not item_domain:
                            item_domain = normalize_domain(item.get("url")) or ""
                        if item_domain.startswith("www."):
                            item_domain = item_domain[4:]
                        if item_domain != normalized_target:
                            continue
                        raw_rank = item.get("rank_absolute")
                        try:
                            rank_int = int(raw_rank) if raw_rank is not None else None
                        except (TypeError, ValueError):
                            rank_int = None
                        if rank_int is None or rank_int <= 0:
                            continue
                        if best_rank is None or rank_int < best_rank:
                            best_rank = rank_int
            if best_rank is not None:
                out[keyword.lower()] = int(best_rank)
        except Exception:
            logger.exception(
                "[SEO local verify] failed keyword=%s domain=%s location=%s",
                keyword,
                domain,
                location_name,
            )
            continue
    return out


def enrich_keyword_ranks_from_labs(
    *,
    domain: str,
    location_code: int,
    language_code: str,
    top_keywords: List[Dict[str, Any]],
    user=None,
    business_profile: Optional[BusinessProfile] = None,
    force_refresh_domain_intersection: bool = False,
) -> Dict[str, int]:
    """
    Enrich existing top_keywords rank values using Labs APIs:
    - ranked_keywords/live (rank_absolute/rank_group)
    - domain_intersection/live (your_rank via first_domain_serp_element)

    Keeps search_volume source unchanged; only fills/updates rank + missed searches.
    Returns enrichment stats for coverage checks/logging.
    """
    if not top_keywords:
        return {
            "total": 0,
            "null_before": 0,
            "non_null_after": 0,
            "filled_from_ranked": 0,
            "filled_from_gap": 0,
        }

    bp = business_profile if business_profile is not None else (
        _resolve_profile_for_rank_enrichment(user) if user is not None else None
    )

    total = len(top_keywords)
    null_before = sum(1 for k in top_keywords if k.get("rank") is None)

    ranked_items = fetch_ranked_keyword_items(
        domain=domain,
        location_code=location_code,
        language_code=language_code,
        user=user,
        business_profile=bp,
    )
    ranked_map = _best_rank_from_ranked_items(ranked_items)
    filled_from_ranked = 0
    filled_from_gap = 0

    for row in top_keywords:
        kw = str(row.get("keyword", "")).strip().lower()
        if not kw:
            continue
        if kw not in ranked_map:
            continue
        rank_int = ranked_map[kw]
        existing_rank = row.get("rank")
        try:
            existing_rank_int = int(existing_rank) if existing_rank is not None else None
        except (TypeError, ValueError):
            existing_rank_int = None
        if existing_rank_int is None or existing_rank_int <= 0 or rank_int < existing_rank_int:
            row["rank"] = rank_int
            try:
                sv = float(row.get("search_volume") or 0)
            except (TypeError, ValueError):
                sv = 0.0
            row["missed_searches_monthly"] = _estimate_missed_searches_monthly(sv, rank_int)
            filled_from_ranked += 1

    unresolved_keywords = []
    for row in top_keywords:
        if row.get("rank") is None:
            kw = str(row.get("keyword", "")).strip()
            if kw:
                unresolved_keywords.append(kw.lower())

    if unresolved_keywords:
        competitor_selection = get_competitors_for_domain_intersection(
            domain=domain,
            location_code=location_code,
            language_code=language_code,
            user=user,
            profile=bp,
            limit=5,
            min_competitors=int(getattr(settings, "DATAFORSEO_MIN_COMPETITORS_FOR_GAP", 2)),
        )
        competitor_domains = competitor_selection.get("filtered_competitors_used") or []
        if competitor_domains:
            gap_items = get_keyword_gap_keywords(
                domain,
                competitor_domains,
                location_code=location_code,
                language_code=language_code,
                limit=100,
                business_profile=bp,
                force_refresh=force_refresh_domain_intersection,
            )
            gap_rank_map: Dict[str, int] = {}
            for item in gap_items:
                kw = str(item.get("keyword") or "").strip().lower()
                if not kw:
                    continue
                raw_rank = item.get("your_rank")
                try:
                    rank_int = int(raw_rank) if raw_rank is not None else None
                except (TypeError, ValueError):
                    rank_int = None
                if rank_int is None or rank_int <= 0:
                    continue
                prev = gap_rank_map.get(kw)
                if prev is None or rank_int < prev:
                    gap_rank_map[kw] = rank_int

            if gap_rank_map:
                for row in top_keywords:
                    if row.get("rank") is not None:
                        continue
                    kw = str(row.get("keyword", "")).strip().lower()
                    rank_int = gap_rank_map.get(kw)
                    if rank_int is None:
                        continue
                    row["rank"] = rank_int
                    try:
                        sv = float(row.get("search_volume") or 0)
                    except (TypeError, ValueError):
                        sv = 0.0
                    row["missed_searches_monthly"] = _estimate_missed_searches_monthly(sv, rank_int)
                    filled_from_gap += 1

    # Add local rank verification metadata only when SEO location mode is local.
    # This step is intentionally non-breaking: baseline rank pipeline remains primary.
    profile = bp
    location_mode = _get_seo_location_mode(profile)
    for row in top_keywords:
        if "local_verified_rank" not in row:
            row["local_verified_rank"] = None
        if "rank_source" not in row:
            row["rank_source"] = "baseline"

    if location_mode == "local":
        location_name = resolve_local_verification_location(profile)
        if location_name:
            selected_keywords = select_keywords_for_local_verification(top_keywords, min_keywords=5, max_keywords=8)
            try:
                local_rank_map = run_local_keyword_rank_verification(
                    domain=domain,
                    keywords=selected_keywords,
                    location_name=location_name,
                    language_code=language_code,
                    business_profile=profile,
                )
            except Exception:
                local_rank_map = {}
            for row in top_keywords:
                kw_key = str((row or {}).get("keyword") or "").strip().lower()
                if not kw_key:
                    continue
                verified_rank = local_rank_map.get(kw_key)
                if verified_rank is None:
                    continue
                row["local_verified_rank"] = int(verified_rank)
                # Keep baseline `rank` intact in step 1, but expose source metadata.
                row["rank_source"] = "local_verified"
        else:
            logger.info(
                "[SEO local verify] local mode enabled but no city/state resolved; baseline retained domain=%s",
                domain,
            )

    non_null_after = sum(
        1 for k in top_keywords if isinstance(k.get("rank"), int) and (k.get("rank") or 0) > 0
    )
    return {
        "total": total,
        "null_before": null_before,
        "non_null_after": non_null_after,
        "filled_from_ranked": filled_from_ranked,
        "filled_from_gap": filled_from_gap,
    }


def enrich_with_llm_keywords(
    user,
    location_code: int,
    top_keywords: List[Dict[str, Any]],
) -> None:
    """Append validated LLM seed keywords to top_keywords in place. Preserves logging on error."""
    existing_keys = {str(k.get("keyword", "")).lower() for k in top_keywords if k.get("keyword")}
    try:
        _profile = (
            BusinessProfile.objects.filter(user=user, is_main=True).first()
            or BusinessProfile.objects.filter(user=user).first()
        )
    except Exception:
        _profile = None
    homepage_meta: Optional[str] = None
    try:
        if _profile and getattr(_profile, "website_url", None):
            pass
    except Exception:
        pass
    try:
        # Build a quick lookup of existing ranked keywords so we can
        # reuse their rank/volume for very similar brand phrases.
        existing_by_key: Dict[str, Dict[str, Any]] = {}
        for k in top_keywords:
            kw_text = str(k.get("keyword", "")).strip()
            if not kw_text:
                continue
            existing_by_key[kw_text.lower()] = k

        llm_candidates = _get_llm_keyword_candidates(_profile, homepage_meta)
        filtered_candidates: List[str] = []
        seen_llm: set[str] = set()
        for phrase in (llm_candidates or []):
            p = (phrase or "").strip()[:50]
            if not p:
                continue
            p_lower = p.lower()
            if p_lower in seen_llm:
                continue
            seen_llm.add(p_lower)
            if not _is_search_like(p, allow_one_stopword=True):
                continue
            filtered_candidates.append(p)
        if filtered_candidates:
            # Use DataForSEO volumes, but also try to align brand phrases
            # to any existing ranked keyword variant (e.g. "white pine dental"
            # vs "white pine dental care") so we don't falsely mark them
            # as not ranking.
            volumes = _get_search_volumes_for_keywords(
                filtered_candidates,
                location_code,
                business_profile=_profile,
            )
            for phrase in filtered_candidates:
                key_lower = phrase.lower()
                existing_entry = existing_by_key.get(key_lower)

                # Try to find a very similar existing keyword variant: either
                # the candidate is contained within an existing keyword or vice versa.
                # Skip exact match so we don't "match to itself" when rank is null.
                best_match: Optional[Dict[str, Any]] = None
                for ek_lower, ek in existing_by_key.items():
                    if ek_lower == key_lower:
                        continue
                    if key_lower in ek_lower or ek_lower in key_lower:
                        best_match = ek
                        break

                if key_lower in existing_keys:
                    # Keyword already exists in top_keywords.
                    # If it currently has no rank, but we found a similar variant
                    # that DOES have rank, update this entry.
                    if existing_entry is not None:
                        existing_rank = existing_entry.get("rank")
                        try:
                            existing_rank_int: Optional[int] = (
                                int(existing_rank) if existing_rank is not None else None
                            )
                        except (TypeError, ValueError):
                            existing_rank_int = None

                        if (existing_rank_int is None or existing_rank_int <= 0) and best_match is not None:
                            rank_existing = best_match.get("rank")
                            try:
                                rank_int = int(rank_existing) if rank_existing is not None else None
                            except (TypeError, ValueError):
                                rank_int = None

                            if rank_int is not None and rank_int > 0:
                                sv_existing = int(best_match.get("search_volume") or 0)
                                missed = _estimate_missed_searches_monthly(sv_existing, rank_int)
                                existing_entry["rank"] = rank_int
                                existing_entry["search_volume"] = sv_existing
                                existing_entry["missed_searches_monthly"] = missed
                                if best_match.get("top_competitor") is not None:
                                    existing_entry["top_competitor"] = best_match.get("top_competitor")
                                if best_match.get("top_competitor_domain") is not None:
                                    existing_entry["top_competitor_domain"] = best_match.get("top_competitor_domain")
                                if best_match.get("top_competitor_rank") is not None:
                                    existing_entry["top_competitor_rank"] = best_match.get("top_competitor_rank")
                                if isinstance(best_match.get("competitors"), list):
                                    existing_entry["competitors"] = (best_match.get("competitors") or [])[:3]
                    continue

                if best_match is not None:
                    # Candidate keyword does not exist yet: add it by reusing the
                    # similar existing keyword's rank and volume so the UI shows
                    # "You rank #X" for the brand phrase variant.
                    sv_existing = int(best_match.get("search_volume") or 0)
                    rank_existing = best_match.get("rank")
                    try:
                        rank_int = int(rank_existing) if rank_existing is not None else None
                    except (TypeError, ValueError):
                        rank_int = None
                    missed = _estimate_missed_searches_monthly(sv_existing, rank_int)
                    top_keywords.append(
                        {
                            "keyword": phrase,
                            "search_volume": sv_existing,
                            "rank": rank_int,
                            "local_verified_rank": None,
                            "rank_source": "baseline",
                            "keyword_origin": "profile_seed",
                            "missed_searches_monthly": missed,
                            "top_competitor": best_match.get("top_competitor"),
                            "top_competitor_domain": best_match.get("top_competitor_domain"),
                            "top_competitor_rank": best_match.get("top_competitor_rank"),
                            "competitors": (best_match.get("competitors") or [])[:3]
                            if isinstance(best_match.get("competitors"), list)
                            else [],
                        }
                    )
                    existing_keys.add(key_lower)
                    continue

                # Fallback: use DataForSEO volume for this exact phrase with no rank.
                vol = volumes.get(key_lower, 0)
                if vol <= 0:
                    continue
                missed = _estimate_missed_searches_monthly(vol, None)
                top_keywords.append(
                    {
                        "keyword": phrase,
                        "search_volume": int(vol),
                        "rank": None,
                        "local_verified_rank": None,
                        "rank_source": "baseline",
                        "keyword_origin": "profile_seed",
                        "missed_searches_monthly": missed,
                        "top_competitor": None,
                        "top_competitor_domain": None,
                        "top_competitor_rank": None,
                        "competitors": [],
                    }
                )
                existing_keys.add(key_lower)
    except Exception:
        logger.exception(
            "[SEO score] Error while adding LLM seed keywords for user_id=%s",
            getattr(user, "id", None),
        )


def compute_visibility_metrics(
    top_keywords_sorted: List[Dict[str, Any]],
    estimated_traffic: float,
    keywords_ranking: int,
    top3_positions: int,
    top10_positions: int,
    avg_difficulty: Optional[float],
    domain: str,
    location_code: int,
    language_code: str,
    seo_location_mode: str = "organic",
    business_profile: Optional[BusinessProfile] = None,
) -> Dict[str, Any]:
    """
    From combined top_keywords and ranked metrics, compute total_search_volume (from keywords),
    search_visibility_percent, missed_searches_monthly, and search_performance_score.
    """
    total_search_volume = sum(k.get("search_volume", 0) for k in top_keywords_sorted)
    estimated_search_appearances = 0.0
    for row in top_keywords_sorted:
        try:
            sv = max(float((row or {}).get("search_volume") or 0.0), 0.0)
        except (TypeError, ValueError):
            sv = 0.0
        rank_int = effective_rank_for_aggregate_metrics(row, seo_location_mode=seo_location_mode)
        if isinstance(rank_int, int) and rank_int > 0:
            estimated_search_appearances += sv * _appearance_weight_for_position(rank_int)
    if total_search_volume > 0:
        search_visibility_percent = int(round(
            max(0.0, min(100.0, (estimated_search_appearances / total_search_volume) * 100.0))
        ))
    else:
        search_visibility_percent = 0
    missed_searches_monthly = int(round(max(0.0, total_search_volume - estimated_search_appearances)))

    competitor_avg_traffic = _get_competitor_average_traffic(
        domain,
        location_code=location_code,
        language_code=language_code,
        business_profile=business_profile,
    )
    search_performance_score = compute_professional_seo_score(
        estimated_traffic=estimated_traffic,
        keywords_count=keywords_ranking,
        top3_positions=top3_positions,
        top10_positions=top10_positions,
        avg_keyword_difficulty=avg_difficulty,
        competitor_avg_traffic=competitor_avg_traffic,
    )
    return {
        "total_search_volume": total_search_volume,
        "estimated_search_appearances_monthly": int(round(max(0.0, estimated_search_appearances))),
        "search_visibility_percent": search_visibility_percent,
        "missed_searches_monthly": missed_searches_monthly,
        "search_performance_score": search_performance_score,
    }


def recompute_snapshot_metrics_from_keywords(
    *,
    top_keywords: List[Dict[str, Any]],
    domain: str,
    location_code: int,
    language_code: str = "en",
    seo_location_mode: str = "organic",
    business_profile: Optional[BusinessProfile] = None,
) -> Dict[str, int]:
    """
    Recompute aggregate snapshot metrics from enriched keyword rows.
    Keeps spotlight metrics in sync with refreshed top_keywords ranks.
    """
    rows = list(top_keywords or [])
    total_search_volume = 0.0
    estimated_traffic = 0.0
    keywords_ranking = 0
    top3_positions = 0
    top10_positions = 0
    difficulty_values: List[float] = []

    for row in rows:
        try:
            sv = max(float((row or {}).get("search_volume") or 0), 0.0)
        except (TypeError, ValueError):
            sv = 0.0
        total_search_volume += sv

        rank_int = effective_rank_for_aggregate_metrics(row, seo_location_mode=seo_location_mode)
        if isinstance(rank_int, int) and rank_int > 0:
            keywords_ranking += 1
            if rank_int <= 3:
                top3_positions += 1
            if rank_int <= 10:
                top10_positions += 1
            estimated_traffic += sv * _ctr_for_position(rank_int)

        for diff_key in ("keyword_difficulty", "difficulty"):
            diff_raw = (row or {}).get(diff_key)
            if diff_raw is None:
                continue
            try:
                dv = float(diff_raw)
            except (TypeError, ValueError):
                continue
            difficulty_values.append(max(0.0, min(100.0, dv)))
            break

    avg_difficulty = (sum(difficulty_values) / len(difficulty_values)) if difficulty_values else None
    visibility = compute_visibility_metrics(
        rows,
        estimated_traffic,
        keywords_ranking,
        top3_positions,
        top10_positions,
        avg_difficulty,
        domain,
        int(location_code),
        language_code,
        seo_location_mode=seo_location_mode,
        business_profile=business_profile,
    )
    return normalize_seo_snapshot_metrics(
        {
        "estimated_traffic": int(round(max(0.0, estimated_traffic))),
        "total_search_volume": int(visibility.get("total_search_volume") or 0),
        "estimated_search_appearances_monthly": int(visibility.get("estimated_search_appearances_monthly") or 0),
        "search_visibility_percent": int(visibility.get("search_visibility_percent") or 0),
        "missed_searches_monthly": int(visibility.get("missed_searches_monthly") or 0),
        "search_performance_score": int(visibility.get("search_performance_score") or 0),
        "keywords_ranking": int(keywords_ranking),
        "top3_positions": int(top3_positions),
        }
    )


def normalize_seo_snapshot_metrics(metrics: Dict[str, Any]) -> Dict[str, int]:
    """
    Canonical SEO snapshot invariants.
    - visibility_percent = appearances / total_search_volume * 100
    - missed_searches = total_search_volume - appearances
    """
    total_search_volume = max(0, int(metrics.get("total_search_volume") or 0))
    estimated_search_appearances_monthly = max(
        0,
        min(total_search_volume, int(metrics.get("estimated_search_appearances_monthly") or 0)),
    )
    if total_search_volume > 0:
        search_visibility_percent = int(
            round((estimated_search_appearances_monthly / total_search_volume) * 100)
        )
    else:
        search_visibility_percent = 0
    missed_searches_monthly = max(0, total_search_volume - estimated_search_appearances_monthly)
    return {
        "estimated_traffic": max(0, int(metrics.get("estimated_traffic") or 0)),
        "total_search_volume": total_search_volume,
        "estimated_search_appearances_monthly": estimated_search_appearances_monthly,
        "search_visibility_percent": max(0, min(100, search_visibility_percent)),
        "missed_searches_monthly": missed_searches_monthly,
        "search_performance_score": max(0, int(metrics.get("search_performance_score") or 0)),
        "keywords_ranking": max(0, int(metrics.get("keywords_ranking") or 0)),
        "top3_positions": max(0, int(metrics.get("top3_positions") or 0)),
    }


def seo_issue_aux_context_for_snapshot(snapshot: Any) -> Dict[str, Any]:
    """
    Build ``on_page`` / ``serp`` blobs for ``build_structured_issues`` from data tied to the snapshot.

    Uses the latest completed onboarding on-page crawl for a business profile whose website domain
    matches ``snapshot.cached_domain`` (FAQ signals). SERP feature rows are not stored on the
    snapshot today; callers may still pass ``serp`` via ``seo_data`` when available.
    """
    from .models import BusinessProfile, OnboardingOnPageCrawl

    out_on: Dict[str, Any] = {}
    out_serp: List[Dict[str, Any]] = []

    domain = (getattr(snapshot, "cached_domain", "") or "").strip().lower()
    if not domain:
        return {"on_page": out_on, "serp": out_serp}

    profile = getattr(snapshot, "business_profile", None)
    if profile is not None:
        profile = (
            BusinessProfile.objects.filter(pk=profile.pk).only("id", "website_url", "is_main").first()
            or profile
        )
    uid = getattr(snapshot, "user_id", None)
    if profile is None and uid:
        for p in BusinessProfile.objects.filter(user_id=uid).only("id", "website_url", "is_main"):
            if normalize_domain(getattr(p, "website_url", "") or "") == domain:
                profile = p
                break
        if profile is None:
            profile = (
                BusinessProfile.objects.filter(user_id=uid, is_main=True).only("id", "website_url").first()
                or BusinessProfile.objects.filter(user_id=uid).only("id", "website_url").first()
            )
    if profile is None:
        return {"on_page": out_on, "serp": out_serp}

    crawl = (
        OnboardingOnPageCrawl.objects.filter(
            business_profile=profile,
            domain__iexact=domain,
            status=OnboardingOnPageCrawl.STATUS_COMPLETED,
        )
        .order_by("-id")
        .first()
    )
    if not crawl:
        return {"on_page": out_on, "serp": out_serp}

    pages = list(getattr(crawl, "pages", None) or [])[:25]
    if not pages:
        return {"on_page": out_on, "serp": out_serp}

    faq = compute_faq_readiness_for_pages(pages)
    out_on["faq_blocks_found"] = int(faq.get("faq_blocks_found") or 0)
    out_on["faq_schema_present"] = bool(faq.get("faq_schema_present"))
    out_on["faq_content_present"] = out_on["faq_blocks_found"] > 0
    first_url = ""
    for pg in pages:
        if isinstance(pg, dict):
            u = str(pg.get("url") or pg.get("page_url") or "").strip()
            if u:
                first_url = u
                break
    if first_url:
        out_on["user_url"] = first_url

    return {"on_page": out_on, "serp": out_serp}


def save_seo_snapshot(
    business_profile: BusinessProfile,
    period_start,
    domain: str,
    now_utc: datetime,
    organic_visitors: int,
    keywords_ranking: int,
    top3_positions: int,
    top_keywords_sorted: List[Dict[str, Any]],
    total_search_volume: int,
    estimated_search_appearances_monthly: int,
    missed_searches_monthly: int,
    search_visibility_percent: int,
    search_performance_score: int,
    cached_location_mode: str = "organic",
    cached_location_code: int = 0,
    cached_location_label: str = "",
    local_verification_applied: bool = False,
    local_verified_keyword_count: int = 0,
):
    """
    Persist keyword list and search metrics to SEOOverviewSnapshot. Logs on failure.
    Returns the snapshot instance on success so callers can enqueue async tasks by snapshot.id; returns None on failure.
    """
    user = getattr(business_profile, "user", None)
    if user is None:
        logger.warning(
            "[SEO score] save_seo_snapshot: business_profile id=%s has no user",
            getattr(business_profile, "id", None),
        )
        return None
    try:
        snapshot, _ = SEOOverviewSnapshot.objects.get_or_create(
            business_profile=business_profile,
            period_start=period_start,
            cached_location_mode=str(cached_location_mode or "organic"),
            cached_location_code=int(cached_location_code or 0),
            defaults={"user": user},
        )
        if getattr(snapshot, "user_id", None) != getattr(user, "id", None):
            snapshot.user = user
        snapshot.organic_visitors = int(organic_visitors or 0)
        snapshot.keywords_ranking = int(keywords_ranking or 0)
        snapshot.top3_positions = int(top3_positions or 0)
        snapshot.refreshed_at = now_utc
        snapshot.cached_domain = domain
        snapshot.cached_location_mode = str(cached_location_mode or "organic")
        snapshot.cached_location_code = int(cached_location_code or 0)
        snapshot.cached_location_label = str(cached_location_label or "")
        snapshot.local_verification_applied = bool(local_verification_applied)
        snapshot.local_verified_keyword_count = int(local_verified_keyword_count or 0)
        snapshot.top_keywords = top_keywords_sorted
        snapshot.total_search_volume = int(total_search_volume or 0)
        snapshot.estimated_search_appearances_monthly = int(estimated_search_appearances_monthly or 0)
        snapshot.missed_searches_monthly = int(missed_searches_monthly or 0)
        snapshot.search_visibility_percent = int(search_visibility_percent or 0)
        snapshot.search_performance_score = int(search_performance_score or 0)
        snapshot.save()
        return snapshot
    except Exception:
        logger.exception(
            "[SEO score] Failed to save keyword cache for profile_id=%s",
            getattr(business_profile, "id", None),
        )
        return None


def generate_or_get_next_steps(
    business_profile: Optional[BusinessProfile],
    period_start,
    result: Dict[str, Any],
    now_utc: datetime,
    location_mode: str = "organic",
    location_code: int = 0,
) -> None:
    """Fill result['seo_next_steps'] from cache (if fresh) or OpenAI; save to snapshot. Mutates result."""
    if not business_profile:
        result["seo_next_steps"] = []
        return
    user = business_profile.user
    steps_ttl = timedelta(days=7)
    snapshot_for_steps = SEOOverviewSnapshot.objects.filter(
        business_profile=business_profile,
        period_start=period_start,
        cached_location_mode=str(location_mode or "organic"),
        cached_location_code=int(location_code or 0),
    ).first()
    cached_steps: List[Any] = []
    if snapshot_for_steps and getattr(snapshot_for_steps, "seo_next_steps_refreshed_at", None):
        refreshed_at = snapshot_for_steps.seo_next_steps_refreshed_at
        if refreshed_at and (now_utc - refreshed_at) <= steps_ttl:
            cached_steps = getattr(snapshot_for_steps, "seo_next_steps", None) or []
    if cached_steps:
        result["seo_next_steps"] = cached_steps
    else:
        try:
            from .openai_utils import generate_seo_next_steps

            snap, _ = SEOOverviewSnapshot.objects.get_or_create(
                business_profile=business_profile,
                period_start=period_start,
                cached_location_mode=str(location_mode or "organic"),
                cached_location_code=int(location_code or 0),
                defaults={"user": user},
            )
            if getattr(snap, "user_id", None) != getattr(user, "id", None):
                snap.user = user
                snap.save(update_fields=["user"])
            steps = generate_seo_next_steps(result, snapshot=snap)
            result["seo_next_steps"] = steps if steps else []
            snap.seo_next_steps = result.get("seo_next_steps") or []
            snap.seo_next_steps_refreshed_at = now_utc
            snap.save(update_fields=["seo_next_steps", "seo_next_steps_refreshed_at"])
        except Exception:
            logger.exception(
                "[SEO score] Failed to generate seo_next_steps for profile_id=%s",
                getattr(business_profile, "id", None),
            )
            result["seo_next_steps"] = []


def build_seo_response(
    *,
    search_performance_score: int,
    organic_visitors: int,
    total_search_volume: int,
    estimated_search_appearances_monthly: int,
    keywords_ranking: int,
    top3_positions: int,
    search_visibility_percent: int,
    missed_searches_monthly: int,
    top_keywords: List[Dict[str, Any]],
    seo_next_steps: Optional[List[Any]] = None,
    keyword_action_suggestions: Optional[List[Any]] = None,
    seo_structured_issues: Optional[List[Any]] = None,
    enrichment_status: str = "complete",
    seo_metrics_location_mode: str = "organic",
    seo_location_label: str = "",
    local_verified_keyword_count: int = 0,
    local_verification_affects_visibility: bool = False,
) -> Dict[str, Any]:
    """Build the SEO overview API payload.

    On-page/technical audit is intentionally disabled to avoid DataForSEO On-Page costs.
    """
    normalized = normalize_seo_snapshot_metrics(
        {
            "estimated_traffic": organic_visitors,
            "total_search_volume": total_search_volume,
            "estimated_search_appearances_monthly": estimated_search_appearances_monthly,
            "search_visibility_percent": search_visibility_percent,
            "missed_searches_monthly": missed_searches_monthly,
            "search_performance_score": search_performance_score,
            "keywords_ranking": keywords_ranking,
            "top3_positions": top3_positions,
        }
    )
    overall = int(normalized["search_performance_score"] or 0)
    meta = build_seo_snapshot_api_metadata(
        seo_location_label=str(seo_location_label or ""),
        local_verified_keyword_count=int(local_verified_keyword_count or 0),
        local_verification_affects_visibility=bool(local_verification_affects_visibility),
    )
    return {
        "seo_score": overall,
        "search_performance_score": int(normalized["search_performance_score"] or 0),
        "organic_visitors": int(normalized["estimated_traffic"] or 0),
        "total_search_volume": int(normalized["total_search_volume"] or 0),
        "estimated_search_appearances_monthly": int(normalized["estimated_search_appearances_monthly"] or 0),
        "keywords_ranking": int(normalized["keywords_ranking"] or 0),
        "top3_positions": int(normalized["top3_positions"] or 0),
        "search_visibility_percent": int(normalized["search_visibility_percent"] or 0),
        "missed_searches_monthly": int(normalized["missed_searches_monthly"] or 0),
        "top_keywords": top_keywords or [],
        "seo_next_steps": seo_next_steps if seo_next_steps is not None else [],
        "keyword_action_suggestions": keyword_action_suggestions if keyword_action_suggestions is not None else [],
        "seo_structured_issues": []
        if seo_structured_issues is None
        else list(seo_structured_issues),
        "enrichment_status": enrichment_status,
        "seo_metrics_location_mode": str(seo_metrics_location_mode or "organic"),
        **meta,
    }


def _build_empty_seo_response(
    user,
    site_url: str,
    *,
    seo_metrics_location_mode: str = "organic",
    seo_location_label: str = "",
) -> Dict[str, Any]:
    """Build zeroed SEO response when no ranked data."""
    fallback_score = compute_professional_seo_score(
        estimated_traffic=0.0, keywords_count=0, top3_positions=0, top10_positions=0,
        avg_keyword_difficulty=None, competitor_avg_traffic=0.0,
    )
    return build_seo_response(
        search_performance_score=fallback_score,
        organic_visitors=0, total_search_volume=0, estimated_search_appearances_monthly=0, keywords_ranking=0, top3_positions=0,
        search_visibility_percent=0, missed_searches_monthly=0, top_keywords=[], seo_next_steps=[],
        keyword_action_suggestions=[],
        seo_structured_issues=[],
        enrichment_status="complete",
        seo_metrics_location_mode=str(seo_metrics_location_mode or "organic"),
        seo_location_label=str(seo_location_label or ""),
        local_verified_keyword_count=0,
        local_verification_affects_visibility=False,
    )


def _log_seo_skip_and_return_none(message: str, user, **data_extra: Any) -> None:
    """Preserve agent debug log when skipping SEO score (no site_url or no domain)."""
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "sessionId": "098bfd", "runId": "pre-fix", "hypothesisId": "H2",
                "location": "accounts/dataforseo_utils.py:get_or_refresh_seo_score_for_user",
                "message": message,
                "data": {"user_id": getattr(user, "id", None), **data_extra},
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            }) + "\n")
    except Exception:
        pass


def get_or_refresh_seo_score_for_user(
    user,
    *,
    site_url: str | None,
    force_refresh: bool = False,
    business_profile: Optional[BusinessProfile] = None,
) -> Dict[str, Any] | None:
    """
    Fetch a cached, professional-grade SEO score + core metrics for the given user,
    refreshing from DataForSEO at most once per hour (same cadence as the dashboard)
    and combining:
    - Search Performance (ranked_keywords + competitors)
    - On-Page SEO (metadata, headings, alt text)
    - Technical SEO (links, canonical, robots, sitemap)

    Returns a dict with at least:
    - seo_score (Overall SEO Score 0–100)
    - search_performance_score
    - onpage_seo_score
    - technical_seo_score
    - organic_visitors (estimated traffic)
    - keywords_ranking
    - top3_positions
    or None if no website is configured / domain cannot be derived.

    When ``business_profile`` is provided, snapshot cache reads/writes are scoped to that
    profile so multi-company accounts do not clobber each other's monthly rows.
    """
    today = datetime.now(timezone.utc).date()
    start_current = today.replace(day=1)
    now_utc = datetime.now(timezone.utc)
    # Cache cadence for keyword ranks + competitor evidence:
    # Requirement: keep keyword data stable for 7 days unless refresh is explicitly requested.
    cache_ttl = timedelta(days=7)
    if not site_url:
        _log_seo_skip_and_return_none("No site_url; skipping SEO score calculation", user)
        return None
    domain = normalize_domain(site_url)
    if not domain:
        _log_seo_skip_and_return_none("Could not normalise domain from site_url", user, site_url=site_url)
        return None

    default_location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    language_code = getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en")
    profile = business_profile or _resolve_profile_for_rank_enrichment(user)
    if profile is None:
        _log_seo_skip_and_return_none("No business profile for SEO snapshot scope", user)
        return None
    api_user = getattr(profile, "user", None) or user
    snapshot_context = resolve_snapshot_location_context(
        profile=profile,
        default_location_code=default_location_code,
    )
    location_mode = str(snapshot_context.get("mode") or "organic")
    snapshot_location_code = int(snapshot_context.get("code") or 0)
    location_label = str(snapshot_context.get("label") or "")
    ranking_location_code = int(snapshot_location_code or default_location_code)
    if location_mode != "local":
        ranking_location_code = int(default_location_code)

    snapshot = get_cached_seo_snapshot(
        profile,
        domain,
        start_current,
        cache_ttl,
        now_utc,
        location_mode=location_mode,
        location_code=snapshot_location_code,
    )
    # #region agent log
    _debug.log(
        "dataforseo_utils.py:get_or_refresh_seo_score_for_user:cache_probe",
        "SEO snapshot cache probe",
        {
            "user_id": getattr(user, "id", None),
            "domain": domain,
            "force_refresh": bool(force_refresh),
            "snapshot_found": snapshot is not None,
            "cache_ttl_days": int(cache_ttl.days),
        },
        "S4",
    )
    # #endregion
    if snapshot and not force_refresh:
        try:
            cached_keywords = getattr(snapshot, "top_keywords", None) or []
            cached_rank_positive = sum(
                1 for k in cached_keywords if isinstance(k.get("rank"), int) and (k.get("rank") or 0) > 0
            )
            cached_rank_none = sum(1 for k in cached_keywords if k.get("rank") is None)
            refreshed_at = getattr(snapshot, "refreshed_at", None)
            age_sec = None
            if refreshed_at:
                try:
                    age_sec = (now_utc - refreshed_at).total_seconds()
                except Exception:
                    age_sec = None
            _dbg_ba84ae_log(
                hypothesisId="H2_cache_hit_serves_null_ranks",
                location="accounts/dataforseo_utils.py:get_or_refresh_seo_score_for_user",
                message="cache hit: serving snapshot (rank evidence)",
                data={
                    "domain": domain,
                    "top_keywords_count": len(cached_keywords),
                    "rank_positive_count": cached_rank_positive,
                    "rank_none_count": cached_rank_none,
                    "age_sec": age_sec,
                },
                runId="pre-fix",
            )
        except Exception:
            pass
        # Self-heal stale/inconsistent snapshot metrics from current keyword rows.
        affects_visibility = False
        try:
            cached_keywords = list(getattr(snapshot, "top_keywords", None) or [])
            repaired_metrics = recompute_snapshot_metrics_from_keywords(
                top_keywords=cached_keywords,
                domain=domain,
                location_code=int(ranking_location_code),
                language_code=language_code,
                seo_location_mode=location_mode,
                business_profile=profile,
            )
            if location_mode == "local":
                baseline_cmp = recompute_snapshot_metrics_from_keywords(
                    top_keywords=cached_keywords,
                    domain=domain,
                    location_code=int(ranking_location_code),
                    language_code=language_code,
                    seo_location_mode="organic",
                    business_profile=profile,
                )
                affects_visibility = local_verification_affects_visibility(
                    seo_location_mode=location_mode,
                    baseline_metrics=baseline_cmp,
                    local_mode_metrics=repaired_metrics,
                )
            stored_visibility = int(getattr(snapshot, "search_visibility_percent", 0) or 0)
            repaired_visibility = int(repaired_metrics.get("search_visibility_percent") or 0)
            stored_appearances = int(getattr(snapshot, "estimated_search_appearances_monthly", 0) or 0)
            repaired_appearances = int(repaired_metrics.get("estimated_search_appearances_monthly") or 0)
            stored_traffic = int(snapshot.organic_visitors or 0)
            repaired_traffic = int(repaired_metrics.get("estimated_traffic") or 0)
            stored_kw = int(snapshot.keywords_ranking or 0)
            repaired_kw = int(repaired_metrics.get("keywords_ranking") or 0)
            stored_top3 = int(snapshot.top3_positions or 0)
            repaired_top3 = int(repaired_metrics.get("top3_positions") or 0)
            stored_score = int(snapshot.search_performance_score or 0)
            repaired_score = int(repaired_metrics.get("search_performance_score") or 0)
            stored_missed = int(snapshot.missed_searches_monthly or 0)
            repaired_missed = int(repaired_metrics.get("missed_searches_monthly") or 0)
            if (
                stored_visibility != repaired_visibility
                or stored_appearances != repaired_appearances
                or stored_traffic != repaired_traffic
                or stored_kw != repaired_kw
                or stored_top3 != repaired_top3
                or stored_score != repaired_score
                or stored_missed != repaired_missed
            ):
                snapshot.organic_visitors = int(repaired_metrics["estimated_traffic"])
                snapshot.total_search_volume = int(repaired_metrics["total_search_volume"])
                snapshot.estimated_search_appearances_monthly = int(repaired_metrics["estimated_search_appearances_monthly"])
                snapshot.search_visibility_percent = int(repaired_metrics["search_visibility_percent"])
                snapshot.missed_searches_monthly = int(repaired_metrics["missed_searches_monthly"])
                snapshot.search_performance_score = int(repaired_metrics["search_performance_score"])
                snapshot.keywords_ranking = int(repaired_metrics["keywords_ranking"])
                snapshot.top3_positions = int(repaired_metrics["top3_positions"])
                snapshot.save(
                    update_fields=[
                        "organic_visitors",
                        "total_search_volume",
                        "estimated_search_appearances_monthly",
                        "search_visibility_percent",
                        "missed_searches_monthly",
                        "search_performance_score",
                        "keywords_ranking",
                        "top3_positions",
                    ]
                )
        except Exception:
            logger.exception(
                "[SEO score] cache-hit metric repair failed user_id=%s domain=%s",
                getattr(user, "id", None),
                domain,
            )

        enrichment_status = "complete" if (
            getattr(snapshot, "keywords_enriched_at", None)
            and getattr(snapshot, "seo_next_steps_refreshed_at", None)
            and getattr(snapshot, "keyword_action_suggestions_refreshed_at", None)
        ) else "pending"
        snap_mode = str(getattr(snapshot, "cached_location_mode", None) or location_mode or "organic")
        snap_label = str(getattr(snapshot, "cached_location_label", None) or location_label or "")
        verified_n = int(getattr(snapshot, "local_verified_keyword_count", 0) or 0)
        return build_seo_response(
            search_performance_score=int(snapshot.search_performance_score or 0),
            organic_visitors=int(snapshot.organic_visitors or 0),
            total_search_volume=int(getattr(snapshot, "total_search_volume", 0) or 0),
            estimated_search_appearances_monthly=int(getattr(snapshot, "estimated_search_appearances_monthly", 0) or 0),
            keywords_ranking=int(snapshot.keywords_ranking or 0),
            top3_positions=int(snapshot.top3_positions or 0),
            search_visibility_percent=int(getattr(snapshot, "search_visibility_percent", 0) or 0),
            missed_searches_monthly=int(getattr(snapshot, "missed_searches_monthly", 0) or 0),
            top_keywords=getattr(snapshot, "top_keywords", None) or [],
            seo_next_steps=getattr(snapshot, "seo_next_steps", None) or [],
            keyword_action_suggestions=getattr(snapshot, "keyword_action_suggestions", None) or [],
            seo_structured_issues=getattr(snapshot, "seo_structured_issues", None) or [],
            enrichment_status=enrichment_status,
            seo_metrics_location_mode=snap_mode,
            seo_location_label=snap_label,
            local_verified_keyword_count=verified_n,
            local_verification_affects_visibility=affects_visibility,
        )

    _dbg_ba84ae_log(
        hypothesisId="H2_cache_miss_about_to_call_ranked_keywords",
        location="accounts/dataforseo_utils.py:get_or_refresh_seo_score_for_user",
        message="cache miss: will call ranked_keywords/live",
        data={"domain": domain},
        runId="pre-fix",
    )

    from accounts.third_party_usage import usage_profile_context

    def _live_seo_refresh_branch() -> Dict[str, Any] | None:
        ranked_kw_limit = int(getattr(settings, "DATAFORSEO_RANKED_KEYWORDS_LIMIT", 100))
        items = fetch_ranked_keyword_items(
            domain,
            ranking_location_code,
            language_code,
            api_user,
            limit=ranked_kw_limit,
            business_profile=profile,
        )
        if not items:
            # #region agent log
            _debug.log(
                "dataforseo_utils.py:get_or_refresh_seo_score_for_user:empty_items_zero_fallback",
                "No ranked keyword items; returning zero SEO response",
                {"user_id": getattr(user, "id", None), "domain": domain},
                "S5",
            )
            # #endregion
            return _build_empty_seo_response(
                user,
                site_url,
                seo_metrics_location_mode=location_mode,
                seo_location_label=location_label,
            )

        metrics = compute_ranked_metrics(items)
        # Ranked-only keywords for sync path: no gap/LLM enrichment here (done in background).
        max_kw = int(getattr(settings, "SEO_TOP_KEYWORDS_MAX_PERSISTED", 200))
        top_keywords_ranked = sort_top_keywords_for_display(metrics["top_keywords"], max_rows=max_kw)

        agg = normalize_seo_snapshot_metrics(
            recompute_snapshot_metrics_from_keywords(
                top_keywords=top_keywords_ranked,
                domain=domain,
                location_code=int(ranking_location_code),
                language_code=language_code,
                seo_location_mode=location_mode,
                business_profile=profile,
            )
        )
        affects_miss = False
        if location_mode == "local":
            baseline_agg = normalize_seo_snapshot_metrics(
                recompute_snapshot_metrics_from_keywords(
                    top_keywords=top_keywords_ranked,
                    domain=domain,
                    location_code=int(ranking_location_code),
                    language_code=language_code,
                    seo_location_mode="organic",
                    business_profile=profile,
                )
            )
            affects_miss = local_verification_affects_visibility(
                seo_location_mode=location_mode,
                baseline_metrics=baseline_agg,
                local_mode_metrics=agg,
            )

        snapshot = save_seo_snapshot(
            profile,
            start_current,
            domain,
            now_utc,
            organic_visitors=int(agg["estimated_traffic"] or 0),
            keywords_ranking=int(agg["keywords_ranking"] or 0),
            top3_positions=int(agg["top3_positions"] or 0),
            top_keywords_sorted=top_keywords_ranked,
            total_search_volume=int(agg["total_search_volume"] or 0),
            estimated_search_appearances_monthly=int(agg["estimated_search_appearances_monthly"] or 0),
            missed_searches_monthly=int(agg["missed_searches_monthly"] or 0),
            search_visibility_percent=int(agg["search_visibility_percent"] or 0),
            search_performance_score=int(agg["search_performance_score"] or 0),
            cached_location_mode=location_mode,
            cached_location_code=snapshot_location_code,
            cached_location_label=location_label,
            local_verification_applied=any(
                str((row or {}).get("rank_source") or "baseline") == "local_verified"
                for row in (top_keywords_ranked or [])
            ),
            local_verified_keyword_count=sum(
                1
                for row in (top_keywords_ranked or [])
                if (row or {}).get("local_verified_rank") is not None
            ),
        )

        if snapshot:
            try:
                from .tasks import (
                    enrich_snapshot_keywords_task,
                    generate_snapshot_next_steps_task,
                    generate_keyword_action_suggestions_task,
                )
                enrich_snapshot_keywords_task.delay(snapshot.id)
                generate_snapshot_next_steps_task.delay(snapshot.id)
                generate_keyword_action_suggestions_task.delay(snapshot.id)
            except Exception as e:
                logger.warning("[SEO score] Could not enqueue enrichment tasks: %s", e)

        return build_seo_response(
            search_performance_score=int(agg["search_performance_score"] or 0),
            organic_visitors=int(agg["estimated_traffic"] or 0),
            total_search_volume=int(agg["total_search_volume"] or 0),
            estimated_search_appearances_monthly=int(agg.get("estimated_search_appearances_monthly") or 0),
            keywords_ranking=int(agg["keywords_ranking"] or 0),
            top3_positions=int(agg["top3_positions"] or 0),
            search_visibility_percent=int(agg["search_visibility_percent"] or 0),
            missed_searches_monthly=int(agg["missed_searches_monthly"] or 0),
            top_keywords=top_keywords_ranked,
            seo_next_steps=[],
            keyword_action_suggestions=[],
            seo_structured_issues=[],
            enrichment_status="pending",
            seo_metrics_location_mode=str(location_mode or "organic"),
            seo_location_label=str(location_label or ""),
            local_verified_keyword_count=sum(
                1 for row in (top_keywords_ranked or []) if (row or {}).get("local_verified_rank") is not None
            ),
            local_verification_affects_visibility=affects_miss,
        )

    with usage_profile_context(profile):
        return _live_seo_refresh_branch()

