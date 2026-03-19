from __future__ import annotations

"""
Helpers for calling DataForSEO Labs APIs for the SEO Agent.

Primary endpoints used:
- POST /v3/dataforseo_labs/google/ranked_keywords/live      → current visibility
- POST /v3/dataforseo_labs/google/domain_intersection/live  → keyword gap vs competitors
"""

from typing import Any, Dict, List, Optional

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
import re

import json
import math
import requests
from pathlib import Path
from django.conf import settings

from .models import BusinessProfile, SEOOverviewSnapshot

logger = logging.getLogger(__name__)

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

    # Always fetch raw competitors so we can output debug evidence.
    # Use a larger initial limit to allow for filtering down to "limit".
    raw_competitor_candidates = _get_competitor_domains(
        domain,
        location_code=location_code,
        language_code=language_code,
        limit=min(max(limit * 2, limit + 2), 10),
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
    data = _post("/v3/keywords_data/google/search_volume/live", payload)
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


def _post(path: str, payload: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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

    if resp.status_code != 200:
        # DataForSEO returns 200 with status_code inside body for logical errors;
        # non-200 here is a transport-level issue.
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

    try:
        data = resp.json()
    except ValueError:  # pragma: no cover - unexpected non-JSON
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
) -> float:
    """
    Call competitors_domain/live to estimate the average organic traffic/visibility
    of the top competitors for the given domain.
    """
    payload = [
        {
            "target": target_domain,
            "location_code": int(location_code),
            "language_code": language_code,
        },
    ]

    data = _post("/v3/dataforseo_labs/google/competitors_domain/live", payload)
    if not data:
        return 0.0

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            return 0.0
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            return 0.0
        result = results[0]
        items = result.get("items") or []
        if not items:
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
            return 0.0

        return sum(competitor_scores) / len(competitor_scores)
    except Exception:
        logger.exception(
            "[DataForSEO] competitors_domain parsing failed for target_domain=%s",
            target_domain,
        )
        return 0.0


def _get_competitor_domains(
    target_domain: str,
    *,
    location_code: int,
    language_code: str,
    limit: int = 5,
) -> List[str]:
    """
    Call competitors_domain/live and return 3–5 main competitor domain names
    (excluding the target). Used to feed get_keyword_gap_keywords for
    high-intent opportunity keywords.
    """
    payload = [
        {
            "target": target_domain,
            "location_code": int(location_code),
            "language_code": language_code,
            "limit": min(limit + 5, 100),
        },
    ]
    data = _post("/v3/dataforseo_labs/google/competitors_domain/live", payload)
    if not data:
        return []

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            return []
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            return []
        result = results[0]
        items = result.get("items") or []
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
        return domains[:limit]
    except Exception:
        logger.exception(
            "[DataForSEO] competitors_domain domain list parsing failed for target_domain=%s",
            target_domain,
        )
        return []


def get_ranked_keywords_visibility(
    target_domain: str,
    *,
    location_code: int,
    language_code: str = "en",
    limit: int = 100,
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

    data = _post("/v3/dataforseo_labs/google/ranked_keywords/live", payload)
    # #region agent log
    _debug.log("dataforseo_utils.py:get_ranked_keywords_visibility:after_post", "API response", {"has_data": data is not None}, "H4")
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


def get_keyword_gap_keywords(
    target_domain: str,
    competitor_domains: List[str],
    *,
    location_code: int,
    language_code: str = "en",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Call domain_intersection/live to compute keyword gaps vs competitors.

    For now we return a simplified list of items suitable for the frontend SEO keywords table:
    - keyword
    - search_volume
    """
    cleaned_competitors = [c.strip().lower() for c in competitor_domains if c.strip()]
    if not cleaned_competitors:
        logger.info(
            "[DataForSEO] domain_intersection skipped for %s: no competitors configured",
            target_domain,
        )
        return []

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

        data = _post("/v3/dataforseo_labs/google/domain_intersection/live", payload)
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


def get_cached_seo_snapshot(
    user,
    domain: str,
    period_start,
    cache_ttl: timedelta,
    now_utc: datetime,
) -> Optional[Any]:
    """
    Return a fresh SEOOverviewSnapshot for this user/period if cache is valid
    (refreshed within cache_ttl and cached_domain matches). Otherwise return None.
    """
    try:
        snapshot = SEOOverviewSnapshot.objects.filter(
            user=user,
            period_start=period_start,
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
        logger.exception("[SEO score] Error reading keyword cache for user_id=%s", getattr(user, "id", None))
        return None


def fetch_ranked_keyword_items(
    domain: str,
    location_code: int,
    language_code: str,
    user=None,
) -> List[Dict[str, Any]]:
    """
    Call DataForSEO ranked_keywords/live and return parsed items list.
    Returns [] on any failure. Logs request and first-item preview.
    """
    payload = [
        {
            "target": domain,
            "location_code": int(location_code),
            "language_code": language_code,
            "limit": 100,
        },
    ]
    logger.info(
        "[SEO score] ranked_keywords request user_id=%s domain=%s payload=%s",
        getattr(user, "id", None),
        domain,
        payload,
    )
    ranked_data = _post("/v3/dataforseo_labs/google/ranked_keywords/live", payload)
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
            raise ValueError("no tasks")
        task = tasks[0]
        results = task.get("result") or []
        if not results:
            raise ValueError("no results")
        result = results[0]
        items = result.get("items") or []
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
                        gap_items_existing_key_rank_updated += 1
                continue

            top_keywords.append(
                {
                    "keyword": str(kw),
                    "search_volume": int(sv_gap_f),
                    "rank": your_rank_int,
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


def enrich_keyword_ranks_from_labs(
    *,
    domain: str,
    location_code: int,
    language_code: str,
    top_keywords: List[Dict[str, Any]],
    user=None,
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

    total = len(top_keywords)
    null_before = sum(1 for k in top_keywords if k.get("rank") is None)

    ranked_items = fetch_ranked_keyword_items(
        domain=domain,
        location_code=location_code,
        language_code=language_code,
        user=user,
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
            profile=None,
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
            volumes = _get_search_volumes_for_keywords(filtered_candidates, location_code)
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
) -> Dict[str, Any]:
    """
    From combined top_keywords and ranked metrics, compute total_search_volume (from keywords),
    search_visibility_percent, missed_searches_monthly, and search_performance_score.
    """
    total_search_volume = sum(k.get("search_volume", 0) for k in top_keywords_sorted)
    if total_search_volume > 0:
        search_visibility_percent = int(round(
            max(0.0, min(100.0, (estimated_traffic / total_search_volume) * 100.0))
        ))
    else:
        search_visibility_percent = 0
    missed_searches_monthly = int(round(max(0.0, total_search_volume - estimated_traffic)))

    competitor_avg_traffic = _get_competitor_average_traffic(
        domain,
        location_code=location_code,
        language_code=language_code,
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
        "search_visibility_percent": search_visibility_percent,
        "missed_searches_monthly": missed_searches_monthly,
        "search_performance_score": search_performance_score,
    }


def save_seo_snapshot(
    user,
    period_start,
    domain: str,
    now_utc: datetime,
    organic_visitors: int,
    keywords_ranking: int,
    top3_positions: int,
    top_keywords_sorted: List[Dict[str, Any]],
    total_search_volume: int,
    missed_searches_monthly: int,
    search_visibility_percent: int,
    search_performance_score: int,
):
    """
    Persist keyword list and search metrics to SEOOverviewSnapshot. Logs on failure.
    Returns the snapshot instance on success so callers can enqueue async tasks by snapshot.id; returns None on failure.
    """
    try:
        snapshot, _ = SEOOverviewSnapshot.objects.get_or_create(
            user=user,
            period_start=period_start,
        )
        snapshot.organic_visitors = int(organic_visitors or 0)
        snapshot.keywords_ranking = int(keywords_ranking or 0)
        snapshot.top3_positions = int(top3_positions or 0)
        snapshot.refreshed_at = now_utc
        snapshot.cached_domain = domain
        snapshot.top_keywords = top_keywords_sorted
        snapshot.total_search_volume = int(total_search_volume or 0)
        snapshot.missed_searches_monthly = int(missed_searches_monthly or 0)
        snapshot.search_visibility_percent = int(search_visibility_percent or 0)
        snapshot.search_performance_score = int(search_performance_score or 0)
        snapshot.save()
        return snapshot
    except Exception:
        logger.exception("[SEO score] Failed to save keyword cache for user_id=%s", getattr(user, "id", None))
        return None


def generate_or_get_next_steps(
    user,
    period_start,
    result: Dict[str, Any],
    now_utc: datetime,
) -> None:
    """Fill result['seo_next_steps'] from cache (if fresh) or OpenAI; save to snapshot. Mutates result."""
    steps_ttl = timedelta(days=7)
    snapshot_for_steps = SEOOverviewSnapshot.objects.filter(
        user=user,
        period_start=period_start,
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
            steps = generate_seo_next_steps(result)
            result["seo_next_steps"] = steps if steps else []
        except Exception:
            logger.exception("[SEO score] Failed to generate seo_next_steps for user_id=%s", getattr(user, "id", None))
            result["seo_next_steps"] = []
        try:
            snap, _ = SEOOverviewSnapshot.objects.get_or_create(
                user=user,
                period_start=period_start,
            )
            snap.seo_next_steps = result.get("seo_next_steps") or []
            snap.seo_next_steps_refreshed_at = now_utc
            snap.save(update_fields=["seo_next_steps", "seo_next_steps_refreshed_at"])
        except Exception:
            logger.exception("[SEO score] Failed to save seo_next_steps to snapshot for user_id=%s", getattr(user, "id", None))


def build_seo_response(
    *,
    search_performance_score: int,
    organic_visitors: int,
    total_search_volume: int,
    keywords_ranking: int,
    top3_positions: int,
    search_visibility_percent: int,
    missed_searches_monthly: int,
    top_keywords: List[Dict[str, Any]],
    seo_next_steps: Optional[List[Any]] = None,
    enrichment_status: str = "complete",
) -> Dict[str, Any]:
    """Build the SEO overview API payload.

    On-page/technical audit is intentionally disabled to avoid DataForSEO On-Page costs.
    """
    overall = int(search_performance_score or 0)
    return {
        "seo_score": overall,
        "search_performance_score": int(search_performance_score or 0),
        "organic_visitors": int(organic_visitors or 0),
        "total_search_volume": int(total_search_volume or 0),
        "keywords_ranking": int(keywords_ranking or 0),
        "top3_positions": int(top3_positions or 0),
        "search_visibility_percent": int(search_visibility_percent or 0),
        "missed_searches_monthly": int(missed_searches_monthly or 0),
        "top_keywords": top_keywords or [],
        "seo_next_steps": seo_next_steps if seo_next_steps is not None else [],
        "enrichment_status": enrichment_status,
    }


def _build_empty_seo_response(user, site_url: str) -> Dict[str, Any]:
    """Build zeroed SEO response when no ranked data."""
    fallback_score = compute_professional_seo_score(
        estimated_traffic=0.0, keywords_count=0, top3_positions=0, top10_positions=0,
        avg_keyword_difficulty=None, competitor_avg_traffic=0.0,
    )
    return build_seo_response(
        search_performance_score=fallback_score,
        organic_visitors=0, total_search_volume=0, keywords_ranking=0, top3_positions=0,
        search_visibility_percent=0, missed_searches_monthly=0, top_keywords=[], seo_next_steps=[],
        enrichment_status="complete",
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

    snapshot = get_cached_seo_snapshot(user, domain, start_current, cache_ttl, now_utc)
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
        enrichment_status = "complete" if (
            getattr(snapshot, "keywords_enriched_at", None)
            and getattr(snapshot, "seo_next_steps_refreshed_at", None)
            and getattr(snapshot, "keyword_action_suggestions_refreshed_at", None)
        ) else "pending"
        return build_seo_response(
            search_performance_score=int(snapshot.search_performance_score or 0),
            organic_visitors=int(snapshot.organic_visitors or 0),
            total_search_volume=int(getattr(snapshot, "total_search_volume", 0) or 0),
            keywords_ranking=int(snapshot.keywords_ranking or 0),
            top3_positions=int(snapshot.top3_positions or 0),
            search_visibility_percent=int(getattr(snapshot, "search_visibility_percent", 0) or 0),
            missed_searches_monthly=int(getattr(snapshot, "missed_searches_monthly", 0) or 0),
            top_keywords=getattr(snapshot, "top_keywords", None) or [],
            seo_next_steps=getattr(snapshot, "seo_next_steps", None) or [],
            enrichment_status=enrichment_status,
        )

    _dbg_ba84ae_log(
        hypothesisId="H2_cache_miss_about_to_call_ranked_keywords",
        location="accounts/dataforseo_utils.py:get_or_refresh_seo_score_for_user",
        message="cache miss: will call ranked_keywords/live",
        data={"domain": domain},
        runId="pre-fix",
    )
    location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
    language_code = getattr(settings, "DATAFORSEO_LANGUAGE_CODE", "en")
    items = fetch_ranked_keyword_items(domain, location_code, language_code, user)
    if not items:
        return _build_empty_seo_response(user, site_url)

    metrics = compute_ranked_metrics(items)
    # Ranked-only keywords for sync path: no gap/LLM enrichment here (done in background).
    top_keywords_ranked = sorted(
        metrics["top_keywords"],
        key=lambda x: x.get("search_volume", 0),
        reverse=True,
    )[:20]

    visibility = compute_visibility_metrics(
        top_keywords_ranked,
        metrics["estimated_traffic"],
        metrics["keywords_ranking"],
        metrics["top3_positions"],
        metrics["top10_positions"],
        metrics["avg_difficulty"],
        domain,
        location_code,
        language_code,
    )

    snapshot = save_seo_snapshot(
        user, start_current, domain, now_utc,
        organic_visitors=int(round(metrics["estimated_traffic"]) or 0),
        keywords_ranking=int(metrics["keywords_ranking"] or 0),
        top3_positions=int(metrics["top3_positions"] or 0),
        top_keywords_sorted=top_keywords_ranked,
        total_search_volume=int(visibility["total_search_volume"] or 0),
        missed_searches_monthly=int(visibility["missed_searches_monthly"] or 0),
        search_visibility_percent=int(visibility["search_visibility_percent"] or 0),
        search_performance_score=int(visibility["search_performance_score"] or 0),
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
        search_performance_score=int(visibility["search_performance_score"] or 0),
        organic_visitors=int(round(metrics["estimated_traffic"]) or 0),
        total_search_volume=int(visibility["total_search_volume"] or 0),
        keywords_ranking=int(metrics["keywords_ranking"] or 0),
        top3_positions=int(metrics["top3_positions"] or 0),
        search_visibility_percent=int(visibility["search_visibility_percent"] or 0),
        missed_searches_monthly=int(visibility["missed_searches_monthly"] or 0),
        top_keywords=top_keywords_ranked,
        seo_next_steps=[],
        enrichment_status="pending",
    )

