"""
Deterministic SEO/AEO issue detection and recommendation scaffolding.

Pipeline:
Data -> Clustered keyword issues -> Priority scoring -> Structured recommendations
"""
# ruff: noqa: E501,I001,UP035,UP006,UP045,TRY300,PLR2004,C901,PLR0912,COM812
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Literal, Optional

IssueCategory = Literal["content", "technical", "authority", "local", "competitor"]
IssuePriority = Literal["high", "medium", "low"]
IntentType = Literal["transactional", "local", "informational", "navigational"]

MAX_STRUCTURED_ISSUES = 8
MAX_CREATE_PAGE_ISSUES = 2

SERVICE_INTENT_HINTS = {
    "service",
    "services",
    "company",
    "provider",
    "repair",
    "install",
    "installation",
    "consultant",
    "agency",
    "clinic",
    "dentist",
    "plumber",
    "contractor",
}
TRANSACTIONAL_HINTS = {
    "buy",
    "pricing",
    "price",
    "cost",
    "quote",
    "book",
    "hire",
    "near",
    "appointment",
    "estimate",
}
DECISION_HINTS = {"vs", "versus", "best", "top", "compare", "comparison", "alternatives"}
NON_RELEVANT_HINTS = {
    "toothpaste",
    "toothbrush",
    "shampoo",
    "conditioner",
    "makeup",
    "perfume",
}


@dataclass(frozen=True)
class SeoIssue:
    issue_id: str
    category: IssueCategory
    priority: IssuePriority
    impact_score: float
    effort_score: float
    confidence: float
    evidence: Dict[str, Any]
    recommended_action_type: str


ISSUE_META: Dict[str, Dict[str, Any]] = {
    "missing_keyword_page": {
        "category": "content",
        "action_type": "create_cluster_page",
        "effort": 0.55,
        "confidence": 0.95,
    },
    "improve_existing_page": {
        "category": "content",
        "action_type": "optimize_existing_page",
        "effort": 0.35,
        "confidence": 0.9,
    },
    "competitor_outperforming": {
        "category": "competitor",
        "action_type": "competitor_gap_content",
        "effort": 0.45,
        "confidence": 0.9,
    },
    "insufficient_content_depth": {
        "category": "content",
        "action_type": "expand_content_depth",
        "effort": 0.6,
        "confidence": 0.8,
    },
    "missing_faq_for_question_intent": {
        "category": "technical",
        "action_type": "add_faq_schema_and_section",
        "effort": 0.25,
        "confidence": 0.82,
    },
    "missing_quick_answer_block": {
        "category": "content",
        "action_type": "add_quick_answer_block",
        "effort": 0.25,
        "confidence": 0.85,
    },
    "missing_structured_facts": {
        "category": "content",
        "action_type": "add_structured_facts",
        "effort": 0.3,
        "confidence": 0.8,
    },
    "missing_comparison_table": {
        "category": "content",
        "action_type": "add_comparison_table",
        "effort": 0.35,
        "confidence": 0.78,
    },
    "weak_entity_signals": {
        "category": "authority",
        "action_type": "strengthen_entity_signals",
        "effort": 0.35,
        "confidence": 0.78,
    },
    "missing_local_trust_signals": {
        "category": "local",
        "action_type": "add_local_trust_signals",
        "effort": 0.35,
        "confidence": 0.8,
    },
}


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        iv = int(value)
        if iv <= 0:
            return None
        return iv
    except (TypeError, ValueError):
        return None


def _to_float_01(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        f = 0.0
    return max(0.0, min(1.0, f))


def _normalize_search_volume(search_volume: Any, max_volume: int) -> float:
    try:
        sv = max(0, int(search_volume or 0))
    except (TypeError, ValueError):
        sv = 0
    if max_volume <= 0:
        return 0.0
    return max(0.0, min(1.0, sv / max_volume))


def _priority_weight(priority: str) -> int:
    if priority == "high":
        return 3
    if priority == "medium":
        return 2
    return 1


def _stem_token(token: str) -> str:
    t = token.strip().lower()
    for suffix in ("ing", "ed", "ies", "es", "s"):
        if len(t) > len(suffix) + 2 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def _keyword_tokens(text: str) -> list[str]:
    if not text:
        return []
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem_token(tok) for tok in raw if tok]


def _keyword_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_norm = " ".join(_keyword_tokens(a))
    b_norm = " ".join(_keyword_tokens(b))
    if not a_norm or not b_norm:
        return 0.0
    seq = SequenceMatcher(None, a_norm, b_norm).ratio()
    a_set, b_set = set(a_norm.split()), set(b_norm.split())
    if not a_set or not b_set:
        return seq
    jacc = len(a_set & b_set) / float(len(a_set | b_set))
    return max(seq, jacc)


def _keyword_cluster_id(primary_keyword: str, index: int) -> str:
    norm = "-".join(_keyword_tokens(primary_keyword)[:5])
    return f"kwc_{index}_{norm or 'cluster'}"


def _classify_intent(keyword: str, *, brand_tokens: set[str] | None = None) -> tuple[IntentType, float]:
    kw = (keyword or "").strip().lower()
    toks = set(_keyword_tokens(kw))
    if not kw:
        return "informational", 0.0
    if "near me" in kw or " in " in kw and any(x in kw for x in ("city", "downtown", "near")):
        return "local", 0.95
    if toks & TRANSACTIONAL_HINTS:
        return "transactional", 0.9
    if toks & SERVICE_INTENT_HINTS:
        return "transactional", 0.78
    if brand_tokens and (toks & brand_tokens):
        return "navigational", 0.8
    return "informational", 0.65


def _business_relevance(keyword: str, *, business_terms: set[str], blog_strategy_enabled: bool) -> float:
    toks = set(_keyword_tokens(keyword))
    if not toks:
        return 0.0
    if blog_strategy_enabled:
        return 1.0
    if toks & NON_RELEVANT_HINTS:
        return 0.25
    if business_terms:
        overlap = len(toks & business_terms)
        if overlap >= 2:
            return 1.0
        if overlap == 1:
            return 0.8
    return 0.65


def _question_intent_keyword(keyword: str) -> bool:
    kw = (keyword or "").strip().lower()
    if not kw:
        return False
    starters = ("how", "what", "why", "when", "where", "who", "which", "can", "does", "is", "are")
    if kw.startswith(starters) or "?" in kw:
        return True
    return " vs " in kw


def _serp_has_question_intent(serp_rows: Iterable[Dict[str, Any]]) -> bool:
    for row in serp_rows:
        row_type = str((row or {}).get("type") or "").lower()
        if "people_also_ask" in row_type or "related_questions" in row_type:
            return True
    return False


def _cluster_keywords(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    sorted_rows = sorted(
        [r for r in rows if str((r or {}).get("keyword") or "").strip()],
        key=lambda r: int((r or {}).get("search_volume") or 0),
        reverse=True,
    )
    for row in sorted_rows:
        kw = str((row or {}).get("keyword") or "").strip()
        placed = False
        for cluster in clusters:
            primary = str(cluster["primary_keyword"])
            sim = _keyword_similarity(kw, primary)
            if sim >= 0.72:
                cluster["rows"].append(row)
                related = cluster["related_keywords"]
                if kw not in related:
                    related.append(kw)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "cluster_id": "",
                    "primary_keyword": kw,
                    "related_keywords": [kw],
                    "rows": [row],
                }
            )
    for idx, c in enumerate(clusters, start=1):
        c["cluster_id"] = _keyword_cluster_id(str(c["primary_keyword"]), idx)
    return clusters


def _best_rank_from_rows(rows: list[dict[str, Any]]) -> int | None:
    ranks: list[int] = []
    for row in rows:
        rank = _to_int_or_none((row or {}).get("rank"))
        if rank is not None:
            ranks.append(rank)
    return min(ranks) if ranks else None


def _priority_from_intent_and_volume(
    issue_id: str,
    *,
    intent_type: str,
    search_volume: int,
    business_relevance: float,
) -> IssuePriority:
    if business_relevance < 0.35:
        return "low"
    if intent_type in {"transactional", "local"} and search_volume > 500:
        return "high"
    if issue_id in {"competitor_outperforming", "missing_local_trust_signals"} and search_volume > 350:
        return "high"
    if intent_type == "informational" and search_volume > 700:
        return "medium"
    if intent_type == "navigational" and search_volume > 1000:
        return "medium"
    if search_volume > 500 and business_relevance >= 0.7:
        return "medium"
    return "low"


def _score_issue(issue_id: str, evidence: Dict[str, Any], max_search_volume: int) -> SeoIssue:
    meta = ISSUE_META[issue_id]
    rank = _to_int_or_none(evidence.get("rank"))
    sv = int(evidence.get("search_volume") or 0)
    sv_norm = _normalize_search_volume(sv, max_search_volume)
    if rank is None:
        rank_gap = 1.0
    else:
        rank_gap = max(0.1, min(1.0, rank / 60.0))
    business_relevance = _to_float_01(evidence.get("business_relevance", 1.0))
    impact_score = _to_float_01(((0.65 * sv_norm) + (0.35 * rank_gap)) * max(0.45, business_relevance))
    confidence = _to_float_01(meta["confidence"])
    effort_score = _to_float_01(meta["effort"])
    priority = _priority_from_intent_and_volume(
        issue_id,
        intent_type=str(evidence.get("intent_type") or "informational"),
        search_volume=sv,
        business_relevance=business_relevance,
    )
    return SeoIssue(
        issue_id=issue_id,
        category=meta["category"],
        priority=priority,
        impact_score=impact_score,
        effort_score=effort_score,
        confidence=confidence,
        evidence=evidence,
        recommended_action_type=meta["action_type"],
    )


def _issue_rank_key(issue: SeoIssue) -> tuple[float, float]:
    bonus = 0.0
    if issue.issue_id == "missing_local_trust_signals":
        bonus = 2.0
    return (
        (_priority_weight(issue.priority) * 10.0)
        + (issue.impact_score * issue.confidence * (1.0 - issue.effort_score))
        + bonus,
        issue.impact_score,
    )


def _reduce_issues_for_quality(issues: list[SeoIssue]) -> list[SeoIssue]:
    best_by_cluster: dict[str, SeoIssue] = {}
    sitewide: list[SeoIssue] = []
    for issue in issues:
        cluster_id = str((issue.evidence or {}).get("cluster_id") or "")
        if cluster_id and not cluster_id.startswith("sitewide"):
            current = best_by_cluster.get(cluster_id)
            if current is None or _issue_rank_key(issue) > _issue_rank_key(current):
                best_by_cluster[cluster_id] = issue
        else:
            sitewide.append(issue)

    merged = list(best_by_cluster.values()) + sitewide
    merged.sort(key=_issue_rank_key, reverse=True)

    out: list[SeoIssue] = []
    create_page_count = 0
    for issue in merged:
        if issue.recommended_action_type == "create_cluster_page":
            if create_page_count >= MAX_CREATE_PAGE_ISSUES:
                continue
            create_page_count += 1
        out.append(issue)
        if len(out) >= MAX_STRUCTURED_ISSUES:
            break
    return out


def build_structured_issues(
    *,
    ranked_keywords: Optional[List[Dict[str, Any]]] = None,
    domain_intersection: Optional[List[Dict[str, Any]]] = None,
    on_page: Optional[Dict[str, Any]] = None,
    serp: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    ranked_keywords = list(ranked_keywords or [])
    domain_intersection = list(domain_intersection or [])
    on_page = dict(on_page or {})
    serp = list(serp or [])

    all_volumes: List[int] = []
    for row in ranked_keywords + domain_intersection:
        try:
            all_volumes.append(max(0, int((row or {}).get("search_volume") or 0)))
        except (TypeError, ValueError):
            continue
    max_search_volume = max(all_volumes) if all_volumes else 0

    raw_business = " ".join(
        [
            str(on_page.get("business_name") or ""),
            str(on_page.get("industry") or ""),
            str(on_page.get("business_description") or ""),
        ]
    )
    business_terms = set(_keyword_tokens(raw_business))
    if not business_terms:
        for row in ranked_keywords[:12]:
            toks = _keyword_tokens(str((row or {}).get("keyword") or ""))
            business_terms.update(toks[:2])
    blog_strategy_enabled = bool(on_page.get("blog_strategy_enabled"))

    cluster_rows = ranked_keywords if ranked_keywords else domain_intersection
    clusters = _cluster_keywords(cluster_rows)
    cluster_by_keyword: dict[str, dict[str, Any]] = {}
    for c in clusters:
        for kw in list(c.get("related_keywords") or []):
            cluster_by_keyword[kw.strip().lower()] = c

    issues: list[SeoIssue] = []

    has_answer_signal = any(k in on_page for k in ("answer_blocks_found", "quick_answer_present", "snippet_readiness_score"))
    answer_blocks_found = int(on_page.get("answer_blocks_found") or 0)
    quick_answer_present = bool(on_page.get("quick_answer_present")) or answer_blocks_found > 0
    has_structured_facts_signal = any(k in on_page for k in ("structured_facts_present", "facts_block_present", "table_blocks_found"))
    structured_facts_present = bool(on_page.get("structured_facts_present") or on_page.get("facts_block_present"))
    table_blocks_found = int(on_page.get("table_blocks_found") or 0)
    comparison_table_present = bool(on_page.get("comparison_table_present")) or table_blocks_found > 0

    has_on_page_faq_signal = (
        "faq_schema_present" in on_page
        or "faq_content_present" in on_page
        or "faq_blocks_found" in on_page
    )
    faq_schema_present = bool(on_page.get("faq_schema_present"))
    faq_content_present = bool(on_page.get("faq_content_present"))
    faq_blocks_found = int(on_page.get("faq_blocks_found") or 0)
    missing_faq = (not faq_schema_present) and (not faq_content_present) and faq_blocks_found <= 0

    for cluster in clusters:
        rows = list(cluster.get("rows") or [])
        primary_row = max(rows, key=lambda r: int((r or {}).get("search_volume") or 0))
        primary_keyword = str(cluster.get("primary_keyword") or "").strip()
        target_keywords = [str(x) for x in list(cluster.get("related_keywords") or []) if str(x).strip()]
        search_volume = max(0, int(sum(int((r or {}).get("search_volume") or 0) for r in rows)))
        rank = _best_rank_from_rows(rows)
        row_url = str((primary_row or {}).get("url") or (primary_row or {}).get("your_url") or "").strip()
        intent_type, intent_confidence = _classify_intent(
            primary_keyword,
            brand_tokens=business_terms,
        )
        business_relevance = _business_relevance(
            primary_keyword,
            business_terms=business_terms,
            blog_strategy_enabled=blog_strategy_enabled,
        )
        common = {
            "cluster_id": str(cluster.get("cluster_id") or ""),
            "primary_keyword": primary_keyword,
            "target_keywords": target_keywords,
            "related_keywords": target_keywords,
            "keyword": primary_keyword,
            "search_volume": search_volume,
            "rank": rank,
            "url": row_url,
            "intent_type": intent_type,
            "intent_confidence": round(intent_confidence, 2),
            "business_relevance": business_relevance,
        }

        if search_volume > 500 and (rank is None or rank > 50):
            issues.append(_score_issue("missing_keyword_page", common, max_search_volume))
        elif rank is not None and rank > 10 and search_volume > 300:
            issues.append(_score_issue("improve_existing_page", common, max_search_volume))

        comp_rows = [
            r for r in domain_intersection if str((r or {}).get("keyword") or "").strip().lower() in set(k.lower() for k in target_keywords)
        ]
        best_comp_rank: int | None = None
        best_comp_domain = ""
        best_comp_url = ""
        for row in comp_rows:
            comp_rank = _to_int_or_none((row or {}).get("competitor_rank")) or _to_int_or_none((row or {}).get("top_competitor_rank"))
            if comp_rank is None:
                continue
            if best_comp_rank is None or comp_rank < best_comp_rank:
                best_comp_rank = comp_rank
                best_comp_domain = str((row or {}).get("top_competitor_domain") or "").strip()
                best_comp_url = str(
                    (row or {}).get("competitor_url")
                    or (row or {}).get("top_competitor_url")
                    or (row or {}).get("top_competitor")
                    or ""
                ).strip()
        if best_comp_rank is not None and best_comp_rank <= 5 and (rank is None or rank > 20):
            ev = dict(common)
            ev.update(
                {
                    "competitor_rank": best_comp_rank,
                    "competitor_domain": best_comp_domain,
                    "competitor_url": best_comp_url,
                    "your_url": row_url,
                }
            )
            issues.append(_score_issue("competitor_outperforming", ev, max_search_volume))

        if has_answer_signal and not quick_answer_present and search_volume > 300:
            issues.append(_score_issue("missing_quick_answer_block", dict(common), max_search_volume))

        if has_structured_facts_signal and (not structured_facts_present) and search_volume > 300:
            issues.append(_score_issue("missing_structured_facts", dict(common), max_search_volume))

        decision_intent = any(x in primary_keyword.lower() for x in DECISION_HINTS)
        if decision_intent and has_structured_facts_signal and (not comparison_table_present):
            issues.append(_score_issue("missing_comparison_table", dict(common), max_search_volume))

        cluster_question_intent = _question_intent_keyword(primary_keyword)
        if has_on_page_faq_signal and missing_faq and cluster_question_intent:
            faq_ev = dict(common)
            faq_ev.update(
                {
                    "faq_schema_present": faq_schema_present,
                    "faq_content_present": faq_content_present,
                    "faq_blocks_found": faq_blocks_found,
                    "question_intent_detected": True,
                    "question_intent_source": "keyword",
                }
            )
            issues.append(_score_issue("missing_faq_for_question_intent", faq_ev, max_search_volume))

        if intent_type == "local":
            local_signals = int(on_page.get("local_trust_signals_count") or 0)
            has_local_schema = bool(on_page.get("local_business_schema_present"))
            has_reviews = bool(on_page.get("review_signals_present"))
            if local_signals <= 0 and not has_local_schema and not has_reviews:
                local_ev = dict(common)
                local_ev.update(
                    {
                        "local_trust_signals_count": local_signals,
                        "local_business_schema_present": has_local_schema,
                        "review_signals_present": has_reviews,
                    }
                )
                issues.append(_score_issue("missing_local_trust_signals", local_ev, max_search_volume))

    # Sitewide AEO/SEO issues.
    user_word_count = _to_int_or_none(on_page.get("user_word_count"))
    competitor_word_counts_raw = [
        _to_int_or_none(x) for x in list(on_page.get("competitor_word_counts") or [])
    ]
    competitor_word_counts: list[int] = [int(x) for x in competitor_word_counts_raw if x is not None]
    if user_word_count and competitor_word_counts:
        top3 = sorted(competitor_word_counts, reverse=True)[:3]
        competitor_avg = sum(top3) / len(top3)
        if competitor_avg > user_word_count * 1.5:
            evidence = {
                "cluster_id": "sitewide_content_depth",
                "target_keywords": [str((r or {}).get("keyword") or "") for r in ranked_keywords[:6] if str((r or {}).get("keyword") or "").strip()],
                "user_word_count": user_word_count,
                "competitor_word_counts": top3,
                "competitor_avg_word_count": competitor_avg,
                "url": on_page.get("user_url"),
                "search_volume": int(on_page.get("search_volume") or 0),
                "rank": _to_int_or_none(on_page.get("rank")),
                "keyword": str(on_page.get("keyword") or ""),
                "intent_type": "informational",
                "intent_confidence": 0.55,
                "business_relevance": 1.0,
            }
            issues.append(_score_issue("insufficient_content_depth", evidence, max_search_volume))

    has_entity_signal = any(k in on_page for k in ("organization_schema_present", "author_info_present", "about_page_present", "entity_mentions_count"))
    if has_entity_signal:
        org_schema = bool(on_page.get("organization_schema_present"))
        author_info = bool(on_page.get("author_info_present"))
        about_page = bool(on_page.get("about_page_present"))
        mentions = int(on_page.get("entity_mentions_count") or 0)
        if not org_schema or (mentions <= 1 and not about_page and not author_info):
            ev = {
                "cluster_id": "sitewide_entity_signals",
                "target_keywords": [str((r or {}).get("keyword") or "") for r in ranked_keywords[:5] if str((r or {}).get("keyword") or "").strip()],
                "organization_schema_present": org_schema,
                "author_info_present": author_info,
                "about_page_present": about_page,
                "entity_mentions_count": mentions,
                "search_volume": int(sum(int((r or {}).get("search_volume") or 0) for r in ranked_keywords[:5])),
                "rank": _best_rank_from_rows(ranked_keywords[:5]),
                "intent_type": "informational",
                "intent_confidence": 0.55,
                "business_relevance": 0.95,
            }
            issues.append(_score_issue("weak_entity_signals", ev, max_search_volume))

    question_intent_any = any(
        _question_intent_keyword(str((row or {}).get("keyword") or "")) for row in ranked_keywords
    ) or _serp_has_question_intent(serp)
    if has_on_page_faq_signal and missing_faq and question_intent_any:
        faq_evidence = {
            "cluster_id": "sitewide_faq_intent",
            "faq_schema_present": faq_schema_present,
            "faq_content_present": faq_content_present,
            "faq_blocks_found": faq_blocks_found,
            "question_intent_detected": question_intent_any,
            "question_intent_source": "serp_or_keyword",
            "keywords": [str((r or {}).get("keyword") or "") for r in ranked_keywords[:6]],
            "target_keywords": [str((r or {}).get("keyword") or "") for r in ranked_keywords[:6]],
            "search_volume": int(sum(int((r or {}).get("search_volume") or 0) for r in ranked_keywords[:6])),
            "rank": _best_rank_from_rows(ranked_keywords[:6]),
            "intent_type": "informational",
            "intent_confidence": 0.8,
            "business_relevance": 1.0,
        }
        issues.append(_score_issue("missing_faq_for_question_intent", faq_evidence, max_search_volume))

    reduced = _reduce_issues_for_quality(issues)
    return [issue.__dict__ for issue in reduced]


def _recommendation_type(issue_id: str) -> str:
    if issue_id in {"missing_keyword_page"}:
        return "page_creation"
    if issue_id in {"improve_existing_page", "missing_quick_answer_block", "missing_structured_facts", "missing_comparison_table", "missing_faq_for_question_intent", "insufficient_content_depth"}:
        return "content_improvement"
    if issue_id in {"competitor_outperforming", "weak_entity_signals", "missing_local_trust_signals"}:
        return "authority"
    return "technical"


def _issue_type(issue_id: str) -> str:
    if issue_id in {"missing_keyword_page", "improve_existing_page", "insufficient_content_depth"}:
        return "content_gap"
    if issue_id in {"missing_faq_for_question_intent"}:
        return "faq_gap"
    if issue_id in {"missing_local_trust_signals"}:
        return "local_trust_gap"
    if issue_id in {"competitor_outperforming", "weak_entity_signals"}:
        return "authority_gap"
    if issue_id in {"missing_quick_answer_block", "missing_structured_facts", "missing_comparison_table"}:
        return "content_gap"
    return "technical_gap"


def _ctr_for_rank(rank: int | None) -> float:
    if rank is None:
        return 0.002
    if rank <= 1:
        return 0.28
    if rank <= 2:
        return 0.15
    if rank <= 3:
        return 0.11
    if rank <= 4:
        return 0.08
    if rank <= 5:
        return 0.06
    if rank <= 6:
        return 0.045
    if rank <= 7:
        return 0.035
    if rank <= 8:
        return 0.028
    if rank <= 9:
        return 0.022
    if rank <= 10:
        return 0.018
    if rank <= 20:
        return 0.010
    if rank <= 50:
        return 0.004
    return 0.001


def _target_rank_for_issue(issue_id: str, rank: int | None) -> int:
    if rank is None:
        return 8
    if issue_id in {"missing_keyword_page", "competitor_outperforming"}:
        return 8
    if rank > 20:
        return 10
    if rank > 10:
        return 6
    return max(2, rank - 2)


def _impact_payload(issue_id: str, *, search_volume: int, rank: int | None, priority: str, confidence: float) -> dict[str, Any]:
    current_ctr = _ctr_for_rank(rank)
    target_ctr = _ctr_for_rank(_target_rank_for_issue(issue_id, rank))
    gain = max(0.0, target_ctr - current_ctr)
    traffic_gain_estimate = int(round(max(0, search_volume) * gain))
    if priority == "high" and issue_id in {
        "missing_quick_answer_block",
        "missing_structured_facts",
        "missing_comparison_table",
        "missing_faq_for_question_intent",
        "missing_local_trust_signals",
    }:
        lift = "high"
    elif issue_id in {"missing_keyword_page", "improve_existing_page", "insufficient_content_depth", "weak_entity_signals"}:
        lift = "medium"
    else:
        lift = "low"
    return {
        "traffic_gain_estimate": traffic_gain_estimate,
        "ai_citation_lift": lift,
        "confidence": round(max(0.0, min(1.0, float(confidence or 0.0))), 3),
    }


def _detected_issues(issue_id: str, ev: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if issue_id == "missing_keyword_page":
        out.append(
            f"Ranking URL not detected in analyzed signals for '{ev.get('primary_keyword') or ev.get('keyword') or 'target keyword'}'",
        )
        out.append(f"Search volume is {int(ev.get('search_volume') or 0)} per month")
    elif issue_id == "improve_existing_page":
        out.append(f"Current best rank is {ev.get('rank')} for cluster keyword")
        out.append(f"Search volume is {int(ev.get('search_volume') or 0)} per month")
    elif issue_id == "competitor_outperforming":
        out.append(f"Competitor rank is {ev.get('competitor_rank')} while your rank is {ev.get('rank')}")
        if ev.get("competitor_domain"):
            out.append(f"Competing domain detected: {ev.get('competitor_domain')}")
    elif issue_id == "insufficient_content_depth":
        out.append(f"Your page word count is {int(ev.get('user_word_count') or 0)}")
        out.append(f"Top competitor average word count is {int(ev.get('competitor_avg_word_count') or 0)}")
    elif issue_id == "missing_faq_for_question_intent":
        out.append("Question intent detected from keyword/PAA signals")
        out.append(f"FAQ blocks found: {int(ev.get('faq_blocks_found') or 0)}")
        out.append(f"FAQ schema present: {bool(ev.get('faq_schema_present'))}")
    elif issue_id == "missing_quick_answer_block":
        out.append("Quick answer block not detected near top of analyzed page content")
        out.append("Quick-answer block count in analyzed extraction: zero or unavailable (not proof of absence site-wide)")
    elif issue_id == "missing_structured_facts":
        out.append("Structured facts block (bullets/table) not detected in analyzed page content")
        out.append("Pricing/timeline/facts not surfaced in a scannable format in extracted content")
    elif issue_id == "missing_comparison_table":
        out.append("Decision-intent keyword cluster detected")
        out.append("Comparison table not detected on analyzed target page")
    elif issue_id == "weak_entity_signals":
        out.append(f"Organization schema present in analyzed signals: {bool(ev.get('organization_schema_present'))}")
        out.append(f"About page signal present in analyzed signals: {bool(ev.get('about_page_present'))}")
    elif issue_id == "missing_local_trust_signals":
        out.append("Address not detected in extracted page content used for this audit")
        out.append("LocalBusiness JSON-LD not detected in crawl/schema signals used for this audit")
        out.append("Customer reviews not detected in extracted content / signals used for this audit")
    if not out:
        out.append("Deterministic issue signal detected from ranking and on-page evidence")
    return out


def _competing_domains(ev: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("competitor_domain",):
        val = str(ev.get(key) or "").strip().lower()
        if val and val not in out:
            out.append(val)
    return out[:6]


def _cluster_url(primary_keyword: str, fallback: str) -> str:
    if fallback:
        return fallback
    toks = [t for t in _keyword_tokens(primary_keyword) if t]
    slug = "-".join(toks[:6]) or "topic"
    return f"/{slug}/"


def _faq_suggestions(primary_keyword: str, target_keywords: list[str], *, include_local: bool = False) -> list[dict[str, str]]:
    base = primary_keyword or (target_keywords[0] if target_keywords else "this service")
    faqs = [
        {
            "question": f"What is {base} and who is it for?",
            "answer": f"{base} is a service for people who need clear outcomes with predictable steps. It is best for customers who want measurable results and transparent expectations.",
        },
        {
            "question": f"How much does {base} cost?",
            "answer": f"Pricing depends on scope and urgency. List a starter range, what is included, and what changes the final price so users and AI answers can quote accurate ranges.",
        },
        {
            "question": f"How long does {base} usually take?",
            "answer": f"Most projects follow a defined timeline with an assessment, execution, and review phase. Publish a realistic timeframe and note factors that can extend it.",
        },
    ]
    if include_local:
        faqs.append(
            {
                "question": "What areas do you serve?",
                "answer": "List exact service areas and response times by location so local-intent users can confirm fit quickly.",
            }
        )
    return faqs[:5]


def _citation_block(primary_keyword: str, *, include_local: bool = False) -> dict[str, Any]:
    bullets = [
        f"{primary_keyword or 'Service'} typical timeline: state normal range and fast-track option.",
        f"{primary_keyword or 'Service'} pricing structure: base price, optional add-ons, and what is included.",
        "Best fit criteria: clearly state who should choose this option and who should not.",
    ]
    if include_local:
        bullets.append("Service area and verification: exact neighborhoods/cities, business credentials, and review count.")
    return {
        "title": f"Quick facts: {primary_keyword or 'service'}",
        "bullets": bullets,
    }


def build_structured_recommendation(issue: Dict[str, Any]) -> Dict[str, Any]:
    issue_id = str(issue.get("issue_id") or "")
    ev = dict(issue.get("evidence") or {})
    keyword = str(ev.get("keyword") or ev.get("primary_keyword") or "").strip()
    url = str(ev.get("url") or ev.get("your_url") or "").strip()
    competitor_domain = str(ev.get("competitor_domain") or "").strip()
    search_volume = int(ev.get("search_volume") or 0)
    target_keywords = [str(x).strip() for x in list(ev.get("target_keywords") or ev.get("related_keywords") or []) if str(x).strip()]
    if keyword and keyword not in target_keywords:
        target_keywords = [keyword] + target_keywords

    blueprint_url = _cluster_url(keyword, url)
    execution: dict[str, Any] = {}

    if issue_id == "missing_keyword_page":
        issue_label = "Create one authority page for clustered demand"
        why = (
            f"Demand for '{keyword}' and related queries is ~{search_volume}/mo in analyzed keyword data. "
            f"Publish one authority page structured for extractable answers and FAQs so answer engines can cite a single definitive URL "
            f"(observed rank in this audit: {ev.get('rank')})."
        )
        fix = (
            f"Create one new authority page at {blueprint_url} for keyword cluster '{keyword}'."
        )
        execution = {
            "page_blueprint": {
                "url": blueprint_url,
                "h1": f"{keyword} Guide: Cost, Timeline, and Best-Fit Criteria",
                "sections": [
                    "Quick answer (2-3 sentences)",
                    "Who this is for",
                    "Pricing and what affects cost",
                    "Timeline and implementation steps",
                    "FAQ (3-5 questions)",
                ],
                "internal_links": [
                    "/services/",
                    "/pricing/",
                    "/about/",
                ],
            },
            "faq_suggestions": _faq_suggestions(keyword, target_keywords),
            "ai_citation_block": _citation_block(keyword),
        }
        example = f"Use the page blueprint and FAQ block as-is, then tailor pricing numbers for {keyword}."
        aeo_boost = "One authoritative page with quick answers and facts gives assistants a single URL to cite across keyword variants."
    elif issue_id == "improve_existing_page":
        issue_label = f"Rebuild page structure for '{keyword}' cluster"
        why = (
            f"This cluster has demand (~{search_volume}/mo) but the current page rank is {ev.get('rank')}, which indicates weak answer structure."
        )
        fix = f"Rework {url or blueprint_url} with answer-first structure and explicit factual sections."
        execution = {
            "page_blueprint": {
                "url": url or blueprint_url,
                "h1": f"{keyword}: Direct answer, pricing, and timeline",
                "sections": [
                    "Quick answer block",
                    "Key facts table (price, timeline, fit)",
                    "Detailed steps",
                    "FAQ with 3-5 direct Q&A pairs",
                ],
                "internal_links": ["/pricing/", "/faq/", "/contact/"],
            },
            "ai_citation_block": _citation_block(keyword),
            "faq_suggestions": _faq_suggestions(keyword, target_keywords),
        }
        example = "Place the quick answer directly below H1 and keep each fact bullet under 20 words."
        aeo_boost = "Answer-first structure increases extractable passages that assistants can quote accurately."
    elif issue_id == "competitor_outperforming":
        issue_label = f"Publish competitor-gap page for '{keyword}'"
        why = (
            f"{competitor_domain or 'A competitor'} ranks at {ev.get('competitor_rank')} while your rank is {ev.get('rank')}; "
            f"demand is ~{search_volume}/mo."
        )
        fix = (
            f"Add or rebuild {blueprint_url} for '{keyword}' with a direct comparison section and factual proof blocks."
        )
        execution = {
            "page_blueprint": {
                "url": blueprint_url,
                "h1": f"{keyword}: Comparison, pricing, and outcomes",
                "sections": [
                    "Quick answer",
                    "Comparison table",
                    "Proof and outcomes",
                    "FAQ",
                ],
                "internal_links": ["/case-studies/", "/pricing/", "/contact/"],
            },
            "ai_citation_block": _citation_block(keyword),
            "faq_suggestions": _faq_suggestions(keyword, target_keywords),
        }
        example = "Include one table row per option with columns: best for, expected result, typical timeline."
        aeo_boost = "Comparison-ready facts improve inclusion in decision-style AI answers."
    elif issue_id == "insufficient_content_depth":
        issue_label = f"Expand depth on {url or blueprint_url}"
        why = (
            f"Your page has {int(ev.get('user_word_count') or 0)} words vs competitor average "
            f"{int(ev.get('competitor_avg_word_count') or 0)}."
        )
        fix = (
            f"Expand {url or blueprint_url} with additional depth and a structured facts block "
            f"(prioritize sections that are thin in analyzed page content)."
        )
        execution = {
            "page_blueprint": {
                "url": url or blueprint_url,
                "h1": f"{keyword or 'Service'}: complete implementation guide",
                "sections": [
                    "Quick answer",
                    "Step-by-step process",
                    "Costs and constraints",
                    "Common mistakes",
                    "FAQ",
                ],
                "internal_links": ["/services/", "/pricing/"],
            },
            "ai_citation_block": _citation_block(keyword),
        }
        example = "Each section should contain one short definition paragraph and one bulleted fact list."
        aeo_boost = "Expanded coverage gives AI systems more relevant passages for follow-up questions."
    elif issue_id == "missing_quick_answer_block":
        issue_label = "Add top-of-page quick answer block"
        why = (
            f"Search demand is ~{search_volume}/mo in analyzed data; add a quick answer block immediately below H1 "
            f"so assistants can extract a concise, citation-ready passage from {url or blueprint_url}."
        )
        fix = f"Insert a 2–3 sentence direct answer immediately below H1 on {url or blueprint_url}."
        execution = {
            "ai_citation_block": _citation_block(keyword),
        }
        example = "Start with: 'In short:' then provide one sentence on cost, one on timeline, one on ideal fit."
        aeo_boost = "Short direct answers are high-probability citation snippets."
    elif issue_id == "missing_structured_facts":
        issue_label = "Add structured facts section"
        why = (
            f"Search demand is ~{search_volume}/mo in analyzed data; publish a structured facts block (bullets or table) "
            f"on {url or blueprint_url} so answer engines can quote concrete ranges and constraints."
        )
        fix = f"Add a bullet list or table with pricing, duration, benefits, and limitations on {url or blueprint_url}."
        execution = {
            "ai_citation_block": _citation_block(keyword),
        }
        example = "Use 4 bullet points with numeric ranges where possible."
        aeo_boost = "Structured fact blocks increase quotable, low-ambiguity content for assistants."
    elif issue_id == "missing_comparison_table":
        issue_label = "Add comparison table for decision intent"
        why = (
            f"Decision-intent cluster (~{search_volume}/mo in analyzed data); add a comparison table under a dedicated H2 "
            f"on {url or blueprint_url} to strengthen extractable decision summaries for assistants."
        )
        fix = f"Add a comparison table under a dedicated H2 on {url or blueprint_url}."
        execution = {
            "page_blueprint": {
                "url": url or blueprint_url,
                "h1": f"{keyword}: options and differences",
                "sections": ["Quick answer", "Comparison table", "FAQ"],
                "internal_links": ["/pricing/", "/contact/"],
            },
            "ai_citation_block": _citation_block(keyword),
        }
        example = "Table columns: option, expected result, timeline, best fit."
        aeo_boost = "Comparison tables map directly to decision queries in AI-generated summaries."
    elif issue_id == "missing_faq_for_question_intent":
        issue_label = "Add FAQ block for question-intent traffic"
        why = (
            f"Question-style intent is present for this cluster (~{search_volume}/mo in analyzed data); "
            f"add or expand 3–5 FAQ Q&A pairs on {url or blueprint_url} so assistants can extract direct answers."
        )
        fix = f"Add 3–5 FAQ Q&A pairs on {url or blueprint_url} using direct wording from user intent."
        execution = {
            "faq_suggestions": _faq_suggestions(keyword, target_keywords),
            "ai_citation_block": _citation_block(keyword),
        }
        example = "Keep each answer between 2 and 4 sentences and include one concrete number/range where possible."
        aeo_boost = "Explicit Q&A format improves extraction for AI follow-up questions."
    elif issue_id == "weak_entity_signals":
        issue_label = "Build stronger entity trust signals"
        why = (
            "Strengthen verifiable entity signals (consistent business name, location, credentials, and proof) "
            "across About, footer, and key service pages so answer engines can attribute citations with higher confidence."
        )
        fix = "Add clear entity details and proof blocks on About, footer, and service pages."
        execution = {
            "ai_citation_block": {
                "title": "Company facts for citation",
                "bullets": [
                    "Who we are: one-sentence business definition.",
                    "Where we operate: city/region coverage.",
                    "Credentials: certifications, years in business, review count.",
                ],
            }
        }
        example = "Add the same business name, service area, and credential line consistently across key pages."
        aeo_boost = "Consistent entity signals improve source confidence and citation eligibility."
    elif issue_id == "missing_local_trust_signals":
        issue_label = "Build local trust block"
        why = (
            "Local-intent cluster is present in analyzed keyword data; add visible NAP, LocalBusiness JSON-LD, "
            "and first-party review proof on analyzed landing pages so location-aware answers can cite verifiable trust signals."
        )
        fix = (
            f"Add a local trust block on {url or blueprint_url} with visible NAP, service area, first-party reviews, "
            f"and credentials (verify and publish even if some signals exist outside analyzed excerpts)."
        )
        execution = {
            "local_trust_block": {
                "elements": ["address", "reviews", "credentials", "service_area"],
                "example_content": "Serving Denver metro including Aurora and Lakewood. Licensed team, 4.8/5 from 120+ reviews, same-week appointments available.",
            },
            "faq_suggestions": _faq_suggestions(keyword, target_keywords, include_local=True),
            "ai_citation_block": _citation_block(keyword, include_local=True),
        }
        example = "Place trust block above the fold on local landing pages and mirror it in footer/contact page."
        aeo_boost = "Local trust signals increase inclusion in location-aware AI answers."
    else:
        issue_label = "AEO opportunity detected"
        why = "Deterministic evidence indicates this page needs a specific citation-ready content block."
        fix = "Add one concrete execution block (answer, FAQ, facts, or trust) on the target page."
        execution = {
            "ai_citation_block": _citation_block(keyword),
        }
        example = ""
        aeo_boost = "Improves answerability and citation likelihood in assistant-generated responses."

    evidence_payload = {
        "cluster_id": str(ev.get("cluster_id") or ""),
        "primary_keyword": keyword,
        "search_volume": search_volume,
        "rank": ev.get("rank"),
        "detected_issues": _detected_issues(issue_id, ev),
        "competing_domains": _competing_domains(ev),
        "target_keywords": target_keywords[:12],
    }
    priority = str(issue.get("priority") or "medium")
    impact_payload = _impact_payload(
        issue_id,
        search_volume=search_volume,
        rank=_to_int_or_none(ev.get("rank")),
        priority=priority,
        confidence=float(issue.get("confidence") or 0.0),
    )

    return {
        "issue_id": issue_id,
        "issue_type": _issue_type(issue_id),
        "issue": issue_label,
        "priority": priority,
        "type": _recommendation_type(issue_id),
        "target_keywords": target_keywords[:12],
        "impact": impact_payload,
        "evidence": evidence_payload,
        "why_it_matters": why,
        "exact_fix": fix,
        "aeo_boost": aeo_boost,
        "example": example,
        "execution": execution,
        "raw_evidence": ev,
        "recommended_action_type": str(issue.get("recommended_action_type") or ""),
        "impact_score": float(issue.get("impact_score") or 0.0),
        "effort_score": float(issue.get("effort_score") or 0.0),
        "confidence": float(issue.get("confidence") or 0.0),
    }


def build_structured_recommendations(
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [build_structured_recommendation(issue) for issue in issues]
