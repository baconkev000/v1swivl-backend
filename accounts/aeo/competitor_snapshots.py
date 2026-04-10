from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from django.db import transaction
from django.db.models import Prefetch, QuerySet
from django.utils import timezone

from accounts.models import (
    AEOCompetitorSnapshot,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    BusinessProfile,
)
from .aeo_extraction_utils import parse_competitor_raw_item, root_domain_from_fragment


@dataclass(frozen=True)
class CompetitorVisibilityRow:
    domain: str
    display_name: str
    appearances: int
    visibility_pct: float
    rank: int
    last_seen_at: datetime | None


@dataclass(frozen=True)
class CompetitorVisibilityResult:
    total_slots: int
    rows: list[CompetitorVisibilityRow]
    window_start: datetime | None
    window_end: datetime | None
    platform_scope: str


def _profile_root_domain(profile: BusinessProfile) -> str:
    return (root_domain_from_fragment(getattr(profile, "website_url", "") or "") or "").strip().lower()


def _latest_extraction_for_response(response: AEOResponseSnapshot) -> AEOExtractionSnapshot | None:
    all_rows = list(getattr(response, "_prefetched_objects_cache", {}).get("extraction_snapshots", []))
    if not all_rows:
        return None
    return all_rows[0]


def _scope_queryset(
    profile: BusinessProfile,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    platform_scope: str = "all",
) -> QuerySet[AEOResponseSnapshot]:
    qs = AEOResponseSnapshot.objects.filter(profile=profile)
    if platform_scope and platform_scope != "all":
        qs = qs.filter(platform=platform_scope)
    if window_start is not None:
        qs = qs.filter(created_at__gte=window_start)
    if window_end is not None:
        qs = qs.filter(created_at__lte=window_end)
    return qs.prefetch_related(
        Prefetch(
            "extraction_snapshots",
            queryset=AEOExtractionSnapshot.objects.order_by("-created_at", "-id"),
        ),
    ).order_by("id")


def calculate_competitor_visibility(
    profile: BusinessProfile,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    platform_scope: str = "all",
) -> CompetitorVisibilityResult:
    """
    Deterministic competitor-domain visibility:
    - denominator = response slots (AEOResponseSnapshot rows in scope)
    - numerator/domain = slots where domain appears in competitors_json (deduped per slot)
    """
    platform = (platform_scope or "all").strip().lower() or "all"
    responses = list(
        _scope_queryset(
            profile,
            window_start=window_start,
            window_end=window_end,
            platform_scope=platform,
        ),
    )
    total_slots = len(responses)
    if total_slots == 0:
        return CompetitorVisibilityResult(
            total_slots=0,
            rows=[],
            window_start=window_start,
            window_end=window_end,
            platform_scope=platform,
        )

    self_domain = _profile_root_domain(profile)
    appearances: dict[str, int] = defaultdict(int)
    display_name_by_domain: dict[str, str] = {}
    last_seen_at_by_domain: dict[str, datetime] = {}

    for response in responses:
        extraction = _latest_extraction_for_response(response)
        if extraction is None:
            continue
        raw_competitors = getattr(extraction, "competitors_json", None) or []
        if not isinstance(raw_competitors, list):
            continue
        domains_seen_in_slot: set[str] = set()
        names_for_slot: dict[str, str] = {}
        for raw in raw_competitors:
            parsed = parse_competitor_raw_item(raw)
            domain = (root_domain_from_fragment(parsed.get("url") or "") or "").strip().lower()
            if not domain:
                continue
            if self_domain and domain == self_domain:
                continue
            domains_seen_in_slot.add(domain)
            if domain not in names_for_slot:
                names_for_slot[domain] = (parsed.get("name") or "").strip()
        for dom in domains_seen_in_slot:
            appearances[dom] += 1
            if dom in names_for_slot and names_for_slot[dom]:
                display_name_by_domain.setdefault(dom, names_for_slot[dom])
            seen_at = response.created_at
            prev_seen_at = last_seen_at_by_domain.get(dom)
            if prev_seen_at is None or seen_at > prev_seen_at:
                last_seen_at_by_domain[dom] = seen_at

    ordered_domains = sorted(appearances.keys(), key=lambda d: (-appearances[d], d))
    rows: list[CompetitorVisibilityRow] = []
    for idx, dom in enumerate(ordered_domains, start=1):
        app = int(appearances[dom])
        pct = round((app / total_slots) * 100.0, 1)
        rows.append(
            CompetitorVisibilityRow(
                domain=dom,
                display_name=display_name_by_domain.get(dom) or dom,
                appearances=app,
                visibility_pct=pct,
                rank=idx,
                last_seen_at=last_seen_at_by_domain.get(dom),
            ),
        )
    return CompetitorVisibilityResult(
        total_slots=total_slots,
        rows=rows,
        window_start=window_start,
        window_end=window_end,
        platform_scope=platform,
    )


@transaction.atomic
def compute_and_save_competitor_snapshot(
    profile: BusinessProfile,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    platform_scope: str = "all",
) -> AEOCompetitorSnapshot:
    result = calculate_competitor_visibility(
        profile,
        window_start=window_start,
        window_end=window_end,
        platform_scope=platform_scope,
    )
    rows_json: list[dict[str, Any]] = []
    for row in result.rows:
        rows_json.append(
            {
                "domain": row.domain,
                "display_name": row.display_name,
                "appearances": row.appearances,
                "visibility_pct": row.visibility_pct,
                "rank": row.rank,
                "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
            },
        )
    snapshot, _created = AEOCompetitorSnapshot.objects.update_or_create(
        profile=profile,
        platform_scope=result.platform_scope,
        window_start=result.window_start,
        window_end=result.window_end,
        defaults={
            "total_slots": result.total_slots,
            "rows_json": rows_json,
            "updated_at": timezone.now(),
        },
    )
    return snapshot
