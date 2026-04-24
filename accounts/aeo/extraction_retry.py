"""
Helpers for staff-initiated Phase-3 extraction retries (e.g. after provider rate limits).

Does not replace the main ``run_aeo_phase3_extraction_task`` orchestration; used when
snapshots on a completed run still lack ``AEOExtractionSnapshot`` rows.
"""

from __future__ import annotations

from django.db.models import Exists, OuterRef

from accounts.models import AEOExtractionSnapshot, AEOResponseSnapshot


def list_aeo_response_snapshot_ids_missing_extractions(
    execution_run_id: int,
    *,
    platform: str | None = None,
) -> list[int]:
    """
    Response snapshots for ``execution_run_id`` with zero related extraction rows.

    When ``platform`` is set (e.g. ``"perplexity"``), only that provider's rows are included.
    """
    rid = int(execution_run_id)
    has_ext = AEOExtractionSnapshot.objects.filter(response_snapshot_id=OuterRef("pk"))
    qs = (
        AEOResponseSnapshot.objects.filter(execution_run_id=rid)
        .annotate(_has_extraction=Exists(has_ext))
        .filter(_has_extraction=False)
    )
    plat = (platform or "").strip().lower()
    if plat:
        qs = qs.filter(platform=plat)
    return list(qs.order_by("id").values_list("id", flat=True))
