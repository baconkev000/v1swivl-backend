from __future__ import annotations

from django.core.management.base import BaseCommand

from accounts.aeo.competitor_snapshots import compute_and_save_competitor_snapshot
from accounts.models import AEOCompetitorSnapshot, BusinessProfile


class Command(BaseCommand):
    help = "Backfill competitor visibility snapshots for business profiles."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--force",
            action="store_true",
            help="Recompute for all profiles even when a snapshot already exists.",
        )

    def handle(self, *args, **options) -> None:
        force = bool(options.get("force"))
        created = 0
        skipped = 0
        failed = 0

        qs = BusinessProfile.objects.order_by("id").iterator(chunk_size=200)
        for profile in qs:
            try:
                exists = AEOCompetitorSnapshot.objects.filter(
                    profile=profile,
                    platform_scope="all",
                    window_start=None,
                    window_end=None,
                ).exists()
                if exists and not force:
                    skipped += 1
                    continue
                compute_and_save_competitor_snapshot(profile, platform_scope="all")
                created += 1
            except Exception as exc:
                failed += 1
                self.stderr.write(
                    self.style.WARNING(
                        f"profile_id={profile.id} failed: {type(exc).__name__}: {exc}",
                    ),
                )

        self.stdout.write(
            self.style.SUCCESS(
                f"backfill_competitor_snapshots done created={created} skipped={skipped} failed={failed}",
            ),
        )
