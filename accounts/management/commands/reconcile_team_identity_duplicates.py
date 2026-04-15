from __future__ import annotations

from collections import defaultdict

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from accounts.user_identity_reconciliation import reconcile_user_identity_for_email

User = get_user_model()


class Command(BaseCommand):
    help = (
        "Reconcile duplicate users by email where team memberships/social auth are split "
        "across multiple rows."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--email",
            type=str,
            default="",
            help="Optional single email to reconcile.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report candidates only; do not merge rows.",
        )

    def handle(self, *args, **options):
        target_email = str(options.get("email") or "").strip().lower()
        dry_run = bool(options.get("dry_run"))
        emails: list[str] = []
        if target_email:
            emails = [target_email]
        else:
            buckets: dict[str, int] = defaultdict(int)
            for email in User.objects.exclude(email="").values_list("email", flat=True):
                norm = str(email or "").strip().lower()
                if norm:
                    buckets[norm] += 1
            emails = [e for e, count in buckets.items() if count > 1]

        if not emails:
            self.stdout.write("No duplicate emails found.")
            return

        self.stdout.write(f"Found {len(emails)} duplicate email candidate(s).")
        merged_total = 0
        for email in emails:
            rows = list(User.objects.filter(email__iexact=email).order_by("id"))
            ids = [int(u.id) for u in rows]
            self.stdout.write(f"- {email}: user_ids={ids}")
            if dry_run:
                continue
            result = reconcile_user_identity_for_email(email, reason="management_command")
            if result.reconciled:
                merged_total += len(result.merged_user_ids)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  merged into user_id={result.user.id} merged_user_ids={result.merged_user_ids}"
                    )
                )
            else:
                self.stdout.write("  no changes")

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run only. No changes written."))
            return
        self.stdout.write(self.style.SUCCESS(f"Done. Merged {merged_total} duplicate user row(s)."))
