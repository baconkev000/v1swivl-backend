# Generated manually for per-profile SEO snapshot scoping.

from urllib.parse import urlparse

import django.db.models.deletion
from django.db import migrations, models


def _normalize_domain(site_url):
    if not site_url or not str(site_url).strip():
        return None
    parsed = urlparse(str(site_url).strip())
    domain = (parsed.netloc or parsed.path or "").strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain if domain else None


def forwards_backfill_business_profile(apps, schema_editor):
    SEOOverviewSnapshot = apps.get_model("accounts", "SEOOverviewSnapshot")
    BusinessProfile = apps.get_model("accounts", "BusinessProfile")

    for snap in SEOOverviewSnapshot.objects.filter(business_profile__isnull=True).iterator(chunk_size=200):
        uid = getattr(snap, "user_id", None)
        if not uid:
            continue
        dom = (getattr(snap, "cached_domain", None) or "").strip().lower()
        profiles = list(
            BusinessProfile.objects.filter(user_id=uid).order_by("-is_main", "id")
        )
        chosen = None
        if dom:
            for p in profiles:
                if _normalize_domain(getattr(p, "website_url", None) or "") == dom:
                    chosen = p
                    break
        if chosen is None and profiles:
            chosen = profiles[0]
        if chosen is not None:
            snap.business_profile_id = chosen.id
            snap.save(update_fields=["business_profile_id"])
        else:
            # Orphan row (no business profile for this user); drop it so NOT NULL FK can apply.
            snap.delete()


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0076_actionsgeneratedpagesnapshot_and_more"),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name="seooverviewsnapshot",
            unique_together=set(),
        ),
        migrations.AddField(
            model_name="seooverviewsnapshot",
            name="business_profile",
            field=models.ForeignKey(
                null=True,
                blank=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="seo_overview_snapshots",
                to="accounts.businessprofile",
            ),
        ),
        migrations.RunPython(forwards_backfill_business_profile, noop_reverse),
        migrations.AlterField(
            model_name="seooverviewsnapshot",
            name="business_profile",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="seo_overview_snapshots",
                to="accounts.businessprofile",
            ),
        ),
        migrations.AlterUniqueTogether(
            name="seooverviewsnapshot",
            unique_together={
                (
                    "business_profile",
                    "period_start",
                    "cached_location_mode",
                    "cached_location_code",
                ),
            },
        ),
    ]
