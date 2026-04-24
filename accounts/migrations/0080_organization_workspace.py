# Generated manually for organization / multi-site workspace access.

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def backfill_organizations(apps, schema_editor):
    Organization = apps.get_model("accounts", "Organization")
    OrganizationMembership = apps.get_model("accounts", "OrganizationMembership")
    BusinessProfile = apps.get_model("accounts", "BusinessProfile")
    BusinessProfileMembership = apps.get_model("accounts", "BusinessProfileMembership")

    user_ids = (
        BusinessProfile.objects.values_list("user_id", flat=True)
        .distinct()
        .order_by("user_id")
    )
    for uid in user_ids:
        if uid is None:
            continue
        profiles = list(
            BusinessProfile.objects.filter(user_id=uid).order_by("-is_main", "id")
        )
        if not profiles:
            continue
        main = profiles[0]
        org = Organization.objects.create(
            owner_user_id=uid,
            name=str(getattr(main, "business_name", "") or "")[:255],
        )
        BusinessProfile.objects.filter(user_id=uid).update(organization_id=org.id)

        OrganizationMembership.objects.get_or_create(
            organization_id=org.id,
            user_id=uid,
            defaults={
                "role": "admin",
                "is_owner": True,
                "hidden_from_team_ui": False,
            },
        )

        main_profile = next((p for p in profiles if p.is_main), profiles[0])
        for m in BusinessProfileMembership.objects.filter(business_profile_id=main_profile.id):
            if int(m.user_id) == int(uid):
                continue
            OrganizationMembership.objects.get_or_create(
                organization_id=org.id,
                user_id=m.user_id,
                defaults={
                    "role": m.role,
                    "is_owner": False,
                    "hidden_from_team_ui": bool(m.hidden_from_team_ui),
                },
            )


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0079_businessprofile_aeo_custom_prompt_cap_bonus"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Organization",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(blank=True, default="", help_text="Display label; usually mirrors the main site business name.", max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "owner_user",
                    models.ForeignKey(
                        help_text="Primary Stripe/account holder for this workspace (same as first BusinessProfile.user).",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="owned_organizations",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Organization",
                "verbose_name_plural": "Organizations",
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="OrganizationMembership",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("role", models.CharField(choices=[("admin", "Admin"), ("member", "Member")], default="member", max_length=16)),
                (
                    "is_owner",
                    models.BooleanField(
                        default=False,
                        help_text="True for the primary account holder (same user as Organization.owner_user).",
                    ),
                ),
                (
                    "hidden_from_team_ui",
                    models.BooleanField(
                        default=False,
                        help_text="When true, keep membership for access control but hide from team lists.",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "organization",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="memberships",
                        to="accounts.organization",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="organization_memberships",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "unique_together": {("organization", "user")},
            },
        ),
        migrations.AddField(
            model_name="businessprofile",
            name="organization",
            field=models.ForeignKey(
                blank=True,
                help_text="Workspace grouping; all sites under one billing org share this row.",
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="business_profiles",
                to="accounts.organization",
            ),
        ),
        migrations.RunPython(backfill_organizations, migrations.RunPython.noop),
    ]
