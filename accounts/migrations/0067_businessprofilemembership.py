from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def backfill_owner_memberships(apps, schema_editor):
    BusinessProfile = apps.get_model("accounts", "BusinessProfile")
    Membership = apps.get_model("accounts", "BusinessProfileMembership")
    for bp in BusinessProfile.objects.all().iterator():
        Membership.objects.get_or_create(
            business_profile_id=bp.id,
            user_id=bp.user_id,
            defaults={
                "role": "admin",
                "is_owner": True,
            },
        )


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("accounts", "0066_aeodashboardbundlecache"),
    ]

    operations = [
        migrations.CreateModel(
            name="BusinessProfileMembership",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("role", models.CharField(choices=[("admin", "Admin"), ("member", "Member")], default="member", max_length=16)),
                (
                    "is_owner",
                    models.BooleanField(
                        default=False,
                        help_text="True for the primary account holder (billing owner) of this business profile.",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "business_profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="team_memberships",
                        to="accounts.businessprofile",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="business_profile_memberships",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "unique_together": {("business_profile", "user")},
            },
        ),
        migrations.RunPython(backfill_owner_memberships, migrations.RunPython.noop),
    ]
