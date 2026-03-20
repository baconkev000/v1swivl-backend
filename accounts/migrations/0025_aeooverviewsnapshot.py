from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0024_remove_googleadsmetricscache"),
    ]

    operations = [
        migrations.CreateModel(
            name="AEOOverviewSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("domain", models.CharField(blank=True, max_length=255)),
                ("location_code", models.IntegerField(default=2840)),
                ("location_label", models.CharField(blank=True, default="", max_length=255)),
                ("niche", models.CharField(blank=True, default="", max_length=255)),
                ("aeo_score", models.IntegerField(default=0)),
                ("question_coverage_score", models.IntegerField(default=0)),
                ("questions_found", models.JSONField(blank=True, default=list)),
                ("questions_missing", models.JSONField(blank=True, default=list)),
                ("faq_readiness_score", models.IntegerField(default=0)),
                ("faq_blocks_found", models.IntegerField(default=0)),
                ("faq_schema_present", models.BooleanField(default=False)),
                ("snippet_readiness_score", models.IntegerField(default=0)),
                ("answer_blocks_found", models.IntegerField(default=0)),
                ("aeo_recommendations", models.JSONField(blank=True, default=list)),
                ("refreshed_at", models.DateTimeField(auto_now=True)),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="aeo_overview_snapshots",
                        to="accounts.businessprofile",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="aeo_overview_snapshots",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "AEO overview snapshot",
                "verbose_name_plural": "AEO overview snapshots",
                "unique_together": {("profile", "domain", "location_code")},
            },
        ),
    ]
