import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0033_aeoscoresnapshot"),
    ]

    operations = [
        migrations.CreateModel(
            name="AEORecommendationRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("recommendations_json", models.JSONField(blank=True, default=list)),
                ("visibility_score_at_run", models.FloatField(default=0.0)),
                ("weighted_position_score_at_run", models.FloatField(default=0.0)),
                ("citation_share_at_run", models.FloatField(default=0.0)),
                (
                    "actions_completed_json",
                    models.JSONField(
                        blank=True,
                        default=list,
                        help_text="Optional log of completed actions (indices, timestamps) for trend tracking.",
                    ),
                ),
                (
                    "effectiveness_json",
                    models.JSONField(
                        blank=True,
                        default=dict,
                        help_text="Placeholder for future outcome / effectiveness metrics per recommendation.",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="aeo_recommendation_runs",
                        to="accounts.businessprofile",
                    ),
                ),
                (
                    "score_snapshot",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="recommendation_runs",
                        to="accounts.aeoscoresnapshot",
                    ),
                ),
            ],
            options={
                "verbose_name": "AEO recommendation run",
                "verbose_name_plural": "AEO recommendation runs",
                "ordering": ("-created_at",),
            },
        ),
    ]
