import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0032_aeoextractionsnapshot"),
    ]

    operations = [
        migrations.CreateModel(
            name="AEOScoreSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("visibility_score", models.FloatField(default=0.0)),
                ("weighted_position_score", models.FloatField(default=0.0)),
                ("citation_share", models.FloatField(default=0.0)),
                ("competitor_dominance_json", models.JSONField(blank=True, default=dict)),
                ("total_prompts", models.IntegerField(default=0)),
                ("total_mentions", models.IntegerField(default=0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="aeo_score_snapshots",
                        to="accounts.businessprofile",
                    ),
                ),
            ],
            options={
                "verbose_name": "AEO score snapshot",
                "verbose_name_plural": "AEO score snapshots",
                "ordering": ("-created_at",),
            },
        ),
    ]
