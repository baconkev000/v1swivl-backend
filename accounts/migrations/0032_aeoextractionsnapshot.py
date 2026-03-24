import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0031_aeoresponsesnapshot"),
    ]

    operations = [
        migrations.CreateModel(
            name="AEOExtractionSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("brand_mentioned", models.BooleanField(default=False)),
                (
                    "mention_position",
                    models.CharField(
                        choices=[
                            ("top", "Top"),
                            ("middle", "Middle"),
                            ("bottom", "Bottom"),
                            ("none", "None"),
                        ],
                        default="none",
                        max_length=16,
                    ),
                ),
                ("mention_count", models.IntegerField(default=0)),
                ("competitors_json", models.JSONField(blank=True, default=list)),
                ("citations_json", models.JSONField(blank=True, default=list)),
                (
                    "sentiment",
                    models.CharField(
                        choices=[
                            ("positive", "Positive"),
                            ("neutral", "Neutral"),
                            ("negative", "Negative"),
                        ],
                        default="neutral",
                        max_length=16,
                    ),
                ),
                ("confidence_score", models.FloatField(blank=True, null=True)),
                ("extraction_model", models.CharField(blank=True, default="", max_length=128)),
                (
                    "extraction_parse_failed",
                    models.BooleanField(
                        default=False,
                        help_text="True when JSON could not be parsed after retry; row stores safe defaults.",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "response_snapshot",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="extraction_snapshots",
                        to="accounts.aeoresponsesnapshot",
                    ),
                ),
            ],
            options={
                "verbose_name": "AEO extraction snapshot",
                "verbose_name_plural": "AEO extraction snapshots",
                "ordering": ("-created_at",),
            },
        ),
    ]
