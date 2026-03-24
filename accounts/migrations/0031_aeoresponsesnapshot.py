import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0030_seo_snapshot_location_context"),
    ]

    operations = [
        migrations.CreateModel(
            name="AEOResponseSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("prompt_text", models.TextField()),
                ("prompt_type", models.CharField(blank=True, default="", max_length=32)),
                ("weight", models.FloatField(default=1.0)),
                ("is_dynamic", models.BooleanField(default=False)),
                ("platform", models.CharField(default="openai", max_length=64)),
                ("model_name", models.CharField(blank=True, default="", max_length=128)),
                ("raw_response", models.TextField(blank=True, default="")),
                ("prompt_hash", models.CharField(db_index=True, max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="aeo_response_snapshots",
                        to="accounts.businessprofile",
                    ),
                ),
            ],
            options={
                "verbose_name": "AEO response snapshot",
                "verbose_name_plural": "AEO response snapshots",
                "ordering": ("-created_at",),
            },
        ),
        migrations.AddIndex(
            model_name="aeoresponsesnapshot",
            index=models.Index(
                fields=["profile", "prompt_hash", "created_at"],
                name="accounts_aeo_rsp_ph_cr",
            ),
        ),
    ]
