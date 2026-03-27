from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0036_businessprofile_selected_aeo_prompts"),
    ]

    operations = [
        migrations.CreateModel(
            name="AEOExecutionRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("prompt_count_requested", models.IntegerField(default=0)),
                ("prompt_count_executed", models.IntegerField(default=0)),
                ("prompt_count_failed", models.IntegerField(default=0)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("cache_hit", models.BooleanField(default=False)),
                (
                    "fetch_mode",
                    models.CharField(
                        choices=[("cache_hit", "Cache Hit"), ("fresh_fetch", "Fresh Fetch")],
                        default="fresh_fetch",
                        max_length=16,
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                            ("skipped_cached", "Skipped Cached"),
                        ],
                        default="pending",
                        max_length=24,
                    ),
                ),
                ("error_message", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="aeo_execution_runs",
                        to="accounts.businessprofile",
                    ),
                ),
            ],
            options={
                "verbose_name": "AEO execution run",
                "verbose_name_plural": "AEO execution runs",
                "ordering": ("-created_at",),
            },
        ),
    ]

