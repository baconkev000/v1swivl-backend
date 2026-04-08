from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0047_third_party_api_error_log"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="background_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                    ("skipped", "Skipped"),
                ],
                default="pending",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="phase1_completed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="phase1_provider_calls",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="aeoscoresnapshot",
            name="score_layer",
            field=models.CharField(
                choices=[("sample", "Sample"), ("confidence", "Confidence")],
                db_index=True,
                default="confidence",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="aeoscoresnapshot",
            name="execution_run",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="score_snapshots",
                to="accounts.aeoexecutionrun",
            ),
        ),
        migrations.CreateModel(
            name="AEOPromptExecutionAggregate",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("prompt_hash", models.CharField(db_index=True, max_length=64)),
                ("prompt_text", models.TextField(blank=True, default="")),
                ("prompt_type", models.CharField(blank=True, default="", max_length=32)),
                ("prompt_category", models.CharField(blank=True, default="", max_length=32)),
                ("openai_pass_count", models.IntegerField(default=0)),
                ("gemini_pass_count", models.IntegerField(default=0)),
                ("openai_brand_cited_count", models.IntegerField(default=0)),
                ("gemini_brand_cited_count", models.IntegerField(default=0)),
                ("openai_wrong_url_count", models.IntegerField(default=0)),
                ("gemini_wrong_url_count", models.IntegerField(default=0)),
                ("total_pass_count", models.IntegerField(default=0)),
                ("total_brand_cited_count", models.IntegerField(default=0)),
                ("last_openai_competitors_json", models.JSONField(blank=True, default=list)),
                ("last_gemini_competitors_json", models.JSONField(blank=True, default=list)),
                ("last_openai_citations_json", models.JSONField(blank=True, default=list)),
                ("last_gemini_citations_json", models.JSONField(blank=True, default=list)),
                ("last_openai_brand_mentioned", models.BooleanField(default=False)),
                ("last_gemini_brand_mentioned", models.BooleanField(default=False)),
                ("openai_last_wrong_url_status", models.CharField(blank=True, default="", max_length=40)),
                ("gemini_last_wrong_url_status", models.CharField(blank=True, default="", max_length=40)),
                (
                    "stability_status",
                    models.CharField(
                        choices=[("pending", "Pending"), ("stable", "Stable"), ("unstable", "Unstable")],
                        db_index=True,
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("stability_reasons", models.JSONField(blank=True, default=list)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "execution_run",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="prompt_aggregates",
                        to="accounts.aeoexecutionrun",
                    ),
                ),
                (
                    "last_gemini_response_snapshot",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="+",
                        to="accounts.aeoresponsesnapshot",
                    ),
                ),
                (
                    "last_openai_response_snapshot",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="+",
                        to="accounts.aeoresponsesnapshot",
                    ),
                ),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="aeo_prompt_execution_aggregates",
                        to="accounts.businessprofile",
                    ),
                ),
            ],
            options={
                "verbose_name": "AEO prompt execution aggregate",
                "verbose_name_plural": "AEO prompt execution aggregates",
                "ordering": ("-updated_at", "-id"),
            },
        ),
        migrations.AddConstraint(
            model_name="aeopromptexecutionaggregate",
            constraint=models.UniqueConstraint(
                fields=("profile", "execution_run", "prompt_hash"),
                name="accounts_aeo_prompt_agg_profile_run_hash_uq",
            ),
        ),
        migrations.AddIndex(
            model_name="aeopromptexecutionaggregate",
            index=models.Index(
                fields=["profile", "execution_run", "stability_status"],
                name="accounts_aeo_prompt_agg_profile_run_status_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="aeopromptexecutionaggregate",
            index=models.Index(fields=["profile", "prompt_hash"], name="accounts_aeo_prompt_agg_profile_hash_idx"),
        ),
    ]

