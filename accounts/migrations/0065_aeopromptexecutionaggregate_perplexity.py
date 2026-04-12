from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0064_businessprofile_aeo_full_phase_eta_state"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_pass_count",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_brand_cited_count",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_wrong_url_count",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="last_perplexity_response_snapshot",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=models.SET_NULL,
                related_name="+",
                to="accounts.aeoresponsesnapshot",
            ),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="last_perplexity_competitors_json",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="last_perplexity_citations_json",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="last_perplexity_brand_mentioned",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_last_wrong_url_status",
            field=models.CharField(blank=True, default="", max_length=40),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_brand_mention_history",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_pass_history_json",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_stability_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("stable", "Stable"),
                    ("unstable", "Unstable"),
                    ("stabilized_after_third", "Stabilized After Third"),
                ],
                db_index=True,
                default="pending",
                max_length=28,
            ),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_third_pass_required",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="perplexity_third_pass_ran",
            field=models.BooleanField(default=False),
        ),
    ]
