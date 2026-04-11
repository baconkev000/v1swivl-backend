from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0062_businessprofile_plan_none_default"),
    ]

    operations = [
        migrations.AddField(
            model_name="businessprofile",
            name="aeo_prompt_expansion_last_error",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="businessprofile",
            name="aeo_prompt_expansion_progress",
            field=models.PositiveIntegerField(
                default=0,
                help_text="len(selected_aeo_prompts) snapshot after last expansion step.",
            ),
        ),
        migrations.AddField(
            model_name="businessprofile",
            name="aeo_prompt_expansion_status",
            field=models.CharField(
                choices=[
                    ("idle", "Idle"),
                    ("queued", "Queued"),
                    ("running", "Running"),
                    ("complete", "Complete"),
                    ("error", "Error"),
                ],
                default="idle",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="businessprofile",
            name="aeo_prompt_expansion_target",
            field=models.PositiveIntegerField(
                blank=True,
                help_text="Monitored prompt cap when expansion last ran (plan-derived).",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="businessprofile",
            name="aeo_prompt_expansion_updated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
