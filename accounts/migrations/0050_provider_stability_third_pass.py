from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0049_onboarding_prompt_plan_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="gemini_brand_mention_history",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="openai_brand_mention_history",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="openai_stability_status",
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
            name="gemini_stability_status",
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
            name="openai_third_pass_required",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="gemini_third_pass_required",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="openai_third_pass_ran",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="gemini_third_pass_ran",
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name="aeopromptexecutionaggregate",
            name="stability_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("stable", "Stable"),
                    ("unstable", "Unstable"),
                    ("stabilized_after_third", "Stabilized After Third"),
                ],
                db_index=True,
                default="pending",
                max_length=16,
            ),
        ),
    ]

