from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0048_progressive_aeo_onboarding"),
    ]

    operations = [
        migrations.AddField(
            model_name="onboardingonpagecrawl",
            name="prompt_plan_error",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="onboardingonpagecrawl",
            name="prompt_plan_finished_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="onboardingonpagecrawl",
            name="prompt_plan_prompt_count",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="onboardingonpagecrawl",
            name="prompt_plan_started_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="onboardingonpagecrawl",
            name="prompt_plan_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("queued", "Queued"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                ],
                db_index=True,
                default="pending",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="onboardingonpagecrawl",
            name="prompt_plan_task_id",
            field=models.CharField(blank=True, default="", max_length=128),
        ),
    ]

