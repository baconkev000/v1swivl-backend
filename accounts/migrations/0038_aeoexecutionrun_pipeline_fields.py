from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0037_aeoexecutionrun"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="extraction_count",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="extraction_status",
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
            name="recommendation_run_id",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="recommendation_status",
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
            name="score_snapshot_id",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="scoring_status",
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
    ]

