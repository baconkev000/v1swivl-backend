from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0038_aeoexecutionrun_pipeline_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="seo_trigger_status",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
        migrations.AddField(
            model_name="aeoexecutionrun",
            name="seo_triggered_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]

