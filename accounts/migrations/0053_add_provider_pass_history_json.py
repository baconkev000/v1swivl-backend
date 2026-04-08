from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0052_remove_response_run_hash_platform_unique"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="openai_pass_history_json",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="gemini_pass_history_json",
            field=models.JSONField(blank=True, default=list),
        ),
    ]

