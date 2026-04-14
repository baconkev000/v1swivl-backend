from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0069_rename_aeo_agg_indexes_short"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeoresponsesnapshot",
            name="is_custom_prompt",
            field=models.BooleanField(
                default=False,
                help_text="True when this snapshot was produced from a user-added custom monitored prompt.",
            ),
        ),
        migrations.AddField(
            model_name="aeopromptexecutionaggregate",
            name="is_custom_prompt",
            field=models.BooleanField(
                default=False,
                help_text="True when this aggregate row tracks a user-added custom monitored prompt.",
            ),
        ),
    ]
