# Generated manually for AEO recommendation strategies (UI-ready grouping).

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0060_businessprofile_stripe_subscription_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeorecommendationrun",
            name="strategies_json",
            field=models.JSONField(
                blank=True,
                default=list,
                help_text="UI-ready hierarchical strategies (grouped by parent_group_id, deduped actions).",
            ),
        ),
    ]
