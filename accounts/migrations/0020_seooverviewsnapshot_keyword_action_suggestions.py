from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0019_seooverviewsnapshot_keywords_enriched_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="seooverviewsnapshot",
            name="keyword_action_suggestions",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="seooverviewsnapshot",
            name="keyword_action_suggestions_refreshed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]

