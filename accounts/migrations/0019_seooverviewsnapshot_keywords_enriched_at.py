from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0018_seooverviewsnapshot_seo_next_steps_refreshed_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="seooverviewsnapshot",
            name="keywords_enriched_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
