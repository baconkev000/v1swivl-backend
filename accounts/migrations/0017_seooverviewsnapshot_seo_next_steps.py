from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0016_seooverviewsnapshot_keyword_cache"),
    ]

    operations = [
        migrations.AddField(
            model_name="seooverviewsnapshot",
            name="seo_next_steps",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
