from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0017_seooverviewsnapshot_seo_next_steps"),
    ]

    operations = [
        migrations.AddField(
            model_name="seooverviewsnapshot",
            name="seo_next_steps_refreshed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
