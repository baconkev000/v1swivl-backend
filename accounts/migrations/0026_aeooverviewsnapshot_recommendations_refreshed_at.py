from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0025_aeooverviewsnapshot"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeooverviewsnapshot",
            name="aeo_recommendations_refreshed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
