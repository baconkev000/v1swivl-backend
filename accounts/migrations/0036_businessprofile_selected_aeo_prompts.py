from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0035_businessprofile_seo_location_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="businessprofile",
            name="selected_aeo_prompts",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
