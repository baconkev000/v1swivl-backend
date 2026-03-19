from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0022_allow_businessprofile_is_main_edit"),
    ]

    operations = [
        migrations.AddField(
            model_name="businessprofile",
            name="seo_competitor_domains_override",
            field=models.TextField(blank=True, default=""),
        ),
    ]

