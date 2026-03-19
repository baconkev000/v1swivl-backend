from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0023_add_seo_competitor_domains_override"),
    ]

    operations = [
        migrations.DeleteModel(
            name="GoogleAdsMetricsCache",
        ),
    ]
