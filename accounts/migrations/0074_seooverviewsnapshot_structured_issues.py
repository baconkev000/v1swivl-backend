from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0073_alter_aeopromptexecutionaggregate_options_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="seooverviewsnapshot",
            name="seo_structured_issues",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="seooverviewsnapshot",
            name="seo_structured_issues_refreshed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
