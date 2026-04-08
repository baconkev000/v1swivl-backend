from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0050_provider_stability_third_pass"),
    ]

    operations = [
        migrations.AlterField(
            model_name="aeopromptexecutionaggregate",
            name="stability_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("stable", "Stable"),
                    ("unstable", "Unstable"),
                    ("stabilized_after_third", "Stabilized After Third"),
                ],
                db_index=True,
                default="pending",
                max_length=28,
            ),
        ),
    ]

