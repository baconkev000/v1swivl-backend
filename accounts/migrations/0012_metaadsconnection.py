# Generated for Meta (Facebook) Ads OAuth connection

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0011_googleadsmetricscache"),
    ]

    operations = [
        migrations.CreateModel(
            name="MetaAdsConnection",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("access_token", models.TextField(blank=True)),
                ("token_type", models.CharField(blank=True, default="Bearer", max_length=32)),
                ("expires_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="meta_ads_connection",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Meta Ads connection",
                "verbose_name_plural": "Meta Ads connections",
            },
        ),
    ]
