from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0042_onboardingonpagecrawl_labs_and_clusters"),
    ]

    operations = [
        migrations.CreateModel(
            name="ThirdPartyApiRequestLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "provider",
                    models.CharField(
                        choices=[
                            ("dataforseo", "DataForSEO"),
                            ("openai", "OpenAI"),
                            ("gemini", "Google Gemini"),
                        ],
                        db_index=True,
                        max_length=32,
                    ),
                ),
                (
                    "operation",
                    models.CharField(
                        blank=True,
                        default="",
                        help_text="Endpoint or logical operation name (e.g. DataForSEO path, openai.chat operation).",
                        max_length=512,
                    ),
                ),
                (
                    "cost_usd",
                    models.DecimalField(
                        blank=True,
                        decimal_places=6,
                        help_text="Provider-reported or estimated cost in USD.",
                        max_digits=14,
                        null=True,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                (
                    "business_profile",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=models.SET_NULL,
                        related_name="third_party_api_logs",
                        to="accounts.businessprofile",
                    ),
                ),
            ],
            options={
                "verbose_name": "Third-party API request",
                "verbose_name_plural": "Third-party API requests",
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="thirdpartyapirequestlog",
            index=models.Index(fields=["provider", "created_at"], name="acc_tpapi_prov_crt_idx"),
        ),
        migrations.AddIndex(
            model_name="thirdpartyapirequestlog",
            index=models.Index(
                fields=["business_profile", "created_at"],
                name="acc_tpapi_prof_crt_idx",
            ),
        ),
    ]
