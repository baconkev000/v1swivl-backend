from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0045_thirdpartyapirequestlog_tokens"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeoextractionsnapshot",
            name="brand_mentioned_url_status",
            field=models.CharField(
                blank=True,
                choices=[
                    ("matched", "Matched canonical domain"),
                    ("not_mentioned", "Not mentioned / no wrong-URL signal"),
                    ("mentioned_url_wrong_live", "Mentioned; wrong URL appears live"),
                    ("mentioned_url_wrong_broken", "Mentioned; wrong URL broken / non-resolving"),
                ],
                db_index=True,
                help_text=(
                    "How the model's attributed URL relates to BusinessProfile.website_url. "
                    "Null on legacy rows or failed parses. Phase 4 visibility still uses brand_mentioned + "
                    "competitor URL match only (wrong-URL cases stay uncredited numerically)."
                ),
                max_length=40,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="aeoextractionsnapshot",
            name="cited_domain_or_url",
            field=models.CharField(
                blank=True,
                default="",
                help_text="Normalized domain or URL the model tied to the business when status is wrong-URL or matched.",
                max_length=512,
            ),
        ),
        migrations.AddField(
            model_name="aeoextractionsnapshot",
            name="url_verification_notes",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Debug payload: dns_ok, http_ok, status_code, error, etc.",
            ),
        ),
        migrations.AddField(
            model_name="aeoextractionsnapshot",
            name="verified_at",
            field=models.DateTimeField(
                blank=True,
                help_text="When live/broken verification last ran (wrong-URL path only).",
                null=True,
            ),
        ),
    ]
