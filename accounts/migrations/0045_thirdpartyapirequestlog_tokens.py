from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0043_third_party_api_request_log"),
    ]

    operations = [
        migrations.AddField(
            model_name="thirdpartyapirequestlog",
            name="tokens_received",
            field=models.PositiveIntegerField(
                blank=True,
                help_text="Output/completion tokens (OpenAI/Gemini); null for providers without token usage.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="thirdpartyapirequestlog",
            name="tokens_sent",
            field=models.PositiveIntegerField(
                blank=True,
                help_text="Input/prompt tokens (OpenAI/Gemini); null for providers without token usage.",
                null=True,
            ),
        ),
    ]
