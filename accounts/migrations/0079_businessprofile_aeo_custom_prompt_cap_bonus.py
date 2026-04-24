from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0078_onboardingonpagecrawl_ranked_keywords_fetch_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="businessprofile",
            name="aeo_custom_prompt_cap_bonus",
            field=models.PositiveIntegerField(
                default=0,
                help_text=(
                    "Extra custom-prompt slots earned by deleting non-custom suggested prompts; "
                    "applies on top of plan custom cap."
                ),
            ),
        ),
    ]
