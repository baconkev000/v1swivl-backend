from django.db import migrations, models

# Idempotent for DBs that already have these columns (manual / external migrations).
ADD_SQL = """
ALTER TABLE accounts_businessprofile
  ADD COLUMN IF NOT EXISTS seo_location_depth integer NOT NULL DEFAULT 0;
ALTER TABLE accounts_businessprofile
  ADD COLUMN IF NOT EXISTS seo_location_code integer NOT NULL DEFAULT 0;
ALTER TABLE accounts_businessprofile
  ADD COLUMN IF NOT EXISTS seo_location_label varchar(255) NOT NULL DEFAULT '';
"""


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0034_aeorecommendationrun"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunSQL(ADD_SQL, reverse_sql=migrations.RunSQL.noop),
            ],
            state_operations=[
                migrations.AddField(
                    model_name="businessprofile",
                    name="seo_location_depth",
                    field=models.IntegerField(
                        default=0,
                        help_text="Legacy; prefer seo_location_mode. Kept for DB compatibility and older clients.",
                    ),
                ),
                migrations.AddField(
                    model_name="businessprofile",
                    name="seo_location_code",
                    field=models.IntegerField(
                        default=0,
                        help_text="DataForSEO location code when seo_location_mode is local; 0 means unset / use default.",
                    ),
                ),
                migrations.AddField(
                    model_name="businessprofile",
                    name="seo_location_label",
                    field=models.CharField(
                        blank=True,
                        default="",
                        help_text="Human-readable location label for local SEO mode.",
                        max_length=255,
                    ),
                ),
            ],
        ),
    ]
