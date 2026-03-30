# Generated manually for dual-provider AEO execution pairing.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0039_aeoexecutionrun_seo_trigger_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="aeoresponsesnapshot",
            name="execution_pair_id",
            field=models.UUIDField(
                blank=True,
                db_index=True,
                help_text="Shared by OpenAI + Gemini snapshots from the same logical prompt execution.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="aeoresponsesnapshot",
            name="execution_run",
            field=models.ForeignKey(
                blank=True,
                help_text="Pipeline run that produced this row (if any).",
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="response_snapshots",
                to="accounts.aeoexecutionrun",
            ),
        ),
        migrations.AddConstraint(
            model_name="aeoresponsesnapshot",
            constraint=models.UniqueConstraint(
                condition=models.Q(execution_run__isnull=False),
                fields=("execution_run", "prompt_hash", "platform"),
                name="accounts_aeo_rsp_run_hash_platform_uq",
            ),
        ),
    ]
