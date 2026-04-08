from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0051_fix_agg_stability_status_length"),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name="aeoresponsesnapshot",
            name="accounts_aeo_rsp_run_hash_platform_uq",
        ),
    ]

