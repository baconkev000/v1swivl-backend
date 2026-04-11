# Paid tier: explicit "no plan" until Stripe maps a tier; backfill from subscription state.

from django.db import migrations, models


def clear_plan_without_active_subscription(apps, schema_editor):
    BusinessProfile = apps.get_model("accounts", "BusinessProfile")
    active = frozenset({"active", "trialing", "past_due"})
    unpaid_ids = []
    for bp in BusinessProfile.objects.iterator():
        sid = (getattr(bp, "stripe_subscription_id", "") or "").strip()
        st = (getattr(bp, "stripe_subscription_status", "") or "").strip().lower()
        if not (sid and st in active):
            unpaid_ids.append(bp.pk)
    if unpaid_ids:
        BusinessProfile.objects.filter(pk__in=unpaid_ids).update(plan="")


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0061_aeorecommendationrun_strategies_json"),
    ]

    operations = [
        migrations.AlterField(
            model_name="businessprofile",
            name="plan",
            field=models.CharField(
                blank=True,
                choices=[
                    ("", "No paid plan"),
                    ("starter", "Starter"),
                    ("pro", "Pro"),
                    ("advanced", "Advanced"),
                ],
                default="",
                max_length=16,
            ),
        ),
        migrations.RunPython(clear_plan_without_active_subscription, migrations.RunPython.noop),
    ]
