from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0014_alter_onpageauditsnapshot_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="onpageauditsnapshot",
            name="pages_audited",
            field=models.IntegerField(default=0),
        ),
    ]

