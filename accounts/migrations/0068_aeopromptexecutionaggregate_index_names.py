"""
Repair AEOPromptExecutionAggregate btree indexes to the canonical names from 0048.

If ``makemigrations`` emitted ``RenameIndex`` from ``accounts_aeo_prompt_agg_profile_run_status_idx``
but that index never existed under that name (auto-generated name differed, or 0048 did not apply),
``migrate`` fails. This migration renames a matching existing index or creates the missing one.

If ``makemigrations`` later emits ``RenameIndex`` for the same names, see
``0075_rename_accounts_aeo_prompt_agg_profile_run_status_idx_acct_aeoagg_runstat_idx_and_more``:
it reconciles duplicate short + legacy long indexes on PostgreSQL.
"""

from django.db import migrations


# <= 30 chars (Django E034); 0069 renames legacy long names from 0048 if needed.
STATUS_CANONICAL = "acct_aeoagg_runstat_idx"
HASH_CANONICAL = "acct_aeoagg_prhash_idx"
LEGACY_STATUS = "accounts_aeo_prompt_agg_profile_run_status_idx"
LEGACY_HASH = "accounts_aeo_prompt_agg_profile_hash_idx"
TABLE = "accounts_aeopromptexecutionaggregate"


def _index_exists(cursor, name: str) -> bool:
    cursor.execute(
        """
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = current_schema()
          AND c.relkind = 'i'
          AND c.relname = %s
        """,
        [name],
    )
    return cursor.fetchone() is not None


def _find_rename_candidate(cursor, required: list[str], forbidden: list[str]) -> str | None:
    cursor.execute(
        """
        SELECT c.relname, pg_get_indexdef(i.indexrelid) AS def
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = current_schema()
          AND t.relname = %s
          AND NOT i.indisunique
        """,
        [TABLE],
    )
    for relname, defn in cursor.fetchall():
        if not relname or not defn:
            continue
        d = defn.lower()
        if any(x.lower() not in d for x in required):
            continue
        if any(x.lower() in d for x in forbidden):
            continue
        if relname in (STATUS_CANONICAL, HASH_CANONICAL, LEGACY_STATUS, LEGACY_HASH):
            continue
        return relname
    return None


def forwards(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    qn = schema_editor.quote_name
    with schema_editor.connection.cursor() as cursor:
        # Status composite index: (profile_id, execution_run_id, stability_status)
        if not _index_exists(cursor, STATUS_CANONICAL):
            old = _find_rename_candidate(
                cursor,
                required=["profile_id", "execution_run_id", "stability_status"],
                forbidden=["prompt_hash"],
            )
            if old:
                cursor.execute(
                    "ALTER INDEX {} RENAME TO {}".format(qn(old), qn(STATUS_CANONICAL)),
                )
            else:
                cursor.execute(
                    "CREATE INDEX {} ON {} (profile_id, execution_run_id, stability_status)".format(
                        qn(STATUS_CANONICAL),
                        qn(TABLE),
                    ),
                )

        # Hash composite index: (profile_id, prompt_hash)
        if not _index_exists(cursor, HASH_CANONICAL):
            old = _find_rename_candidate(
                cursor,
                required=["profile_id", "prompt_hash"],
                forbidden=["execution_run_id", "stability_status"],
            )
            if old:
                cursor.execute(
                    "ALTER INDEX {} RENAME TO {}".format(qn(old), qn(HASH_CANONICAL)),
                )
            else:
                cursor.execute(
                    "CREATE INDEX {} ON {} (profile_id, prompt_hash)".format(
                        qn(HASH_CANONICAL),
                        qn(TABLE),
                    ),
                )


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0067_businessprofilemembership"),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
