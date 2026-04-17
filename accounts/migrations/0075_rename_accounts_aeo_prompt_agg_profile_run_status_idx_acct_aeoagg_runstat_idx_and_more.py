"""
Reconcile AEOPromptExecutionAggregate btree index names with 0068 / 0069.

A plain ``RenameIndex`` on the database fails when the short canonical name already
exists (0068 may ``CREATE INDEX`` it while the 0048 long-named index remains).

We use ``SeparateDatabaseAndState``:
- **database_operations**: PostgreSQL-only ``RunPython`` drops the legacy index when
  both exist (after safety checks), or renames legacy -> canonical when only legacy exists.
- **state_operations**: ``RenameIndex`` only updates Django migration state (no SQL on DB),
  matching ``models.py`` index names.

After deploy, **rebuild** the Django image so this file replaces any older ``RenameIndex``-only
0075 in the image; ``git pull`` alone is not enough when Compose uses a cached image.
"""

from django.db import migrations


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


def _index_meta(cursor, index_name: str):
    cursor.execute(
        """
        SELECT i.indisunique, pg_get_indexdef(i.indexrelid)
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = current_schema()
          AND t.relname = %s
          AND c.relname = %s
        """,
        [TABLE, index_name],
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {"unique": bool(row[0]), "def": str(row[1] or "")}


def _def_matches(defn: str, required: list[str], forbidden: list[str]) -> bool:
    d = defn.lower()
    return all(x.lower() in d for x in required) and not any(x.lower() in d for x in forbidden)


def _reconcile_pair(cursor, qn, legacy: str, canonical: str, required: list[str], forbidden: list[str]) -> None:
    has_canonical = _index_exists(cursor, canonical)
    has_legacy = _index_exists(cursor, legacy)
    if has_canonical and has_legacy:
        meta = _index_meta(cursor, legacy)
        if not meta or meta["unique"]:
            return
        if not _def_matches(meta["def"], required, forbidden):
            return
        cursor.execute("DROP INDEX IF EXISTS {}".format(qn(legacy)))
        return
    if has_legacy and not has_canonical:
        meta = _index_meta(cursor, legacy)
        if not meta or meta["unique"]:
            return
        if not _def_matches(meta["def"], required, forbidden):
            return
        cursor.execute(
            "ALTER INDEX {} RENAME TO {}".format(qn(legacy), qn(canonical)),
        )


def forwards(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    qn = schema_editor.quote_name
    with schema_editor.connection.cursor() as cursor:
        _reconcile_pair(
            cursor,
            qn,
            LEGACY_STATUS,
            STATUS_CANONICAL,
            required=["profile_id", "execution_run_id", "stability_status"],
            forbidden=["prompt_hash"],
        )
        _reconcile_pair(
            cursor,
            qn,
            LEGACY_HASH,
            HASH_CANONICAL,
            required=["profile_id", "prompt_hash"],
            forbidden=["execution_run_id", "stability_status"],
        )


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0074_seooverviewsnapshot_structured_issues"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(forwards, migrations.RunPython.noop),
            ],
            state_operations=[
                migrations.RenameIndex(
                    model_name="aeopromptexecutionaggregate",
                    new_name=STATUS_CANONICAL,
                    old_name=LEGACY_STATUS,
                ),
                migrations.RenameIndex(
                    model_name="aeopromptexecutionaggregate",
                    new_name=HASH_CANONICAL,
                    old_name=LEGACY_HASH,
                ),
            ],
        ),
    ]
