from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[4] / "migrations"
_MIGRATION_LOCK_ID = 7_243_319_072_024_091_601


def apply_migrations(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> None:
    """Apply ordered ``*.sql`` migration files exactly once each.

    A ``schema_migrations`` ledger records applied filenames. A session-level
    advisory lock serializes runners, and each file is committed together with
    its ledger entry so neither can survive without the other.
    """
    sql_files = sorted(migrations_dir.glob("*.sql"))
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
        try:
            with conn.transaction():
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "filename text PRIMARY KEY, "
                    "applied_at timestamptz NOT NULL DEFAULT now())"
                )
            applied = {
                row[0]
                for row in conn.execute(
                    "SELECT filename FROM schema_migrations"
                ).fetchall()
            }
            for path in sql_files:
                if path.name in applied:
                    continue
                with conn.transaction():
                    conn.execute(path.read_text().encode())
                    conn.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s)",
                        (path.name,),
                    )
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))
