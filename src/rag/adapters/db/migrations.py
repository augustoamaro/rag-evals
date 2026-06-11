from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[4] / "migrations"


def apply_migrations(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> None:
    """Apply ordered ``*.sql`` migration files exactly once each.

    A ``schema_migrations`` ledger records applied filenames, so migrations do
    not need to be individually idempotent (the first ``ALTER TABLE`` would
    otherwise break a re-run). Uses its own autocommit connection (not the
    pool) so the ``vector`` extension exists before any pooled connection
    tries to register the type.
    """
    sql_files = sorted(migrations_dir.glob("*.sql"))
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "filename text PRIMARY KEY, "
            "applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {
            row[0]
            for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()
        }
        for path in sql_files:
            if path.name in applied:
                continue
            conn.execute(path.read_text().encode())
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
            )
