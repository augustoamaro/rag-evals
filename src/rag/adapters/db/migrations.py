from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[4] / "migrations"


def apply_migrations(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> None:
    """Apply ordered ``*.sql`` migration files.

    Uses its own autocommit connection (not the pool) so the ``vector`` extension
    is created before any pooled connection tries to register the type.
    """
    sql_files = sorted(migrations_dir.glob("*.sql"))
    with psycopg.connect(database_url, autocommit=True) as conn:
        for path in sql_files:
            conn.execute(path.read_text().encode())
