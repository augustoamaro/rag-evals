from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest

from rag.adapters.db.migrations import apply_migrations
from rag.adapters.db.pool import make_pool

pytestmark = pytest.mark.integration


def test_migrations_create_tables(database_url: str) -> None:
    apply_migrations(database_url)
    pool = make_pool(database_url)
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ).fetchall()
    finally:
        pool.close()
    names = {r[0] for r in rows}
    assert {
        "documents",
        "chunks",
        "eval_runs",
        "eval_case_results",
        "schema_migrations",
    } <= names


def test_migrations_apply_exactly_once(database_url: str) -> None:
    apply_migrations(database_url)
    apply_migrations(database_url)  # second run must be a no-op, not a re-apply
    pool = make_pool(database_url)
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT filename, count(*) FROM schema_migrations GROUP BY filename"
            ).fetchall()
    finally:
        pool.close()
    assert rows, "ledger should record applied migrations"
    assert all(count == 1 for _, count in rows)
    filenames = {name for name, _ in rows}
    assert "0001_init.sql" in filenames
    assert "0002_answer_latency.sql" in filenames


def test_concurrent_runners_apply_non_idempotent_migration_once(
    database_url: str, tmp_path: Path
) -> None:
    migration_name = "9001_concurrent_probe.sql"
    (tmp_path / migration_name).write_text(
        "SELECT pg_sleep(0.2); CREATE TABLE probe_table (id integer PRIMARY KEY);"
    )
    apply_migrations(database_url, tmp_path / "empty")
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS probe_table")
        conn.execute(
            "DELETE FROM schema_migrations WHERE filename = %s", (migration_name,)
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(apply_migrations, database_url, tmp_path)
                for _ in range(2)
            ]
            for future in futures:
                future.result()

        with psycopg.connect(database_url) as conn:
            ledger_count = conn.execute(
                "SELECT count(*) FROM schema_migrations WHERE filename = %s",
                (migration_name,),
            ).fetchone()
            table_exists = conn.execute("SELECT to_regclass('probe_table')").fetchone()
        assert ledger_count == (1,)
        assert table_exists == ("probe_table",)
    finally:
        with psycopg.connect(database_url) as conn:
            conn.execute("DROP TABLE IF EXISTS probe_table")
            conn.execute(
                "DELETE FROM schema_migrations WHERE filename = %s", (migration_name,)
            )


def test_ledger_failure_rolls_back_migration_ddl(
    database_url: str, tmp_path: Path
) -> None:
    migration_name = "9002_ledger_failure_probe.sql"
    (tmp_path / migration_name).write_text(
        "CREATE TABLE probe_table (id integer PRIMARY KEY);"
    )
    apply_migrations(database_url, tmp_path / "empty")
    with psycopg.connect(database_url) as conn:
        conn.execute("DROP TABLE IF EXISTS probe_table")
        conn.execute(
            "DELETE FROM schema_migrations WHERE filename = %s", (migration_name,)
        )
        conn.execute(
            "ALTER TABLE schema_migrations "
            "DROP CONSTRAINT IF EXISTS reject_probe_migration"
        )
        conn.execute(
            "ALTER TABLE schema_migrations ADD CONSTRAINT reject_probe_migration "
            "CHECK (filename <> '9002_ledger_failure_probe.sql')"
        )

    try:
        with pytest.raises(psycopg.errors.CheckViolation):
            apply_migrations(database_url, tmp_path)

        with psycopg.connect(database_url) as conn:
            table_exists = conn.execute("SELECT to_regclass('probe_table')").fetchone()
            ledger_count = conn.execute(
                "SELECT count(*) FROM schema_migrations WHERE filename = %s",
                (migration_name,),
            ).fetchone()
        assert table_exists == (None,)
        assert ledger_count == (0,)
    finally:
        with psycopg.connect(database_url) as conn:
            conn.execute(
                "ALTER TABLE schema_migrations "
                "DROP CONSTRAINT IF EXISTS reject_probe_migration"
            )
            conn.execute("DROP TABLE IF EXISTS probe_table")
