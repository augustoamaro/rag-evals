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
