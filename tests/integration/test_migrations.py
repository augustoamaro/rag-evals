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
    assert {"documents", "chunks", "eval_runs", "eval_case_results"} <= names
