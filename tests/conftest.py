from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from rag.adapters.db.migrations import apply_migrations
from rag.adapters.db.pool import Pool, make_pool


@pytest.fixture
def database_url() -> str:
    # Deliberately a SEPARATE variable from RAG_DATABASE_URL: the integration
    # suite truncates every table in the database it points at, so pointing it
    # somewhere must be an explicit opt-in — never inherited from dev config.
    url = os.environ.get("RAG_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "RAG_TEST_DATABASE_URL not set — integration tests need a disposable "
            "Postgres (the suite truncates its tables)"
        )
    return url


@pytest.fixture
def pool(database_url: str) -> Iterator[Pool]:
    apply_migrations(database_url)
    p = make_pool(database_url)
    with p.connection() as conn:
        conn.execute("TRUNCATE eval_case_results, eval_runs, chunks, documents CASCADE")
        conn.commit()
    yield p
    p.close()
