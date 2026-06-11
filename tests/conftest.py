from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from rag.adapters.db.migrations import apply_migrations
from rag.adapters.db.pool import Pool, make_pool


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("RAG_DATABASE_URL")
    if not url:
        pytest.skip("RAG_DATABASE_URL not set — integration tests need Postgres")
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
