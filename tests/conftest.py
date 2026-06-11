from __future__ import annotations

import os

import pytest


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("RAG_DATABASE_URL")
    if not url:
        pytest.skip("RAG_DATABASE_URL not set — integration tests need Postgres")
    return url
