from __future__ import annotations

from typing import Any

from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

Pool = ConnectionPool[Connection[Any]]


def make_pool(database_url: str) -> Pool:
    """Open a connection pool with the pgvector type registered on each connection.

    Requires the ``vector`` extension to already exist (run migrations first).
    """
    return ConnectionPool(
        database_url, min_size=1, max_size=8, open=True, configure=_configure
    )


def _configure(conn: Connection[Any]) -> None:
    register_vector(conn)
