from __future__ import annotations

import math
from typing import Any

from pgvector import Vector as PgVector
from psycopg import Connection

from rag.adapters.db.pool import Pool
from rag.domain.entities import Chunk, Document, Vector

_EMBEDDING_DIM = 384
type _ChunkRow = tuple[str, str, int, str, PgVector, int, int | None, int | None]


class PgChunkStore:
    """Postgres-backed ChunkStore. One of the two persistence adapters holding SQL."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    def add_document(self, doc: Document, chunks: list[tuple[Chunk, Vector]]) -> None:
        rows = _validate_and_build_rows([(doc, chunks)])
        with self._pool.connection() as conn, conn.transaction():
            conn.execute("LOCK TABLE documents IN SHARE ROW EXCLUSIVE MODE")
            conn.execute(
                "INSERT INTO documents (id, source, title, content) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "source = EXCLUDED.source, title = EXCLUDED.title, "
                "content = EXCLUDED.content",
                (doc.id, doc.source, doc.title, doc.content),
            )
            # Replace, don't upsert: if a document shrinks on re-ingest, an
            # upsert would leave orphan higher-ordinal chunks in retrieval.
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc.id,))
            _insert_chunk_rows(conn, rows)

    def replace_corpus(
        self, documents: list[tuple[Document, list[tuple[Chunk, Vector]]]]
    ) -> None:
        rows = _validate_and_build_rows(documents)
        with self._pool.connection() as conn, conn.transaction():
            # Serialize writers even when the old corpus is empty: DELETE alone
            # cannot lock rows that do not exist and can merge two replacements.
            conn.execute("LOCK TABLE documents IN SHARE ROW EXCLUSIVE MODE")
            conn.execute("DELETE FROM documents")
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents (id, source, title, content) "
                    "VALUES (%s, %s, %s, %s)",
                    [
                        (doc.id, doc.source, doc.title, doc.content)
                        for doc, _ in documents
                    ],
                )
            _insert_chunk_rows(conn, rows)

    def all_chunks(self) -> list[Chunk]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, document_id, ordinal, text, token_count, word_start, "
                "word_end FROM chunks "
                "ORDER BY document_id, ordinal"
            ).fetchall()
        return [
            Chunk(
                id=r[0],
                document_id=r[1],
                ordinal=r[2],
                text=r[3],
                token_count=r[4],
                word_start=r[5],
                word_end=r[6],
            )
            for r in rows
        ]


def _validate_and_build_rows(
    documents: list[tuple[Document, list[tuple[Chunk, Vector]]]],
) -> list[_ChunkRow]:
    document_ids: set[str] = set()
    chunk_ids: set[str] = set()
    rows: list[_ChunkRow] = []
    for doc, chunks in documents:
        if not doc.id:
            raise ValueError("document id must not be empty")
        if doc.id in document_ids:
            raise ValueError(f"duplicate document id: {doc.id!r}")
        document_ids.add(doc.id)
        ordinals: set[int] = set()
        for chunk, vector in chunks:
            if not chunk.id:
                raise ValueError("chunk id must not be empty")
            if chunk.id in chunk_ids:
                raise ValueError(f"duplicate chunk id: {chunk.id!r}")
            chunk_ids.add(chunk.id)
            if chunk.document_id != doc.id:
                raise ValueError(
                    f"chunk {chunk.id!r} belongs to {chunk.document_id!r}, not {doc.id!r}"
                )
            if chunk.ordinal < 0:
                raise ValueError(f"chunk {chunk.id!r} has a negative ordinal")
            if chunk.ordinal in ordinals:
                raise ValueError(
                    f"document {doc.id!r} has duplicate chunk ordinal {chunk.ordinal}"
                )
            ordinals.add(chunk.ordinal)
            if not chunk.text:
                raise ValueError(f"chunk {chunk.id!r} has empty text")
            if chunk.token_count <= 0:
                raise ValueError(f"chunk {chunk.id!r} has invalid token count")
            if len(vector) != _EMBEDDING_DIM:
                raise ValueError(
                    f"chunk {chunk.id!r} has vector dimension {len(vector)}; "
                    f"expected {_EMBEDDING_DIM}"
                )
            try:
                finite = all(math.isfinite(value) for value in vector)
            except TypeError as exc:
                raise ValueError(
                    f"chunk {chunk.id!r} has non-numeric vector values"
                ) from exc
            if not finite:
                raise ValueError(f"chunk {chunk.id!r} has non-finite vector values")
            rows.append(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.ordinal,
                    chunk.text,
                    PgVector(vector),
                    chunk.token_count,
                    chunk.word_start,
                    chunk.word_end,
                )
            )
    return rows


def _insert_chunk_rows(conn: Connection[Any], rows: list[_ChunkRow]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks "
            "(id, document_id, ordinal, text, embedding, token_count, "
            "word_start, word_end) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
