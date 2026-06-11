from __future__ import annotations

from pgvector import Vector as PgVector

from rag.adapters.db.pool import Pool
from rag.domain.entities import Chunk, Document, Vector


class PgChunkStore:
    """Postgres-backed ChunkStore. One of the two persistence adapters holding SQL."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    def add_document(self, doc: Document, chunks: list[tuple[Chunk, Vector]]) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO documents (id, source, title, content) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content",
                (doc.id, doc.source, doc.title, doc.content),
            )
            for chunk, vec in chunks:
                conn.execute(
                    "INSERT INTO chunks "
                    "(id, document_id, ordinal, text, embedding, token_count) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "text = EXCLUDED.text, embedding = EXCLUDED.embedding",
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.text,
                        PgVector(vec),
                        chunk.token_count,
                    ),
                )
            conn.commit()

    def all_chunks(self) -> list[Chunk]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, document_id, ordinal, text, token_count FROM chunks "
                "ORDER BY document_id, ordinal"
            ).fetchall()
        return [
            Chunk(id=r[0], document_id=r[1], ordinal=r[2], text=r[3], token_count=r[4])
            for r in rows
        ]
