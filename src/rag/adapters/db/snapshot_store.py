from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict
from typing import Any, cast

from pgvector import Vector as PgVector
from psycopg.types.json import Jsonb

from rag.adapters.datasets.loader import dataset_from_payload
from rag.adapters.db.pool import Pool
from rag.chunking.chunker import chunk_document
from rag.domain.datasets import Dataset, dataset_digest, dataset_payload
from rag.domain.entities import Chunk, Document, Vector
from rag.domain.snapshots import IndexSnapshot

_DIMENSION = 384
type _Prepared = list[tuple[Document, list[tuple[Chunk, Vector]]]]
type _Row = tuple[str, str, int, str, PgVector, int, int, int]


class SnapshotStore:
    """Publish complete, content-addressed indexes without changing historical rows."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    def publish(
        self,
        dataset: Dataset,
        prepared: _Prepared,
        embedding: dict[str, object],
        chunking: dict[str, object],
    ) -> IndexSnapshot:
        payload = dataset_payload(dataset)
        validated_dataset = dataset_from_payload(payload)
        embedding, chunking = _configuration(embedding, chunking)
        rows, chunks_sha256 = _validate_prepared(validated_dataset, prepared, chunking)
        dataset_sha256 = dataset_digest(validated_dataset)
        index_id = "idx_" + _digest(
            {
                "schema_version": 1,
                "dataset_sha256": dataset_sha256,
                "embedding": embedding,
                "chunking": chunking,
                "chunks_sha256": chunks_sha256,
            }
        )
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (index_id,)
            )
            existing = conn.execute(
                "SELECT 1 FROM index_snapshots WHERE id = %s", (index_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO dataset_snapshots (sha256, payload) VALUES (%s, %s) "
                    "ON CONFLICT (sha256) DO NOTHING",
                    (dataset_sha256, Jsonb(payload)),
                )
                if rows:
                    with conn.cursor() as cursor:
                        cursor.executemany(
                            "INSERT INTO snapshot_chunks "
                            "(index_id, id, document_id, ordinal, text, embedding, "
                            "token_count, word_start, word_end) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            [(index_id, *row) for row in rows],
                        )
                # Inserting the parent is the final, irreversible seal. Deferred
                # chunk foreign keys and the parent commit atomically together.
                conn.execute(
                    "INSERT INTO index_snapshots "
                    "(id, dataset_sha256, embedding, chunking) VALUES (%s, %s, %s, %s)",
                    (index_id, dataset_sha256, Jsonb(embedding), Jsonb(chunking)),
                )
        return IndexSnapshot(index_id, validated_dataset, embedding, chunking)

    def get(self, index_id: str) -> IndexSnapshot | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT i.id, d.payload, i.embedding, i.chunking, d.sha256 "
                "FROM index_snapshots i JOIN dataset_snapshots d "
                "ON d.sha256 = i.dataset_sha256 WHERE i.id = %s",
                (index_id,),
            ).fetchone()
        return _snapshot(row) if row is not None else None

    def find(
        self,
        dataset_sha256: str,
        embedding: dict[str, object],
        chunking: dict[str, object],
    ) -> IndexSnapshot | None:
        embedding, chunking = _configuration(embedding, chunking)
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT i.id, d.payload, i.embedding, i.chunking, d.sha256 "
                "FROM index_snapshots i JOIN dataset_snapshots d "
                "ON d.sha256 = i.dataset_sha256 WHERE i.dataset_sha256 = %s "
                "AND i.embedding = %s AND i.chunking = %s ORDER BY i.id LIMIT 2",
                (dataset_sha256, Jsonb(embedding), Jsonb(chunking)),
            ).fetchall()
        if len(rows) > 1:
            raise ValueError(
                "multiple index snapshots match this dataset/model/configuration; "
                "select an explicit index id"
            )
        return _snapshot(rows[0]) if rows else None


class SnapshotChunkStore:
    """A read-only chunk view permanently bound to one published index."""

    def __init__(self, pool: Pool, index_id: str) -> None:
        self._pool = pool
        self._index_id = index_id

    def all_chunks(self) -> list[Chunk]:
        with self._pool.connection() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM index_snapshots WHERE id = %s", (self._index_id,)
                ).fetchone()
                is None
            ):
                raise ValueError(f"index snapshot {self._index_id!r} does not exist")
            rows = conn.execute(
                "SELECT id, document_id, ordinal, text, token_count, word_start, "
                "word_end FROM snapshot_chunks WHERE index_id = %s "
                "ORDER BY document_id, ordinal, id",
                (self._index_id,),
            ).fetchall()
        return [Chunk(*row) for row in rows]


def _snapshot(row: tuple[Any, ...]) -> IndexSnapshot:
    dataset = dataset_from_payload(row[1])
    if dataset_digest(dataset) != row[4]:
        raise ValueError("stored dataset snapshot does not match its content identity")
    embedding, chunking = _configuration(row[2], row[3])
    return IndexSnapshot(row[0], dataset, embedding, chunking)


def _configuration(
    embedding: dict[str, object], chunking: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    # Strict JSON also detaches the stored identity from mutable caller dictionaries.
    try:
        embedding = cast(dict[str, object], json.loads(_canonical(embedding)))
        chunking = cast(dict[str, object], json.loads(_canonical(chunking)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "snapshot configuration must contain finite JSON values"
        ) from exc
    dimension = embedding.get("dimension")
    if type(dimension) is not int or dimension != _DIMENSION:
        raise ValueError(f"snapshot embedding dimension must be {_DIMENSION}")
    if set(chunking) != {"algorithm", "target_words", "overlap_words"}:
        raise ValueError("chunking must define algorithm, target_words and overlap_words")
    if chunking["algorithm"] != "word_window_v1":
        raise ValueError("unsupported snapshot chunking algorithm")
    target, overlap = chunking["target_words"], chunking["overlap_words"]
    if type(target) is not int or type(overlap) is not int:
        raise ValueError("chunking word counts must be integers")
    if target <= 0 or not 0 <= overlap < target:
        raise ValueError("chunking requires target_words > overlap_words >= 0")
    return embedding, chunking


def _validate_prepared(
    dataset: Dataset, prepared: _Prepared, chunking: dict[str, object]
) -> tuple[list[_Row], str]:
    documents = {document.id: document for document in dataset.documents}
    supplied: dict[str, list[tuple[Chunk, Vector]]] = {}
    for document, chunks in prepared:
        if document.id in supplied:
            raise ValueError(f"duplicate prepared document id {document.id!r}")
        if documents.get(document.id) != document:
            raise ValueError("prepared document does not match the dataset snapshot")
        supplied[document.id] = chunks
    if supplied.keys() != documents.keys():
        raise ValueError("prepared documents do not cover the complete dataset")
    rows: list[_Row] = []
    content: list[dict[str, object]] = []
    for document_id in sorted(documents):
        expected = chunk_document(
            documents[document_id],
            target_words=cast(int, chunking["target_words"]),
            overlap_words=cast(int, chunking["overlap_words"]),
        )
        chunks = sorted(supplied[document_id], key=lambda item: item[0].ordinal)
        if [chunk for chunk, _ in chunks] != expected:
            raise ValueError("prepared chunks do not match dataset content and chunking")
        for chunk, vector in chunks:
            for value in (
                chunk.ordinal,
                chunk.token_count,
                chunk.word_start,
                chunk.word_end,
            ):
                if type(value) is not int:
                    raise ValueError("chunk positions and word counts must be integers")
            if len(vector) != _DIMENSION:
                raise ValueError(f"chunk vector dimension must be {_DIMENSION}")
            try:
                if any(
                    isinstance(value, bool) or not math.isfinite(value)
                    for value in vector
                ):
                    raise ValueError("chunk vector values must be finite numbers")
                vector_bytes = struct.pack(f"!{_DIMENSION}f", *vector)
            except (TypeError, OverflowError, struct.error) as exc:
                raise ValueError(
                    "chunk vector values must be finite float32 numbers"
                ) from exc
            values = struct.unpack(f"!{_DIMENSION}f", vector_bytes)
            if not any(values):
                raise ValueError("chunk vector must not be zero for cosine retrieval")
            assert chunk.word_start is not None and chunk.word_end is not None
            rows.append(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.ordinal,
                    chunk.text,
                    PgVector(values),
                    chunk.token_count,
                    chunk.word_start,
                    chunk.word_end,
                )
            )
            content.append(
                {
                    "chunk": asdict(chunk),
                    "vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
                }
            )
    return rows, _digest(content)


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()
