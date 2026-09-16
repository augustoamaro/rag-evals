from __future__ import annotations

import math

from rag.chunking.chunker import chunk_document
from rag.domain.entities import Chunk, Document, Vector
from rag.domain.ports import ChunkStore, Embedder


class IngestionService:
    def __init__(self, embedder: Embedder, store: ChunkStore) -> None:
        self._embedder = embedder
        self._store = store

    def ingest(self, docs: list[Document]) -> int:
        prepared = prepare_documents(self._embedder, docs)
        self._store.replace_corpus(prepared)
        return sum(len(chunks) for _, chunks in prepared)


def prepare_documents(
    embedder: Embedder,
    docs: list[Document],
    *,
    target_words: int = 180,
    overlap_words: int = 30,
) -> list[tuple[Document, list[tuple[Chunk, Vector]]]]:
    """Prepare a complete corpus before either legacy replacement or publication."""
    document_ids: set[str] = set()
    for doc in docs:
        if doc.id in document_ids:
            raise ValueError(f"duplicate document id: {doc.id!r}")
        document_ids.add(doc.id)
    prepared: list[tuple[Document, list[tuple[Chunk, Vector]]]] = []
    for doc in docs:
        chunks = chunk_document(
            doc, target_words=target_words, overlap_words=overlap_words
        )
        if not chunks:
            prepared.append((doc, []))
            continue
        vectors = embedder.embed([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                "embedding count does not match chunk count: "
                f"expected {len(chunks)}, got {len(vectors)}"
            )
        expected_dim = embedder.dim
        if expected_dim <= 0:
            raise ValueError(f"embedding dimension must be positive, got {expected_dim}")
        for vector in vectors:
            if len(vector) != expected_dim:
                raise ValueError(
                    "embedding dimension does not match embedder: "
                    f"expected {expected_dim}, got {len(vector)}"
                )
            try:
                finite = all(math.isfinite(value) for value in vector)
            except TypeError as exc:
                raise ValueError("embedding values must be numeric") from exc
            if not finite:
                raise ValueError("embedding values must be finite")
        prepared.append((doc, list(zip(chunks, vectors, strict=True))))
    return prepared
