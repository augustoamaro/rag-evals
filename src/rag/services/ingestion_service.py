from __future__ import annotations

from rag.chunking.chunker import chunk_document
from rag.domain.entities import Document
from rag.domain.ports import ChunkStore, Embedder


class IngestionService:
    def __init__(self, embedder: Embedder, store: ChunkStore) -> None:
        self._embedder = embedder
        self._store = store

    def ingest(self, docs: list[Document]) -> int:
        total = 0
        for doc in docs:
            chunks = chunk_document(doc)
            if not chunks:
                continue
            vectors = self._embedder.embed([c.text for c in chunks])
            self._store.add_document(doc, list(zip(chunks, vectors, strict=True)))
            total += len(chunks)
        return total
