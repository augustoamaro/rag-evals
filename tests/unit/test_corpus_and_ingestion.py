from __future__ import annotations

from rag.chunking.chunker import chunk_document
from rag.domain.entities import Chunk, Document, Vector
from rag.metrics.relevance import relevant_chunk_ids
from rag.services.corpus_loader import load_documents, load_golden
from rag.services.ingestion_service import IngestionService


class FakeEmbedder:
    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[Vector]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.docs: list[Document] = []
        self.chunks: list[Chunk] = []

    def add_document(self, doc: Document, chunks: list[tuple[Chunk, Vector]]) -> None:
        self.docs.append(doc)
        self.chunks.extend(c for c, _ in chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)


def test_loads_corpus_and_golden() -> None:
    docs = load_documents()
    cases = load_golden()
    assert len(docs) >= 8
    assert any(d.id == "cap-theorem" for d in docs)
    assert len(cases) >= 24
    assert all(c.relevant_snippets for c in cases)


def test_every_golden_case_anchors_to_a_real_chunk() -> None:
    # Guards that the golden set stays consistent with the corpus: every case's
    # snippets must match at least one actual chunk after ingestion.
    docs = load_documents()
    chunks = [c for d in docs for c in chunk_document(d)]
    for case in load_golden():
        rel = relevant_chunk_ids(chunks, case.relevant_snippets)
        assert rel, f"no chunk matches the relevant snippets for case {case.id!r}"


def test_ingestion_stores_all_chunks() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()
    service = IngestionService(embedder, store)
    docs = [
        Document(id="d1", source="s", title="t", content=" ".join(f"w{i}" for i in range(250))),
        Document(id="d2", source="s", title="t", content="short doc"),
    ]
    total = service.ingest(docs)
    assert total == len(store.chunks)
    assert {c.document_id for c in store.chunks} == {"d1", "d2"}
