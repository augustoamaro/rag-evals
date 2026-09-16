from __future__ import annotations

import pytest

from rag.chunking.chunker import chunk_document
from rag.domain.entities import Chunk, Document, Vector
from rag.metrics.relevance import relevant_chunk_ids
from rag.services.corpus_loader import load_documents, load_golden
from rag.services.ingestion_service import IngestionService


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[Vector]:
        self.calls.append(texts)
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.docs: list[Document] = []
        self.chunks: list[Chunk] = []
        self.replacements = 0

    def add_document(self, doc: Document, chunks: list[tuple[Chunk, Vector]]) -> None:
        self.docs.append(doc)
        self.chunks.extend(c for c, _ in chunks)

    def replace_corpus(
        self, documents: list[tuple[Document, list[tuple[Chunk, Vector]]]]
    ) -> None:
        self.replacements += 1
        self.docs = [doc for doc, _ in documents]
        self.chunks = [chunk for _, pairs in documents for chunk, _ in pairs]

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
    long_content = " ".join(f"w{i}" for i in range(250))
    docs = [
        Document(id="d1", source="s", title="t", content=long_content),
        Document(id="d2", source="s", title="t", content="short doc"),
    ]
    total = service.ingest(docs)
    assert total == len(store.chunks)
    assert {c.document_id for c in store.chunks} == {"d1", "d2"}
    assert store.replacements == 1


def test_ingestion_rejects_duplicate_document_ids_before_embedding() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()
    service = IngestionService(embedder, store)
    docs = [
        Document(id="duplicate", source="one", title="one", content="first"),
        Document(id="duplicate", source="two", title="two", content="second"),
    ]

    with pytest.raises(ValueError, match="duplicate document id"):
        service.ingest(docs)

    assert embedder.calls == []
    assert store.replacements == 0


def test_ingestion_prepares_every_embedding_before_replacing() -> None:
    class FailingEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[Vector]:
            if texts == ["second"]:
                raise RuntimeError("embedding failed")
            return super().embed(texts)

    store = FakeStore()
    store.docs = [Document(id="old", source="old", title="old", content="old")]
    service = IngestionService(FailingEmbedder(), store)

    with pytest.raises(RuntimeError, match="embedding failed"):
        service.ingest(
            [
                Document(id="first", source="s", title="t", content="first"),
                Document(id="second", source="s", title="t", content="second"),
            ]
        )

    assert [doc.id for doc in store.docs] == ["old"]
    assert store.replacements == 0


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[0.0, 0.0]],
        [[0.0, 0.0, 0.0, float("nan")]],
        [[0.0, 0.0, 0.0, float("inf")]],
    ],
)
def test_ingestion_rejects_invalid_embeddings_before_replacing(
    vectors: list[Vector],
) -> None:
    class InvalidEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[Vector]:
            return vectors

    store = FakeStore()

    with pytest.raises(ValueError, match="embedding"):
        IngestionService(InvalidEmbedder(), store).ingest(
            [Document(id="doc", source="s", title="t", content="content")]
        )

    assert store.replacements == 0
