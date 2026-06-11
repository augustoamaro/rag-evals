from __future__ import annotations

import pytest

from rag.adapters.db.chunk_store import PgChunkStore
from rag.adapters.db.pool import Pool
from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
from rag.domain.entities import Chunk, Document, Strategy, Vector

pytestmark = pytest.mark.integration

# Deterministic embedder: each keyword maps to one dimension of a 384-d one-hot.
VOCAB = {"alpha": 0, "beta": 1, "gamma": 2}


class StubEmbedder:
    @property
    def dim(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for text in texts:
            vec = [0.0] * 384
            for word, dim in VOCAB.items():
                if word in text.lower():
                    vec[dim] = 1.0
            if not any(vec):
                vec[383] = 1.0  # avoid an all-zero vector (cosine undefined)
            out.append(vec)
        return out


class ReverseReranker:
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return list(reversed(chunks))


def _seed(pool: Pool, embedder: StubEmbedder) -> None:
    doc = Document(id="d1", source="s", title="t", content="x")
    texts = {
        "a": "alpha topic on distributed systems",
        "b": "beta topic on consistency models",
        "c": "gamma topic on vector search",
    }
    chunks = [
        Chunk(id=cid, document_id="d1", ordinal=i, text=txt, token_count=len(txt.split()))
        for i, (cid, txt) in enumerate(texts.items())
    ]
    vectors = embedder.embed([c.text for c in chunks])
    PgChunkStore(pool).add_document(doc, list(zip(chunks, vectors, strict=True)))


def test_dense_returns_semantically_closest_first(pool: Pool) -> None:
    embedder = StubEmbedder()
    _seed(pool, embedder)
    retriever = PgVectorRetriever(pool, embedder)
    results = retriever.retrieve("alpha", 3, Strategy.DENSE)
    assert results[0].chunk.id == "a"


def test_sparse_matches_keyword_only(pool: Pool) -> None:
    embedder = StubEmbedder()
    _seed(pool, embedder)
    retriever = PgVectorRetriever(pool, embedder)
    ids = [r.chunk.id for r in retriever.retrieve("beta", 3, Strategy.SPARSE)]
    assert "b" in ids
    assert "a" not in ids  # no full-text match for "beta"


def test_hybrid_includes_keyword_and_semantic(pool: Pool) -> None:
    embedder = StubEmbedder()
    _seed(pool, embedder)
    retriever = PgVectorRetriever(pool, embedder)
    ids = [r.chunk.id for r in retriever.retrieve("alpha", 3, Strategy.HYBRID)]
    assert "a" in ids


def test_rerank_reorders_hybrid_candidates(pool: Pool) -> None:
    embedder = StubEmbedder()
    _seed(pool, embedder)
    base = PgVectorRetriever(pool, embedder)
    reranking = PgVectorRetriever(pool, embedder, reranker=ReverseReranker())
    hybrid = [r.chunk.id for r in base.retrieve("alpha", 3, Strategy.HYBRID)]
    reranked = [r.chunk.id for r in reranking.retrieve("alpha", 3, Strategy.HYBRID_RERANK)]
    assert reranked == list(reversed(hybrid))
