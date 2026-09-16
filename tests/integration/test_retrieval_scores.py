from __future__ import annotations

import pytest

from rag.adapters.db.chunk_store import PgChunkStore
from rag.adapters.db.pool import Pool
from rag.adapters.db.snapshot_store import SnapshotStore
from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
from rag.chunking.chunker import chunk_document
from rag.domain.datasets import Dataset
from rag.domain.entities import Chunk, Document, Strategy, Vector

pytestmark = pytest.mark.integration


class QueryEmbedder:
    @property
    def dim(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> list[Vector]:
        return [[1.0, 0.0, *([0.0] * 382)] for _ in texts]


@pytest.fixture(params=["legacy", "snapshot"])
def score_index(pool: Pool, request: pytest.FixtureRequest) -> str | None:
    # Query = (1, 0). Cosines are a=.6, b=1, c=0, d=e=-1. Each occurrence
    # of the single search term contributes the default ts_rank_cd weight .1.
    rows = [
        ("a", "alpha alpha", [3.0, 4.0]),
        ("b", "alpha", [1.0, 0.0]),
        ("c", "beta", [0.0, 1.0]),
        ("d", "alpha", [-1.0, 0.0]),
        ("e", "beta beta", [-1.0, 0.0]),
    ]
    documents = [Document(id_, "fixture", id_, text) for id_, text, _ in rows]
    prepared = [
        (
            doc,
            [
                (chunk, [*vector, *([0.0] * 382)])
                for chunk in chunk_document(doc, target_words=10, overlap_words=0)
            ],
        )
        for doc, (_, _, vector) in zip(documents, rows, strict=True)
    ]
    if request.param == "legacy":
        PgChunkStore(pool).replace_corpus(prepared)
        return None
    dataset = Dataset("scores", "1", documents, [], [])
    snapshot = SnapshotStore(pool).publish(
        dataset,
        prepared,
        {"provider": "fixture", "dimension": 384},
        {"algorithm": "word_window_v1", "target_words": 10, "overlap_words": 0},
    )
    return snapshot.id


def test_dense_reports_cosine_similarity_including_zero_and_negative_scores(
    pool: Pool, score_index: str | None
) -> None:
    results = PgVectorRetriever(pool, QueryEmbedder(), index_id=score_index).retrieve(
        "alpha", 5, Strategy.DENSE
    )

    assert [r.chunk.id for r in results] == ["b:0", "a:0", "c:0", "d:0", "e:0"]
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert [r.score for r in results] == pytest.approx([1.0, 0.6, 0.0, -1.0, -1.0])
    assert all(r.score_type == "cosine_similarity" for r in results)
    for result, expected in zip(results, [1.0, 0.6, 0.0, -1.0, -1.0], strict=True):
        assert result.score_components == pytest.approx({"dense": expected})


def test_sparse_reports_term_rank_and_keeps_nonmatches_out(
    pool: Pool, score_index: str | None
) -> None:
    results = PgVectorRetriever(pool, QueryEmbedder(), index_id=score_index).retrieve(
        "alpha", 5, Strategy.SPARSE
    )

    assert [r.chunk.id for r in results] == ["a:0", "b:0", "d:0"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.score for r in results] == pytest.approx([0.2, 0.1, 0.1])
    assert all(r.score_type == "ts_rank_cd" for r in results)
    for result, expected in zip(results, [0.2, 0.1, 0.1], strict=True):
        assert result.score_components == pytest.approx({"sparse": expected})


def test_hybrid_reports_rrf_from_arm_ranks_and_preserves_ties(
    pool: Pool, score_index: str | None
) -> None:
    results = PgVectorRetriever(pool, QueryEmbedder(), index_id=score_index).retrieve(
        "alpha", 5, Strategy.HYBRID
    )

    # a=(dense 2,sparse 1), b=(1,2), d=(4,3), c=(3,absent), e=(5,absent).
    assert [r.chunk.id for r in results] == ["a:0", "b:0", "d:0", "c:0", "e:0"]
    expected = [
        {"dense": 0.6, "sparse": 0.2, "rrf": 1 / 62 + 1 / 61},
        {"dense": 1.0, "sparse": 0.1, "rrf": 1 / 61 + 1 / 62},
        {"dense": -1.0, "sparse": 0.1, "rrf": 1 / 64 + 1 / 63},
        {"dense": 0.0, "rrf": 1 / 63},
        {"dense": -1.0, "rrf": 1 / 65},
    ]
    for result, components in zip(results, expected, strict=True):
        assert result.score_type == "rrf"
        assert result.score == pytest.approx(components["rrf"])
        assert result.score_components == pytest.approx(components)


class LegacyReverseReranker:
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return list(reversed(chunks))


def test_legacy_reranker_is_explicitly_rank_only_and_keeps_arm_components(
    pool: Pool, score_index: str | None
) -> None:
    retriever = PgVectorRetriever(
        pool, QueryEmbedder(), LegacyReverseReranker(), pool_k=3, index_id=score_index
    )

    results = retriever.retrieve("alpha", 5, Strategy.HYBRID_RERANK)

    # c and d each have one rank-3 contribution; c wins the ID tie at the cutoff.
    assert [r.chunk.id for r in results] == ["c:0", "b:0", "a:0"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert all(r.score is None and r.score_type == "rank_only" for r in results)
    assert all("rerank" not in r.score_components for r in results)
    assert results[0].score_components == pytest.approx({"dense": 0.0, "rrf": 1 / 63})
    assert results[1].score_components == pytest.approx(
        {"dense": 1.0, "sparse": 0.1, "rrf": 1 / 61 + 1 / 62}
    )


class ScoringReranker:
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        raise AssertionError("the scored interface must be preferred")

    def rerank_with_scores(
        self, query: str, chunks: list[Chunk]
    ) -> list[tuple[Chunk, float]]:
        values = {"a:0": -2.0, "b:0": -2.0, "c:0": 9.5}
        return sorted([(c, values[c.id]) for c in chunks], key=lambda row: -row[1])


def test_scored_reranker_keeps_real_scores_and_only_reranks_bounded_candidates(
    pool: Pool, score_index: str | None
) -> None:
    retriever = PgVectorRetriever(
        pool, QueryEmbedder(), ScoringReranker(), pool_k=3, index_id=score_index
    )

    results = retriever.retrieve("alpha", 2, Strategy.HYBRID_RERANK)

    assert [r.chunk.id for r in results] == ["c:0", "a:0"]
    assert [r.rank for r in results] == [1, 2]
    assert [r.score for r in results] == [9.5, -2.0]
    assert all(r.score_type == "cross_encoder" for r in results)
    assert results[0].score_components == pytest.approx(
        {"dense": 0.0, "rrf": 1 / 63, "rerank": 9.5}
    )
    assert results[1].score_components == pytest.approx(
        {"dense": 0.6, "sparse": 0.2, "rrf": 1 / 62 + 1 / 61, "rerank": -2.0}
    )


def test_hybrid_components_omit_scores_for_arms_outside_the_candidate_depth(
    pool: Pool, score_index: str | None
) -> None:
    results = PgVectorRetriever(
        pool, QueryEmbedder(), pool_k=3, index_id=score_index
    ).retrieve("alpha", 5, Strategy.HYBRID)

    assert [r.chunk.id for r in results] == ["a:0", "b:0", "c:0", "d:0"]
    assert results[2].score_components == pytest.approx({"dense": 0.0, "rrf": 1 / 63})
    assert results[3].score_components == pytest.approx({"sparse": 0.1, "rrf": 1 / 63})
