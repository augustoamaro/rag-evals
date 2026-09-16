from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Queue
from threading import Barrier
from time import monotonic, sleep

import pytest
from psycopg.errors import ForeignKeyViolation, RaiseException

from rag.adapters.db.pool import Pool
from rag.adapters.db.snapshot_store import SnapshotChunkStore, SnapshotStore
from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
from rag.chunking.chunker import chunk_document
from rag.domain.datasets import Dataset, dataset_digest
from rag.domain.entities import Chunk, Document, Strategy, Vector

pytestmark = pytest.mark.integration

EMBEDDING: dict[str, object] = {
    "provider": "test",
    "model": "deterministic",
    "dimension": 384,
    "artifact_sha256": "a" * 64,
}
CHUNKING: dict[str, object] = {
    "algorithm": "word_window_v1",
    "target_words": 3,
    "overlap_words": 1,
}


class KeywordEmbedder:
    @property
    def dim(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> list[Vector]:
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            vector[0 if "alpha" in text else 1] = 1.0
            vectors.append(vector)
        return vectors


def _dataset(content: str = "alpha beta gamma delta", doc_id: str = "doc") -> Dataset:
    return Dataset(
        name="snapshot-fixture",
        version="1",
        documents=[
            Document(id=doc_id, source="fixture", title="Fixture", content=content)
        ],
        evidence=[],
        cases=[],
        provenance={"source": "test-fixture"},
    )


def _prepared(
    dataset: Dataset, *, target_words: int = 3, overlap_words: int = 1
) -> list[tuple[Document, list[tuple[Chunk, Vector]]]]:
    embedder = KeywordEmbedder()
    return [
        (
            doc,
            [
                (chunk, embedder.embed([chunk.text])[0])
                for chunk in chunk_document(
                    doc, target_words=target_words, overlap_words=overlap_words
                )
            ],
        )
        for doc in dataset.documents
    ]


def test_publish_round_trips_dataset_and_preserves_old_index_history(pool: Pool) -> None:
    store = SnapshotStore(pool)
    original_dataset = _dataset()
    original = store.publish(
        original_dataset, _prepared(original_dataset), EMBEDDING, CHUNKING
    )
    changed_dataset = _dataset("beta updated text")
    changed = store.publish(
        changed_dataset, _prepared(changed_dataset), EMBEDDING, CHUNKING
    )

    assert original.id != changed.id
    loaded = store.get(original.id)
    assert loaded is not None
    assert loaded.dataset == original_dataset
    assert loaded.embedding == EMBEDDING
    assert loaded.chunking == CHUNKING
    assert [c.text for c in SnapshotChunkStore(pool, original.id).all_chunks()] == [
        "alpha beta gamma",
        "gamma delta",
    ]
    assert [c.text for c in SnapshotChunkStore(pool, changed.id).all_chunks()] == [
        "beta updated text"
    ]
    assert store.find(dataset_digest(original_dataset), EMBEDDING, CHUNKING) == loaded
    assert store.get("missing") is None
    assert store.find("f" * 64, EMBEDDING, CHUNKING) is None


def test_publication_identity_includes_vectors_chunking_model_and_provenance(
    pool: Pool,
) -> None:
    store = SnapshotStore(pool)
    dataset = _dataset()
    original = store.publish(dataset, _prepared(dataset), EMBEDDING, CHUNKING)
    repeated = store.publish(dataset, _prepared(dataset), dict(EMBEDDING), dict(CHUNKING))
    assert repeated.id == original.id

    different_model = store.publish(
        dataset, _prepared(dataset), {**EMBEDDING, "artifact_sha256": "b" * 64}, CHUNKING
    )
    changed_vectors = _prepared(dataset)
    changed_vectors[0][1][0][1][2] = 0.5
    different_vectors = store.publish(dataset, changed_vectors, EMBEDDING, CHUNKING)
    different_chunks = store.publish(
        dataset,
        _prepared(dataset, target_words=4),
        EMBEDDING,
        {**CHUNKING, "target_words": 4},
    )
    changed_provenance = replace(dataset, provenance={"source": "a different fixture"})
    different_provenance = store.publish(
        changed_provenance, _prepared(changed_provenance), EMBEDDING, CHUNKING
    )

    assert (
        len(
            {
                original.id,
                different_model.id,
                different_vectors.id,
                different_chunks.id,
                different_provenance.id,
            }
        )
        == 5
    )
    # A cache lookup must fail closed when an allegedly identical model/config
    # produced multiple different vector artifacts.
    with pytest.raises(ValueError, match="multiple|ambiguous"):
        store.find(dataset_digest(dataset), EMBEDDING, CHUNKING)


@pytest.mark.parametrize("strategy", list(Strategy))
def test_every_retrieval_arm_is_bound_to_one_snapshot(
    pool: Pool, strategy: Strategy
) -> None:
    class ReverseReranker:
        def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
            return list(reversed(chunks))

    store = SnapshotStore(pool)
    original_dataset = _dataset()
    original = store.publish(
        original_dataset, _prepared(original_dataset), EMBEDDING, CHUNKING
    )
    changed_dataset = _dataset("alpha unrelated replacement", doc_id="doc")
    store.publish(changed_dataset, _prepared(changed_dataset), EMBEDDING, CHUNKING)
    foreign_dataset = _dataset("alpha foreign match", doc_id="foreign")
    store.publish(foreign_dataset, _prepared(foreign_dataset), EMBEDDING, CHUNKING)
    retriever = PgVectorRetriever(
        pool, KeywordEmbedder(), reranker=ReverseReranker(), index_id=original.id
    )

    results = retriever.retrieve("alpha", 10, strategy)

    assert results
    assert {r.chunk.text for r in results} <= {"alpha beta gamma", "gamma delta"}
    assert all(r.chunk.document_id == "doc" for r in results)
    assert all(
        r.chunk.word_start is not None and r.chunk.word_end is not None for r in results
    )
    assert retriever._load(["foreign:0"]) == []


@pytest.mark.parametrize("strategy", [Strategy.DENSE, Strategy.SPARSE, Strategy.HYBRID])
def test_ranking_ties_are_ordered_by_chunk_id(pool: Pool, strategy: Strategy) -> None:
    dataset = _dataset("alpha alpha alpha alpha alpha", doc_id="z")
    dataset = replace(
        dataset, documents=[*dataset.documents, replace(dataset.documents[0], id="a")]
    )
    snapshot = SnapshotStore(pool).publish(
        dataset, _prepared(dataset), EMBEDDING, CHUNKING
    )
    results = PgVectorRetriever(pool, KeywordEmbedder(), index_id=snapshot.id).retrieve(
        "alpha", 4, strategy
    )
    assert [r.chunk.id for r in results] == ["a:0", "a:1", "z:0", "z:1"]


def test_missing_snapshot_and_wrong_embedding_dimension_fail_before_retrieval(
    pool: Pool,
) -> None:
    with pytest.raises(ValueError, match="snapshot|index"):
        SnapshotChunkStore(pool, "missing").all_chunks()
    with pytest.raises(ValueError, match="snapshot|index"):
        PgVectorRetriever(pool, KeywordEmbedder(), index_id="missing")
    dataset = _dataset()
    snapshot = SnapshotStore(pool).publish(
        dataset, _prepared(dataset), EMBEDDING, CHUNKING
    )

    class WrongDimension(KeywordEmbedder):
        @property
        def dim(self) -> int:
            return 768

    with pytest.raises(ValueError, match="dimension"):
        PgVectorRetriever(pool, WrongDimension(), index_id=snapshot.id)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_document",
        "duplicate_document",
        "wrong_document",
        "missing_chunk",
        "wrong_text",
        "wrong_span",
        "wrong_id",
        "wrong_token_count",
        "wrong_ordinal",
        "short_vector",
        "nonfinite_vector",
        "overflow_vector",
        "zero_vector",
    ],
)
def test_invalid_prepared_corpus_never_publishes(pool: Pool, mutation: str) -> None:
    dataset = _dataset()
    prepared = _prepared(dataset)
    chunk, vector = prepared[0][1][0]
    if mutation == "missing_document":
        prepared = []
    elif mutation == "duplicate_document":
        prepared.append(prepared[0])
    elif mutation == "wrong_document":
        prepared[0] = (replace(dataset.documents[0], content="different"), prepared[0][1])
    elif mutation == "missing_chunk":
        prepared[0][1].pop()
    elif mutation == "wrong_text":
        prepared[0][1][0] = (replace(chunk, text="fabricated text"), vector)
    elif mutation == "wrong_span":
        prepared[0][1][0] = (replace(chunk, word_start=1), vector)
    elif mutation == "wrong_id":
        prepared[0][1][0] = (replace(chunk, id="unexpected"), vector)
    elif mutation == "wrong_token_count":
        prepared[0][1][0] = (replace(chunk, token_count=9), vector)
    elif mutation == "wrong_ordinal":
        prepared[0][1][0] = (replace(chunk, ordinal=7), vector)
    elif mutation == "short_vector":
        prepared[0][1][0] = (chunk, [1.0])
    elif mutation == "nonfinite_vector":
        vector[0] = float("nan")
    elif mutation == "overflow_vector":
        vector[0] = 1e100
    elif mutation == "zero_vector":
        prepared[0][1][0] = (chunk, [0.0] * 384)

    with pytest.raises(ValueError):
        SnapshotStore(pool).publish(dataset, prepared, EMBEDDING, CHUNKING)
    with pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM index_snapshots").fetchone() == (0,)
        assert conn.execute("SELECT count(*) FROM snapshot_chunks").fetchone() == (0,)


def test_database_failure_rolls_back_dataset_chunks_and_index(pool: Pool) -> None:
    dataset = _dataset()
    with pool.connection() as conn:
        conn.execute("""
            CREATE FUNCTION fail_snapshot_publication() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'simulated publication failure'; END $$;
            CREATE TRIGGER fail_snapshot_publication BEFORE INSERT ON index_snapshots
            FOR EACH ROW EXECUTE FUNCTION fail_snapshot_publication();
        """)
    try:
        with pytest.raises(RaiseException, match="simulated publication failure"):
            SnapshotStore(pool).publish(dataset, _prepared(dataset), EMBEDDING, CHUNKING)
        with pool.connection() as conn:
            assert conn.execute("SELECT count(*) FROM dataset_snapshots").fetchone() == (
                0,
            )
            assert conn.execute("SELECT count(*) FROM snapshot_chunks").fetchone() == (0,)
            assert conn.execute("SELECT count(*) FROM index_snapshots").fetchone() == (0,)
    finally:
        with pool.connection() as conn:
            conn.execute("DROP TRIGGER fail_snapshot_publication ON index_snapshots")
            conn.execute("DROP FUNCTION fail_snapshot_publication()")


def test_concurrent_identical_publications_reuse_one_complete_index(pool: Pool) -> None:
    dataset = _dataset()
    ready = Barrier(3)

    def publish() -> str:
        ready.wait(timeout=5)
        return (
            SnapshotStore(pool)
            .publish(dataset, _prepared(dataset), EMBEDDING, CHUNKING)
            .id
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(publish) for _ in range(3)]
        ids = [future.result(timeout=10) for future in futures]

    assert len(set(ids)) == 1
    assert [c.text for c in SnapshotChunkStore(pool, ids[0]).all_chunks()] == [
        "alpha beta gamma",
        "gamma delta",
    ]
    with pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM dataset_snapshots").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM index_snapshots").fetchone() == (1,)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE dataset_snapshots SET payload = '{}'::jsonb",
        "DELETE FROM dataset_snapshots",
        "UPDATE index_snapshots SET embedding = '{}'::jsonb",
        "DELETE FROM index_snapshots",
        "UPDATE snapshot_chunks SET text = 'tampered'",
        "DELETE FROM snapshot_chunks",
        "INSERT INTO snapshot_chunks "
        "(index_id, id, document_id, ordinal, text, embedding, "
        "token_count, word_start, word_end) SELECT index_id, 'extra', document_id, 9, "
        "text, embedding, token_count, word_start, word_end FROM snapshot_chunks LIMIT 1",
    ],
)
def test_published_rows_reject_mutation_through_sql(pool: Pool, statement: str) -> None:
    dataset = _dataset()
    snapshot = SnapshotStore(pool).publish(
        dataset, _prepared(dataset), EMBEDDING, CHUNKING
    )
    with (
        pytest.raises(RaiseException, match="immutable|published"),
        pool.connection() as conn,
    ):
        conn.execute(statement)
    assert [c.text for c in SnapshotChunkStore(pool, snapshot.id).all_chunks()] == [
        "alpha beta gamma",
        "gamma delta",
    ]


def test_unsealed_chunks_cannot_commit(pool: Pool) -> None:
    dataset = _dataset()
    snapshot = SnapshotStore(pool).publish(
        dataset, _prepared(dataset), EMBEDDING, CHUNKING
    )
    with pytest.raises(ForeignKeyViolation), pool.connection() as conn:
        conn.execute(
            "INSERT INTO snapshot_chunks (index_id, id, document_id, ordinal, text, "
            "embedding, token_count, word_start, word_end) "
            "SELECT %s, id, document_id, ordinal, text, embedding, token_count, "
            "word_start, word_end FROM snapshot_chunks WHERE index_id = %s",
            ("idx_" + "f" * 64, snapshot.id),
        )
    with pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM snapshot_chunks").fetchone() == (2,)


def test_sql_insert_cannot_race_a_publication_into_an_extra_chunk(pool: Pool) -> None:
    dataset = _dataset()
    snapshot = SnapshotStore(pool).publish(
        dataset, _prepared(dataset), EMBEDDING, CHUNKING
    )
    new_index_id = "idx_" + "f" * 64
    backend_pid: Queue[int] = Queue()

    def append_chunk() -> None:
        with pool.connection() as conn:
            backend_pid.put(conn.info.backend_pid)
            conn.execute(
                "INSERT INTO snapshot_chunks (index_id, id, document_id, ordinal, text, "
                "embedding, token_count, word_start, word_end) "
                "SELECT %s, 'unexpected:0', document_id, 9, text, embedding, "
                "token_count, "
                "word_start, word_end FROM snapshot_chunks WHERE index_id = %s LIMIT 1",
                (new_index_id, snapshot.id),
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pool.connection() as publication:
            publication.execute(
                "INSERT INTO snapshot_chunks (index_id, id, document_id, ordinal, text, "
                "embedding, token_count, word_start, word_end) "
                "SELECT %s, id, document_id, ordinal, text, embedding, token_count, "
                "word_start, word_end FROM snapshot_chunks WHERE index_id = %s",
                (new_index_id, snapshot.id),
            )
            publication.execute(
                "INSERT INTO index_snapshots (id, dataset_sha256, embedding, chunking) "
                "SELECT %s, dataset_sha256, embedding, chunking "
                "FROM index_snapshots WHERE id = %s",
                (new_index_id, snapshot.id),
            )
            competing = executor.submit(append_chunk)
            pid = backend_pid.get(timeout=5)
            deadline = monotonic() + 3
            waiting = None
            while monotonic() < deadline:
                with pool.connection() as observer:
                    waiting = observer.execute(
                        "SELECT wait_event FROM pg_stat_activity WHERE pid = %s", (pid,)
                    ).fetchone()
                if waiting == ("advisory",):
                    break
                sleep(0.01)
            assert waiting == ("advisory",)
        with pytest.raises(RaiseException, match="published"):
            competing.result(timeout=5)
    assert [c.text for c in SnapshotChunkStore(pool, new_index_id).all_chunks()] == [
        "alpha beta gamma",
        "gamma delta",
    ]
