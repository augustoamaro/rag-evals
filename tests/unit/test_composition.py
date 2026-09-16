from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn, cast

import pytest
import yaml

import rag.composition as composition
from rag.adapters.db.pool import Pool
from rag.config import Settings
from rag.domain.datasets import Dataset, EvidenceUnit, dataset_digest, dataset_payload
from rag.domain.entities import (
    Chunk,
    Document,
    EvalCase,
    RetrievedChunk,
    Strategy,
    Vector,
)
from rag.domain.errors import EvalPreflightError
from rag.domain.ports import Embedder, Reranker
from rag.domain.snapshots import IndexSnapshot

_EMBEDDING: dict[str, object] = {
    "model": "test-model",
    "dimension": 384,
    "artifact_sha256": "a" * 64,
}
_CHUNKING: dict[str, object] = {
    "algorithm": "word_window_v1",
    "target_words": 3,
    "overlap_words": 1,
}
type _Prepared = list[tuple[Document, list[tuple[Chunk, Vector]]]]


class LocalEmbedder:
    def __init__(self) -> None:
        self.fingerprint = dict(_EMBEDDING)
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return 384

    def identity(self) -> dict[str, object]:
        return dict(self.fingerprint)

    def embed(self, texts: list[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [[1.0, *([0.0] * 383)] for _ in texts]


class MemorySnapshots:
    def __init__(self, snapshot: IndexSnapshot) -> None:
        self.snapshots = {snapshot.id: snapshot}
        self.published: list[_Prepared] = []

    def get(self, index_id: str) -> IndexSnapshot | None:
        return self.snapshots.get(index_id)

    def find(
        self,
        dataset_sha256: str,
        embedding: dict[str, object],
        chunking: dict[str, object],
    ) -> IndexSnapshot | None:
        return next(
            (
                snapshot
                for snapshot in self.snapshots.values()
                if dataset_digest(snapshot.dataset) == dataset_sha256
                and snapshot.embedding == embedding
                and snapshot.chunking == chunking
            ),
            None,
        )

    def publish(
        self,
        dataset: Dataset,
        prepared: _Prepared,
        embedding: dict[str, object],
        chunking: dict[str, object],
    ) -> IndexSnapshot:
        self.published.append(prepared)
        snapshot = IndexSnapshot("idx_new", dataset, embedding, chunking)
        self.snapshots[snapshot.id] = snapshot
        return snapshot


class LocalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(chunk, 1.0 / rank, rank)
            for rank, chunk in enumerate(self.chunks[:k], 1)
        ]


class LocalChunkReader:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)


@dataclass
class Harness:
    settings: Settings
    pool: Pool
    snapshot: IndexSnapshot
    store: MemorySnapshots
    embedder: LocalEmbedder


def _write_dataset(path: Path, dataset: Dataset) -> None:
    path.write_text(yaml.safe_dump(dataset_payload(dataset)), encoding="utf-8")


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Harness:
    dataset = Dataset(
        name="saved-dataset",
        version="1",
        documents=[
            Document("doc", "fixture.md", "Saved document", "Alpha evidence is retained.")
        ],
        evidence=[EvidenceUnit("evidence", "doc", 0, 2, "Alpha evidence")],
        cases=[
            EvalCase(
                id=f"{split}-{language}",
                question="Where is the saved evidence?",
                relevant_snippets=["Alpha evidence"],
                reference_answer="In the saved document.",
                evidence_ids=["evidence"],
                split=split,
                language=language,
                group_id="evidence-group",
            )
            for split in ("smoke", "dev")
            for language in ("en", "pt")
        ],
    )
    snapshot = IndexSnapshot("idx_saved", dataset, dict(_EMBEDDING), dict(_CHUNKING))
    saved_chunks = [
        Chunk("doc:0", "doc", 0, "Alpha evidence is", 3, 0, 3),
        Chunk("doc:1", "doc", 1, "is retained.", 2, 2, 4),
    ]
    # The fake storage accepts only a correctly bound snapshot ID. Evaluation
    # remains real and checks these hand-written rows against expected chunks.
    chunks_by_index = {snapshot.id: saved_chunks}
    store = MemorySnapshots(snapshot)
    embedder = LocalEmbedder()
    settings = Settings(
        _env_file=None,
        dataset_path=tmp_path / "benchmark.yaml",
        index_id=None,
        embedding_model="test-model",
        embedding_dim=384,
        chunk_target_words=3,
        chunk_overlap_words=1,
        anthropic_api_key=None,
    )
    _write_dataset(settings.dataset_path, dataset)

    def retriever(
        pool: Pool,
        model: Embedder,
        reranker: Reranker | None = None,
        pool_k: int = 50,
        *,
        index_id: str | None = None,
    ) -> LocalRetriever:
        assert index_id is not None, "retrieval must bind to an immutable snapshot"
        return LocalRetriever(chunks_by_index[index_id])

    def reader(pool: Pool, index_id: str) -> LocalChunkReader:
        return LocalChunkReader(chunks_by_index[index_id])

    monkeypatch.setattr(composition, "FastEmbedEmbedder", lambda *args: embedder)
    monkeypatch.setattr(composition, "SnapshotStore", lambda pool: store)
    monkeypatch.setattr(composition, "PgVectorRetriever", retriever)
    monkeypatch.setattr(composition, "SnapshotChunkStore", reader)
    monkeypatch.setattr(composition, "code_provenance", lambda: {})
    return Harness(settings, cast(Pool, object()), snapshot, store, embedder)


def _forbid_model_or_retrieval(*args: object, **kwargs: object) -> NoReturn:
    pytest.fail("preflight must reject before constructing retrieval or the provider SDK")


@pytest.mark.parametrize("local_state", ["changed", "missing"])
@pytest.mark.parametrize("index_from_settings", [False, True])
def test_explicit_index_uses_stored_dataset_and_chunking_after_local_changes(
    harness: Harness, local_state: str, index_from_settings: bool
) -> None:
    if local_state == "missing":
        harness.settings.dataset_path.unlink()
    else:
        changed = replace(
            harness.snapshot.dataset,
            version="2",
            documents=[
                Document("doc", "changed.md", "Changed", "Changed evidence is retained.")
            ],
            evidence=[EvidenceUnit("evidence", "doc", 0, 2, "Changed evidence")],
        )
        _write_dataset(harness.settings.dataset_path, changed)
    settings = harness.settings.model_copy(
        update={
            "index_id": harness.snapshot.id if index_from_settings else None,
            "chunk_target_words": 180,
            "chunk_overlap_words": 30,
        }
    )

    runtime = composition.build_runtime(
        settings,
        harness.pool,
        enable_answers=False,
        index_id=None if index_from_settings else harness.snapshot.id,
    )
    run = runtime.evaluator.run(runtime.cases, Strategy.HYBRID, k=2)

    assert runtime.index_id == "idx_saved"
    assert runtime.dataset.version == "1"
    assert runtime.dataset.documents[0].content == "Alpha evidence is retained."
    assert run.manifest["index_matches_expected"] is True
    assert run.manifest["chunk_count"] == 2
    assert run.manifest["index_id"] == "idx_saved"
    assert run.case_results[0].retrieved_ids == ["doc:0", "doc:1"]
    assert run.case_results[0].evidence_recall == 1.0


def test_embedding_artifact_mismatch_rejects_before_retrieval_and_provider_sdk(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.embedder.fingerprint["artifact_sha256"] = "b" * 64
    settings = harness.settings.model_copy(update={"anthropic_api_key": "fixture-key"})
    monkeypatch.setattr(composition, "PgVectorRetriever", _forbid_model_or_retrieval)
    monkeypatch.setattr(composition, "FastEmbedReranker", _forbid_model_or_retrieval)
    monkeypatch.setattr("anthropic.Anthropic", _forbid_model_or_retrieval)

    with pytest.raises(EvalPreflightError, match="embedding identity differs"):
        composition.build_runtime(settings, harness.pool, index_id="idx_saved")


@pytest.mark.parametrize(
    ("split", "language", "expected_case"),
    [
        (None, None, "smoke-en"),
        (None, "pt", "smoke-pt"),
        ("dev", None, "dev-en"),
        ("dev", "pt", "dev-pt"),
    ],
)
def test_runtime_evaluates_only_the_selected_split_and_language(
    harness: Harness, split: str | None, language: str | None, expected_case: str
) -> None:
    if language is None:
        runtime = composition.build_runtime(
            harness.settings, harness.pool, enable_answers=False, split=split
        )
    else:
        runtime = composition.build_runtime(
            harness.settings,
            harness.pool,
            enable_answers=False,
            split=split,
            language=language,
        )
    run = runtime.evaluator.run(runtime.cases, Strategy.HYBRID, k=2)

    assert [case.id for case in runtime.cases] == [expected_case]
    assert [result.case_id for result in run.case_results] == [expected_case]
    assert run.manifest["split"] == (split or "smoke")
    assert run.manifest["language"] == (language or "en")


@pytest.mark.parametrize("split", ["invalid", "test"])
def test_runtime_rejects_invalid_or_absent_splits_before_retrieval(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    monkeypatch.setattr(composition, "PgVectorRetriever", _forbid_model_or_retrieval)
    with pytest.raises(ValueError, match="split"):
        composition.build_runtime(harness.settings, harness.pool, split=split)


def test_runtime_rejects_absent_language_instead_of_returning_empty_cases(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composition, "PgVectorRetriever", _forbid_model_or_retrieval)
    with pytest.raises(EvalPreflightError, match="language"):
        composition.build_runtime(harness.settings, harness.pool, language="de")


@pytest.mark.parametrize("mismatch", ["missing", "embedding", "chunking", "dataset"])
def test_runtime_without_matching_snapshot_requires_ingestion(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    settings = harness.settings
    if mismatch == "missing":
        harness.store.snapshots.clear()
    elif mismatch == "embedding":
        harness.embedder.fingerprint["artifact_sha256"] = "b" * 64
    elif mismatch == "chunking":
        settings = settings.model_copy(update={"chunk_target_words": 10})
    else:
        _write_dataset(
            settings.dataset_path, replace(harness.snapshot.dataset, version="changed")
        )
    monkeypatch.setattr(composition, "PgVectorRetriever", _forbid_model_or_retrieval)
    monkeypatch.setattr("anthropic.Anthropic", _forbid_model_or_retrieval)

    with pytest.raises(EvalPreflightError, match="rag ingest"):
        composition.build_runtime(settings, harness.pool)


def test_ingest_reuses_matching_snapshot_without_embedding_or_publishing(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness.embedder, "embed", _forbid_model_or_retrieval)
    monkeypatch.setattr(harness.store, "publish", _forbid_model_or_retrieval)

    snapshot = composition.ingest_dataset(
        harness.settings, harness.pool, harness.snapshot.dataset
    )

    assert snapshot.id == "idx_saved"
    assert snapshot.dataset == harness.snapshot.dataset


def test_ingest_prepares_new_content_with_the_selected_chunking(harness: Harness) -> None:
    harness.store.snapshots.clear()

    snapshot = composition.ingest_dataset(
        harness.settings, harness.pool, harness.snapshot.dataset
    )

    assert snapshot.id == "idx_new"
    assert harness.embedder.calls == [["Alpha evidence is", "is retained."]]
    prepared = harness.store.published[0]
    assert [document.id for document, _ in prepared] == ["doc"]
    assert [(chunk.word_start, chunk.word_end) for chunk, _ in prepared[0][1]] == [
        (0, 3),
        (2, 4),
    ]
