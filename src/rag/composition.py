from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rag.adapters.datasets.loader import load_dataset, select_cases
from rag.adapters.db.pool import Pool
from rag.adapters.db.snapshot_store import SnapshotChunkStore, SnapshotStore
from rag.adapters.embedding.fastembed_embedder import FastEmbedEmbedder
from rag.adapters.llm.claude_generator import ClaudeGenerator
from rag.adapters.llm.claude_judge import ClaudeJudge
from rag.adapters.provenance import code_provenance
from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
from rag.adapters.retrieval.reranker import FastEmbedReranker
from rag.chunking.chunker import chunk_document
from rag.config import Settings
from rag.domain.datasets import Dataset, dataset_digest
from rag.domain.entities import EvalCase
from rag.domain.errors import EvalPreflightError
from rag.domain.ports import Generator, Retriever
from rag.domain.snapshots import IndexSnapshot
from rag.services.eval_service import EvalService
from rag.services.ingestion_service import prepare_documents


@dataclass(frozen=True)
class Runtime:
    retriever: Retriever
    generator: Generator | None
    evaluator: EvalService
    cases: list[EvalCase]
    dataset: Dataset
    index_id: str


def chunking_config(settings: Settings) -> dict[str, object]:
    return {
        "algorithm": "word_window_v1",
        "target_words": settings.chunk_target_words,
        "overlap_words": settings.chunk_overlap_words,
    }


def ingest_dataset(settings: Settings, pool: Pool, dataset: Dataset) -> IndexSnapshot:
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
    embedding = embedder.identity()
    chunking = chunking_config(settings)
    store = SnapshotStore(pool)
    existing = store.find(dataset_digest(dataset), embedding, chunking)
    if existing is not None:
        return existing
    prepared = prepare_documents(
        embedder,
        dataset.documents,
        target_words=settings.chunk_target_words,
        overlap_words=settings.chunk_overlap_words,
    )
    return store.publish(dataset, prepared, embedding, chunking)


def build_runtime(
    settings: Settings,
    pool: Pool,
    *,
    enable_answers: bool = True,
    dataset_path: Path | None = None,
    index_id: str | None = None,
    split: str | None = None,
    language: str = "en",
) -> Runtime:
    """Bind CLI/API retrieval and evaluation to one immutable index and model."""
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
    embedding = embedder.identity()
    store = SnapshotStore(pool)
    selected_index = index_id or settings.index_id
    if selected_index:
        snapshot = store.get(selected_index)
        if snapshot is None:
            raise EvalPreflightError(f"index snapshot {selected_index!r} does not exist")
        if dataset_path is not None and dataset_digest(
            load_dataset(dataset_path)
        ) != dataset_digest(snapshot.dataset):
            raise EvalPreflightError(
                "selected dataset does not match the requested index"
            )
    else:
        dataset = load_dataset(dataset_path or settings.dataset_path)
        snapshot = store.find(
            dataset_digest(dataset), embedding, chunking_config(settings)
        )
        if snapshot is None:
            raise EvalPreflightError(
                "no index matches this dataset, embedding identity and chunking; "
                "run rag ingest"
            )
    if snapshot.embedding != embedding:
        raise EvalPreflightError(
            "query embedding identity differs from the indexed model; "
            "use the matching runtime or re-ingest"
        )
    cases = [c for c in select_cases(snapshot.dataset, split) if c.language == language]
    if not cases:
        raise EvalPreflightError(
            f"no cases for the selected split and language {language!r}"
        )
    reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"
    pool_k = 50
    retriever = PgVectorRetriever(
        pool,
        embedder,
        reranker=FastEmbedReranker(reranker_model),
        pool_k=pool_k,
        index_id=snapshot.id,
    )
    generator = None
    judge = None
    if enable_answers and settings.anthropic_api_key:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        generator = ClaudeGenerator(client, settings.llm_model)
        judge = ClaudeJudge(client, settings.llm_model)

    expected = [
        chunk
        for doc in snapshot.dataset.documents
        for chunk in chunk_document(
            doc,
            target_words=cast(int, snapshot.chunking["target_words"]),
            overlap_words=cast(int, snapshot.chunking["overlap_words"]),
        )
    ]
    metadata = {
        **code_provenance(),
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "embedding_identity": embedding,
        "index_id": snapshot.id,
        "chunker": snapshot.chunking,
        "retrieval_pool_k": pool_k,
        "rrf_k": 60,
        "reranker_model": reranker_model,
        "generator_prompt_version": "grounded-citations-v2",
        "judge_prompt_version": "structured-scores-v1",
    }
    evaluator = EvalService(
        retriever,
        SnapshotChunkStore(pool, snapshot.id),
        generator,
        judge,
        settings.llm_model,
        expected_chunks=expected,
        runtime=metadata,
        dataset=snapshot.dataset,
    )
    return Runtime(retriever, generator, evaluator, cases, snapshot.dataset, snapshot.id)
