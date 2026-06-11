"""End-to-end: ingest -> eval -> persist -> the API serves the results.

Runs the real stack (Postgres + pgvector, fastembed embeddings, hybrid
retrieval, eval service, eval store, FastAPI app) with only the LLM absent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.adapters.db.chunk_store import PgChunkStore
from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.pool import Pool
from rag.adapters.embedding.fastembed_embedder import FastEmbedEmbedder
from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
from rag.api.app import AppDeps, create_app
from rag.domain.entities import Document, EvalCase, Strategy
from rag.services.eval_service import EvalService
from rag.services.ingestion_service import IngestionService

pytestmark = pytest.mark.integration

DOCS = [
    Document(
        id="cap",
        source="cap.md",
        title="CAP",
        content=(
            "The CAP theorem says that during a network partition a system must "
            "choose between consistency and availability."
        ),
    ),
    Document(
        id="rrf",
        source="rrf.md",
        title="RRF",
        content=(
            "Reciprocal rank fusion merges ranked lists by summing one over a "
            "constant plus the rank of each document."
        ),
    ),
]

CASES = [
    EvalCase(
        id="e2e-cap",
        question="What must a system choose between during a partition?",
        relevant_snippets=["between consistency and availability"],
        reference_answer="Consistency and availability.",
    ),
    EvalCase(
        id="e2e-rrf",
        question="How does reciprocal rank fusion merge lists?",
        relevant_snippets=["summing one over a constant plus the rank"],
        reference_answer="By summing 1/(constant + rank) across lists.",
    ),
]


def test_full_pipeline_ingest_eval_persist_serve(pool: Pool) -> None:
    embedder = FastEmbedEmbedder()
    chunk_store = PgChunkStore(pool)
    retriever = PgVectorRetriever(pool, embedder)
    eval_store = EvalStore(pool)

    # Ingest → eval → persist, all through the real services.
    total = IngestionService(embedder, chunk_store).ingest(DOCS)
    assert total >= 2
    service = EvalService(retriever, chunk_store)
    run = service.run(CASES, Strategy.HYBRID, k=3)
    assert run.retrieval_metrics.recall == 1.0  # both snippets retrieved in top-3
    eval_store.save_run(run, {"strategy": "hybrid", "k": 3})

    # The API serves what the pipeline persisted.
    app = create_app(
        AppDeps(
            retriever=retriever,
            generator=None,
            eval_service=service,
            eval_store=eval_store,
            cases=CASES,
        )
    )
    client = TestClient(app)

    runs = client.get("/evals/runs").json()
    assert any(r["id"] == run.id for r in runs)
    detail = client.get(f"/evals/runs/{run.id}").json()
    assert {c["case_id"] for c in detail["cases"]} == {"e2e-cap", "e2e-rrf"}

    query = client.post(
        "/query",
        json={"question": "What does CAP force you to choose between?", "k": 2},
    ).json()
    assert any(chunk["id"].startswith("cap") for chunk in query["chunks"])
