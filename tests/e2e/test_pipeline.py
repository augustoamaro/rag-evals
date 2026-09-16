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
from rag.adapters.llm.claude_generator import ClaudeGenerator
from rag.adapters.llm.fake import FakeAnthropic
from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
from rag.api.app import AppDeps, create_app
from rag.domain.entities import (
    Answer,
    AnswerScores,
    Document,
    EvalCase,
    RetrievedChunk,
    Strategy,
    Usage,
)
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


@pytest.mark.parametrize("one_success", [False, True])
def test_api_judged_failures_round_trip_with_coverage_and_evidence(
    pool: Pool,
    one_success: bool,
) -> None:
    class FaultJudge:
        def score(
            self,
            question: str,
            answer: Answer,
            context: list[RetrievedChunk],
            reference_answer: str,
        ) -> tuple[AnswerScores, Usage]:
            if one_success and question == CASES[0].question:
                return AnswerScores(1, 1, 1, "offline fixture"), Usage(50, 10)
            raise RuntimeError("offline judge failure")

    embedder = FastEmbedEmbedder()
    chunks = PgChunkStore(pool)
    IngestionService(embedder, chunks).ingest(DOCS)
    retriever = PgVectorRetriever(pool, embedder)
    generator = ClaudeGenerator(FakeAnthropic(text="Retained [cap:0]", usage=(100, 20)))
    service = EvalService(
        retriever,
        chunks,
        generator,
        FaultJudge(),
        expected_chunks=chunks.all_chunks(),
    )
    store = EvalStore(pool)
    client = TestClient(
        create_app(
            AppDeps(
                retriever,
                generator,
                service,
                store,
                CASES,
                answers_available=True,
            )
        )
    )
    response = client.post("/evals/run", json={"k": 3, "with_answers": True})
    assert response.status_code == 200
    run_id = response.json()["id"]
    detail = client.get(f"/evals/runs/{run_id}").json()
    scored = int(one_success)
    assert detail["evaluation"] == {
        "with_answers": True,
        "status": "partial" if one_success else "failed",
        "attempted": 2,
        "scored": scored,
        "failed": 2 - scored,
        "retrieval_complete": True,
        "error_type": None,
    }
    assert detail["cost_usd"] == pytest.approx(0.002 + 0.0005 * scored)
    assert detail["cost_complete"] is False
    assert detail["config"]["index_matches_expected"] is True
    assert detail["cases"][1]["answer"] == "Retained [cap:0]"
    assert detail["cases"][1]["judging"]["status"] == "failed"
    assert detail["cases"][1]["retrieved_context"]
    assert detail["cases"][1]["question"] == CASES[1].question
    assert client.get("/evals/runs").json()[0]["evaluation"] == detail["evaluation"]

    chunks.replace_corpus([])
    stale = client.post("/evals/run", json={"k": 3, "with_answers": True})
    assert stale.status_code == 409
    assert len(store.list_runs()) == 1
