from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from rag.api.app import AppDeps, create_app
from rag.domain.entities import (
    Answer,
    Chunk,
    Citation,
    EvalRun,
    RetrievedChunk,
    RunMetrics,
    Strategy,
)


def _chunks() -> list[RetrievedChunk]:
    c = Chunk(id="cap:0", document_id="cap", ordinal=0, text="CAP says...", token_count=2)
    return [RetrievedChunk(chunk=c, score=0.9, rank=1)]


class FakeRetriever:
    def retrieve(self, query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]:
        return _chunks()


class FakeGenerator:
    def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
        return Answer(text="grounded answer", citations=[Citation("cap:0")])


class FakeEvalService:
    def run(self, cases: Any, strategy: Strategy, k: int, with_answers: bool) -> EvalRun:
        return EvalRun(
            id="run-9",
            strategy=strategy,
            k=k,
            retrieval_metrics=RunMetrics(0.8, 0.5, 0.7, 0.75),
            case_results=[],
        )


class FakeEvalStore:
    def __init__(self) -> None:
        self.saved: list[EvalRun] = []

    def save_run(self, run: EvalRun, config: dict[str, Any]) -> None:
        self.saved.append(run)

    def list_runs(self) -> list[dict[str, Any]]:
        return [{"id": r.id} for r in self.saved]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        for r in self.saved:
            if r.id == run_id:
                return {"id": r.id, "cases": []}
        return None


def _client(generator: FakeGenerator | None = FakeGenerator()) -> TestClient:
    deps = AppDeps(
        retriever=FakeRetriever(),
        generator=generator,
        eval_service=FakeEvalService(),
        eval_store=FakeEvalStore(),
        cases=[],
    )
    return TestClient(create_app(deps))


def test_healthz() -> None:
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_query_returns_answer_and_chunks() -> None:
    resp = _client().post("/query", json={"question": "what is CAP?", "k": 5})
    body = resp.json()
    assert body["answer"] == "grounded answer"
    assert body["citations"] == [{"chunk_id": "cap:0"}]
    assert body["chunks"][0]["id"] == "cap:0"


def test_query_without_generator_still_returns_chunks() -> None:
    resp = _client(generator=None).post("/query", json={"question": "q"})
    body = resp.json()
    assert "ANTHROPIC_API_KEY" in body["answer"]
    assert body["chunks"][0]["id"] == "cap:0"


def test_query_rejects_invalid_strategy() -> None:
    resp = _client().post("/query", json={"question": "q", "strategy": "bogus"})
    assert resp.status_code == 400


def test_eval_run_and_list_and_detail() -> None:
    client = _client()
    run_id = client.post("/evals/run", json={"strategy": "hybrid", "k": 5}).json()["id"]
    assert run_id == "run-9"
    assert client.get("/evals/runs").json() == [{"id": "run-9"}]
    assert client.get("/evals/runs/run-9").json()["id"] == "run-9"
    assert client.get("/evals/runs/missing").status_code == 404
