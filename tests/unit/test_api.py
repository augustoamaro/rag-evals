from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api.app import AppDeps, create_app
from rag.domain.datasets import Dataset
from rag.domain.entities import (
    Answer,
    Chunk,
    Citation,
    EvalCase,
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
            manifest={
                "schema_version": 1,
                "strategy": strategy.value,
                "k": k,
                "case_ids": [case.id for case in cases],
            },
        )


class FakeEvalStore:
    def __init__(self) -> None:
        self.saved: list[EvalRun] = []
        self.configs: list[dict[str, Any]] = []

    def save_run(self, run: EvalRun, config: dict[str, Any]) -> None:
        self.saved.append(run)
        self.configs.append(config)

    def list_runs(self) -> list[dict[str, Any]]:
        return [{"id": r.id} for r in self.saved]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        for r in self.saved:
            if r.id == run_id:
                return {"id": r.id, "cases": []}
        return None


def _client(*, with_generator: bool = True) -> TestClient:
    deps = AppDeps(
        retriever=FakeRetriever(),
        generator=FakeGenerator() if with_generator else None,
        eval_service=FakeEvalService(),
        eval_store=FakeEvalStore(),
        cases=[],
        answers_available=with_generator,
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
    resp = _client(with_generator=False).post("/query", json={"question": "q"})
    body = resp.json()
    # Must name the actual variable (RAG_ prefix from pydantic-settings).
    assert "RAG_ANTHROPIC_API_KEY" in body["answer"]
    assert body["chunks"][0]["id"] == "cap:0"


def test_query_keeps_retrieved_chunks_when_generation_fails() -> None:
    class FailingGenerator:
        def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
            raise RuntimeError("provider said: secret prompt and key material")

    deps = AppDeps(
        retriever=FakeRetriever(),
        generator=FailingGenerator(),
        eval_service=FakeEvalService(),
        eval_store=FakeEvalStore(),
        cases=[],
        answers_available=True,
    )
    resp = TestClient(create_app(deps)).post("/query", json={"question": "q"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks"][0]["id"] == "cap:0"
    assert body["answer"] == "" and body["citations"] == []
    # Only the error type is exposed: provider messages may hold private data.
    assert body["generation_error"] == "RuntimeError"
    assert "secret" not in resp.text


def test_query_reports_no_generation_error_when_the_answer_succeeds() -> None:
    body = _client().post("/query", json={"question": "what is CAP?"}).json()
    assert body["answer"] == "grounded answer"
    assert body["generation_error"] is None


def test_query_rejects_invalid_strategy() -> None:
    resp = _client().post("/query", json={"question": "q", "strategy": "bogus"})
    assert resp.status_code == 400


def test_query_rejects_out_of_range_k_and_empty_question() -> None:
    client = _client()
    assert client.post("/query", json={"question": "q", "k": -1}).status_code == 422
    assert client.post("/query", json={"question": "q", "k": 999}).status_code == 422
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_eval_run_with_answers_unavailable_is_rejected() -> None:
    # Without a configured key the answer track cannot run; requesting it must
    # fail loudly instead of silently returning a retrieval-only run.
    resp = _client(with_generator=False).post(
        "/evals/run", json={"strategy": "hybrid", "k": 5, "with_answers": True}
    )
    assert resp.status_code == 400
    assert "answer track" in resp.json()["detail"]


def test_eval_run_with_answers_available_succeeds() -> None:
    resp = _client().post(
        "/evals/run", json={"strategy": "hybrid", "k": 5, "with_answers": True}
    )
    assert resp.status_code == 200


def test_eval_run_and_list_and_detail() -> None:
    client = _client()
    run_id = client.post("/evals/run", json={"strategy": "hybrid", "k": 5}).json()["id"]
    assert run_id == "run-9"
    assert client.get("/evals/runs").json() == [{"id": "run-9"}]
    assert client.get("/evals/runs/run-9").json()["id"] == "run-9"
    assert client.get("/evals/runs/missing").status_code == 404


def test_api_persists_the_service_manifest_without_rebuilding_it() -> None:
    store = FakeEvalStore()
    client = TestClient(
        create_app(
            AppDeps(
                FakeRetriever(),
                None,
                FakeEvalService(),
                store,
                [],
            )
        )
    )
    assert (
        client.post("/evals/run", json={"strategy": "hybrid", "k": 5}).status_code == 200
    )
    assert store.configs == [store.saved[0].manifest]


def test_api_preflight_error_is_actionable_and_does_not_persist() -> None:
    from rag.domain.errors import EvalPreflightError

    class StaleEval(FakeEvalService):
        def run(
            self, cases: Any, strategy: Strategy, k: int, with_answers: bool
        ) -> EvalRun:
            raise EvalPreflightError("index does not match corpus; run rag ingest")

    store = FakeEvalStore()
    client = TestClient(
        create_app(AppDeps(FakeRetriever(), None, StaleEval(), store, []))
    )
    response = client.post("/evals/run", json={"strategy": "hybrid", "k": 5})
    assert response.status_code == 409
    assert "rag ingest" in response.json()["detail"]
    assert store.saved == []


def _case(case_id: str, *, split: str = "smoke", language: str = "en") -> EvalCase:
    return EvalCase(
        id=case_id,
        question="fixture question",
        relevant_snippets=[],
        reference_answer="",
        split=split,
        language=language,
    )


def _selection_client(
    *, dataset_enabled: bool = True, default_split: str = "smoke"
) -> tuple[TestClient, FakeEvalStore]:
    cases = [
        _case("smoke-en"),
        _case("smoke-pt", language="pt"),
        _case("dev-en", split="dev"),
        _case("dev-pt", split="dev", language="pt"),
    ]
    dataset = (
        Dataset(
            name="api-selection",
            version="1",
            documents=[],
            evidence=[],
            cases=cases,
            default_split=default_split,
        )
        if dataset_enabled
        else None
    )
    store = FakeEvalStore()
    deps = AppDeps(
        retriever=FakeRetriever(),
        generator=None,
        eval_service=FakeEvalService(),
        eval_store=store,
        # The full dataset must remain selectable even when Runtime.cases has
        # already been restricted to the default English slice at startup.
        cases=[cases[0]] if dataset_enabled else cases,
        dataset=dataset,
    )
    return TestClient(create_app(deps)), store


@pytest.mark.parametrize(
    ("payload", "expected_ids"),
    [
        ({}, ["smoke-en"]),
        ({"language": "pt"}, ["smoke-pt"]),
        ({"split": "dev"}, ["dev-en"]),
        ({"split": "dev", "language": "pt"}, ["dev-pt"]),
    ],
)
def test_dataset_evaluation_selects_exact_split_and_language(
    payload: dict[str, str], expected_ids: list[str]
) -> None:
    client, store = _selection_client()

    response = client.post("/evals/run", json=payload)

    assert response.status_code == 200
    assert store.configs[0]["case_ids"] == expected_ids


def test_dataset_evaluation_uses_its_declared_default_split() -> None:
    client, store = _selection_client(default_split="dev")

    assert client.post("/evals/run", json={}).status_code == 200
    assert store.configs[0]["case_ids"] == ["dev-en"]


@pytest.mark.parametrize(
    "payload", [{"split": "missing"}, {"split": "test"}, {"language": "de"}]
)
def test_invalid_dataset_selection_is_rejected_without_persisting(
    payload: dict[str, str],
) -> None:
    client, store = _selection_client()

    response = client.post("/evals/run", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]
    assert store.saved == []


@pytest.mark.parametrize("language", ["", " ", "\n\t"])
def test_eval_request_rejects_empty_or_blank_language(language: str) -> None:
    client, store = _selection_client()

    response = client.post("/evals/run", json={"language": language})

    assert response.status_code == 422
    assert store.saved == []


@pytest.mark.parametrize(
    ("payload", "expected_ids"),
    [
        ({}, ["smoke-en", "dev-en"]),
        ({"language": "pt"}, ["smoke-pt", "dev-pt"]),
        ({"split": "dev"}, ["dev-en"]),
        ({"split": "dev", "language": "pt"}, ["dev-pt"]),
    ],
)
def test_legacy_dependency_cases_support_explicit_selection(
    payload: dict[str, str], expected_ids: list[str]
) -> None:
    client, store = _selection_client(dataset_enabled=False)

    response = client.post("/evals/run", json=payload)

    assert response.status_code == 200
    assert store.configs[0]["case_ids"] == expected_ids


@pytest.mark.parametrize("payload", [{"split": "missing"}, {"language": "de"}])
def test_legacy_selection_with_no_matching_cases_is_rejected(
    payload: dict[str, str],
) -> None:
    client, store = _selection_client(dataset_enabled=False)

    assert client.post("/evals/run", json=payload).status_code == 400
    assert store.saved == []
