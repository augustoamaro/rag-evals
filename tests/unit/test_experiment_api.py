from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api.app import AppDeps, create_app
from rag.domain.entities import EvalCase, Strategy
from rag.domain.evaluation import EvaluationBusyError
from tests.unit.test_api import FakeEvalService, FakeEvalStore, FakeRetriever
from tests.unit.test_run_comparison import stored
from tests.unit.test_versioned_eval import fixture

TOKEN = "11c04832-bfe9-4d23-9e68-3995a1f61e89"


class Jobs:
    available = True

    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.busy = False

    def submit(
        self,
        cases: list[EvalCase],
        strategy: Strategy,
        k: int,
        with_answers: bool,
        request_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if self.busy:
            raise EvaluationBusyError("an evaluation is already active")
        job = {
            "id": "job",
            "run_id": None,
            "config": config,
            "status": "queued",
            "total": len(cases),
            "completed": 0,
            "request_id": request_id,
        }
        self.jobs.append(job)
        return job

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.jobs

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return next((job for job in self.jobs if job["id"] == job_id), None)

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        return next((job for job in self.jobs if job["request_id"] == request_id), None)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if job:
            job["status"] = "cancelling"
        return job

    def wait(self, job_id: str) -> dict[str, Any]:
        return {"id": job_id, "run_id": "result", "status": "completed"}


def client(
    *, answers: bool = False, reference: str = "answer", worker: bool = True
) -> tuple[TestClient, Jobs]:
    dataset, _ = fixture()
    dataset = replace(
        dataset,
        cases=[
            replace(dataset.cases[0], reference_answer=reference),
            replace(
                dataset.cases[0],
                id="pt",
                split="dev",
                language="pt",
                reference_answer=reference,
            ),
        ],
    )
    jobs = Jobs()
    deps = AppDeps(
        FakeRetriever(),
        None,
        FakeEvalService(),
        FakeEvalStore(),
        [],
        dataset=dataset,
        answers_available=answers,
        jobs=jobs if worker else None,
        index_id="immutable-index",
    )
    return TestClient(create_app(deps)), jobs


def test_options_describe_exact_selections_and_english_default() -> None:
    api, _ = client()
    result = api.get("/evals/options")
    assert result.status_code == 200
    body = result.json()
    assert body["default_language"] == "en"
    assert body["dataset"]["index_id"] == "immutable-index"
    selections = {(s["split"], s["language"]): s for s in body["selections"]}
    assert set(selections) == {("smoke", "en"), ("dev", "pt")}
    assert selections[("smoke", "en")]["case_count"] == 1
    assert not selections[("smoke", "en")]["answers_available"]
    assert selections[("smoke", "en")]["answer_unavailable_reason"]


def test_launch_poll_and_cancel_with_separate_portuguese_selection() -> None:
    api, jobs = client()
    response = api.post(
        "/evals/jobs", json={"request_id": TOKEN, "split": "dev", "language": "pt"}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["config"]["language"] == "pt" and body["total"] == 1
    assert api.get("/evals/jobs/job").json()["status"] == "queued"
    assert api.get("/evals/jobs").json() == jobs.jobs
    assert api.post("/evals/jobs/job/cancel").json()["status"] == "cancelling"
    assert api.get("/evals/jobs/missing").status_code == 404
    assert api.post("/evals/jobs/missing/cancel").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"split": "missing"},
        {"language": "pt"},
        {"strategy": "invalid"},
        {"with_answers": True},
        {"k": 51},
        {"request_id": "not-uuid"},
    ],
)
def test_invalid_launch_never_reaches_worker(payload: dict[str, Any]) -> None:
    api, jobs = client()
    result = api.post("/evals/jobs", json={"request_id": TOKEN, **payload})
    assert result.status_code in (400, 422)
    assert not jobs.jobs


def test_missing_references_disable_and_reject_answer_track_even_with_key() -> None:
    api, jobs = client(answers=True, reference="")
    assert not api.get("/evals/options").json()["selections"][0]["answers_available"]
    result = api.post("/evals/jobs", json={"request_id": TOKEN, "with_answers": True})
    assert result.status_code == 400 and "reference" in result.json()["detail"]
    assert not jobs.jobs


def test_busy_and_unavailable_are_actionable() -> None:
    api, jobs = client()
    jobs.busy = True
    assert api.post("/evals/jobs", json={"request_id": TOKEN}).status_code == 409
    offline, _ = client(worker=False)
    assert not offline.get("/evals/options").json()["worker_available"]
    assert offline.post("/evals/jobs", json={"request_id": TOKEN}).status_code == 503


def test_synchronous_production_endpoint_uses_worker_admission() -> None:
    api, jobs = client()
    result = api.post("/evals/run", json={})
    assert result.status_code == 200 and result.json() == {"id": "result"}
    assert len(jobs.jobs) == 1
    jobs.busy = True
    assert api.post("/evals/run", json={}).status_code == 409


def test_compare_endpoint_rejects_mismatch_and_missing_runs() -> None:
    class Store(FakeEvalStore):
        def get_run(self, run_id: str) -> dict[str, Any] | None:
            if run_id not in ("en", "pt"):
                return None
            run = stored([1])
            run["id"] = run_id
            if run_id == "pt":
                run["config"]["language"] = "pt"
            return run

    api = TestClient(
        create_app(AppDeps(FakeRetriever(), None, FakeEvalService(), Store(), []))
    )
    assert (
        api.get("/evals/compare?baseline=en&candidate=pt").json()["compatible"] is False
    )
    assert api.get("/evals/compare?baseline=en&candidate=en").json()["compatible"] is True
    assert api.get("/evals/compare?baseline=en&candidate=missing").status_code == 404


@pytest.mark.parametrize(
    "runtime_change",
    ["index", "provider", "selection", "reference", "default_split", "worker"],
)
def test_terminal_request_retry_survives_runtime_changes(runtime_change: str) -> None:
    dataset, _ = fixture()
    jobs = Jobs()
    deps = AppDeps(
        FakeRetriever(),
        None,
        FakeEvalService(),
        FakeEvalStore(),
        [],
        dataset=dataset,
        answers_available=True,
        jobs=jobs,
        index_id="original-index",
    )
    api = TestClient(create_app(deps))
    body = {"request_id": TOKEN, "with_answers": True}
    launched = api.post("/evals/jobs", json=body)
    assert launched.status_code == 202
    jobs.jobs[0].update(status="completed", run_id="original-run")
    jobs.busy = True
    if runtime_change == "index":
        deps.index_id = "new-index"
    elif runtime_change == "provider":
        deps.answers_available = False
    elif runtime_change == "selection":
        deps.dataset = replace(dataset, cases=[])
    elif runtime_change == "reference":
        deps.dataset = replace(
            dataset, cases=[replace(case, reference_answer="") for case in dataset.cases]
        )
    elif runtime_change == "default_split":
        deps.dataset = replace(
            dataset,
            default_split="dev",
            cases=[replace(case, split="dev") for case in dataset.cases],
        )
    else:
        jobs.available = False

    retried = api.post("/evals/jobs", json=body)
    assert retried.status_code == 202
    assert retried.json() == jobs.jobs[0]
    assert retried.json()["config"]["index_id"] == "original-index"
    assert retried.json()["config"]["split"] == "smoke"
    assert len(jobs.jobs) == 1


@pytest.mark.parametrize(
    "changed", [{"k": 5}, {"split": "smoke"}, {"with_answers": True}]
)
def test_used_token_rejects_changed_client_request_before_current_validation(
    changed: dict[str, Any],
) -> None:
    api, jobs = client()
    first = api.post("/evals/jobs", json={"request_id": TOKEN})
    assert first.status_code == 202
    retried = api.post("/evals/jobs", json={"request_id": TOKEN, **changed})
    assert retried.status_code == 409
    assert len(jobs.jobs) == 1


def test_explicit_defaults_retry_the_same_canonical_request() -> None:
    api, jobs = client()
    first = api.post("/evals/jobs", json={"request_id": TOKEN})
    assert first.status_code == 202
    explicit = {
        "request_id": TOKEN,
        "strategy": "hybrid",
        "k": 10,
        "with_answers": False,
        "split": None,
        "language": "en",
    }
    second = api.post("/evals/jobs", json=explicit)
    assert second.status_code == 202
    assert second.json() == first.json()
    assert len(jobs.jobs) == 1
