from __future__ import annotations

from dataclasses import replace
from threading import Event
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from rag.adapters.db.annotation_store import AnnotationStore
from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.job_store import JobStore
from rag.adapters.db.pool import Pool
from rag.adapters.evaluation_jobs import EvaluationJobs
from rag.api.app import AppDeps, create_app
from rag.domain.entities import Answer, RetrievedChunk
from rag.services.eval_service import EvalService
from tests.unit.test_eval_service import StubGenerator, StubJudge
from tests.unit.test_versioned_eval import fixture

pytestmark = pytest.mark.integration


def test_launch_to_compare_and_annotate_real_persisted_runs(
    pool: Pool, database_url: str
) -> None:
    dataset, index = fixture()
    dataset = replace(
        dataset,
        cases=[
            *dataset.cases,
            replace(dataset.cases[0], id="pt", split="dev", language="pt"),
        ],
    )
    service = EvalService(index, index, dataset=dataset)
    store = EvalStore(pool)
    jobs = EvaluationJobs(service, store, JobStore(pool, database_url))
    assert jobs.start()
    api = TestClient(
        create_app(
            AppDeps(
                index,
                None,
                service,
                store,
                dataset.cases,
                dataset=dataset,
                index_id="test-index",
                jobs=jobs,
                annotations=AnnotationStore(pool),
            )
        )
    )
    try:
        run_ids = []
        for selection in (
            {"strategy": "hybrid"},
            {"strategy": "dense"},
            {"split": "dev", "language": "pt"},
        ):
            request = {"request_id": str(uuid4()), "k": 2, **selection}
            response = api.post("/evals/jobs", json=request)
            assert response.status_code == 202
            job = jobs.wait(response.json()["id"])
            assert job["status"] == "completed" and job["completed"] == job["total"] == 1
            run_ids.append(job["run_id"])
            # Reload/retry an ambiguous HTTP response cannot spend/execute again.
            assert api.post("/evals/jobs", json=request).json()["id"] == job["id"]
            detail = api.get(f"/evals/runs/{job['run_id']}").json()
            assert detail["evaluation"]["retrieval_complete"] is True
            assert len(detail["cases"]) == 1
        comparison = api.get(
            f"/evals/compare?baseline={run_ids[0]}&candidate={run_ids[1]}"
        ).json()
        assert comparison["compatible"] and comparison["passed"]
        assert comparison["cases"][0]["metrics"]["evidence_recall"]["delta"] == 0
        mismatch = api.get(
            f"/evals/compare?baseline={run_ids[0]}&candidate={run_ids[2]}"
        ).json()
        assert not mismatch["compatible"] and mismatch["metrics"] == {}
        endpoint = f"/evals/runs/{run_ids[0]}/annotations"
        assert (
            api.put(
                endpoint, json={"note": "Baseline reviewed", "revision": 0}
            ).status_code
            == 200
        )
        assert api.get(endpoint).json()[0]["note"] == "Baseline reviewed"
        assert index.calls == 3
    finally:
        jobs.close()


def test_api_cancellation_preserves_inflight_generation_without_judge(
    pool: Pool, database_url: str
) -> None:
    dataset, index = fixture()
    dataset = replace(dataset, cases=[*dataset.cases, replace(dataset.cases[0], id="q2")])
    entered, release = Event(), Event()

    class BlockingGenerator(StubGenerator):
        def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
            entered.set()
            assert release.wait(timeout=5)
            return super().answer(question, context)

    generator = BlockingGenerator()
    service = EvalService(index, index, generator, StubJudge(), dataset=dataset)
    store = EvalStore(pool)
    jobs = EvaluationJobs(service, store, JobStore(pool, database_url))
    assert jobs.start()
    api = TestClient(
        create_app(
            AppDeps(
                index,
                generator,
                service,
                store,
                dataset.cases,
                answers_available=True,
                dataset=dataset,
                jobs=jobs,
            )
        )
    )
    try:
        launched = api.post(
            "/evals/jobs", json={"request_id": str(uuid4()), "k": 2, "with_answers": True}
        )
        assert launched.status_code == 202 and entered.wait(timeout=5)
        job_id = launched.json()["id"]
        progress = api.get(f"/evals/jobs/{job_id}").json()
        assert progress["stage"] == "generation" and progress["completed"] == 0
        assert progress["run_id"] is not None
        assert api.post(f"/evals/jobs/{job_id}/cancel").json()["status"] == "cancelling"
        release.set()
        terminal = jobs.wait(job_id)
        assert terminal["status"] == "cancelled"
        result = api.get(f"/evals/runs/{terminal['run_id']}").json()
        assert result["evaluation"]["retrieval_complete"] is False
        assert result["cases"][0]["answer"] == "grounded answer"
        assert result["cases"][0]["judging"]["status"] == "skipped"
        assert result["cost_usd"] == pytest.approx(0.001) and result["cost_complete"]
        assert index.calls == 1
        assert api.post(f"/evals/jobs/{job_id}/cancel").json()["status"] == "cancelled"
        comparison = api.get(
            f"/evals/compare?baseline={terminal['run_id']}&candidate={terminal['run_id']}"
        ).json()
        assert not comparison["compatible"]
    finally:
        release.set()
        jobs.close()
