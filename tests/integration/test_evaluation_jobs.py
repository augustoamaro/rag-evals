from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest
from psycopg import Connection
from psycopg_pool import ConnectionPool

from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.job_store import JobStore
from rag.adapters.db.pool import Pool
from rag.adapters.evaluation_jobs import EvaluationJobs
from rag.domain.entities import CaseResult, EvalCase, EvalRun, RunMetrics, Strategy
from rag.domain.errors import EvalPreflightError
from rag.domain.evaluation import (
    EvaluationBusyError,
    EvaluationControl,
    EvaluationRequestConflictError,
)
from rag.services.run_comparison import compare_runs

pytestmark = pytest.mark.integration


def _run(run_id: str = "checkpoint-run") -> EvalRun:
    return EvalRun(
        id=run_id,
        strategy=Strategy.HYBRID,
        k=10,
        retrieval_metrics=RunMetrics(1.0, 1.0, 1.0, 1.0),
        case_results=[CaseResult("q1", "hybrid", ["chunk-1"], 1, 1, 1, 1)],
    )


def test_checkpoint_updates_existing_run_and_case_without_duplicates(pool: Pool) -> None:
    store = EvalStore(pool)
    first = _run()
    store.save_run(first, {"strategy": "hybrid", "k": 10})
    initial = store.get_run(first.id)
    assert initial is not None
    final = replace(
        first,
        cost_usd=0.002,
        case_results=[
            replace(first.case_results[0], answer="retained answer", cost_usd=0.002),
            CaseResult("q2", "hybrid", ["chunk-2"], 1, 1, 1, 1),
        ],
    )
    store.save_run(final, {"strategy": "hybrid", "k": 10})
    store.save_run(final, {"strategy": "hybrid", "k": 10})

    saved = store.get_run(first.id)
    assert saved is not None
    assert saved["created_at"] == initial["created_at"]
    assert saved["cost_usd"] == pytest.approx(0.002)
    assert len(saved["cases"]) == 2
    assert saved["cases"][0]["answer"] == "retained answer"
    assert saved["cases"][0]["cost_usd"] == pytest.approx(0.002)


def _config(*, with_answers: bool = False) -> dict[str, Any]:
    return {
        "strategy": "hybrid",
        "k": 10,
        "with_answers": with_answers,
        "split": "smoke",
        "language": "en",
    }


def _cases() -> list[EvalCase]:
    return [EvalCase("q1", "Question?", ["evidence"], "Reference")]


class BlockingEvaluator:
    def __init__(self, *, error: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.error = error
        self.calls = 0

    def run(
        self,
        cases: list[EvalCase],
        strategy: Strategy,
        k: int,
        with_answers: bool = False,
        *,
        control: EvaluationControl | None = None,
    ) -> EvalRun:
        assert control is not None
        self.calls += 1
        checkpoint = replace(
            _run(control.run_id),
            status="running",
            retrieval_complete=False,
            with_answers=with_answers,
            cost_usd=0.002,
            case_results=[
                replace(_run().case_results[0], answer="retained", cost_usd=0.002)
            ],
        )
        control.checkpoint(
            checkpoint, 0, "judging" if with_answers else "retrieval", "q1"
        )
        self.entered.set()
        assert self.release.wait(5), "test did not release the in-flight call"
        if self.error:
            raise RuntimeError("private provider details must never be persisted")
        return replace(checkpoint, status="completed", retrieval_complete=True)


def _submit(
    worker: EvaluationJobs, *, request_id: str | None = None, with_answers: bool = False
) -> dict[str, Any]:
    return worker.submit(
        _cases(),
        Strategy.HYBRID,
        10,
        with_answers,
        request_id or str(uuid.uuid4()),
        _config(with_answers=with_answers),
    )


def _wait_status(store: JobStore, job_id: str, expected: set[str]) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = store.get_job(job_id)
        assert job is not None
        if job["status"] in expected:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {expected}")


def test_admission_is_idempotent_for_active_and_terminal_requests(
    pool: Pool, database_url: str
) -> None:
    store = JobStore(pool, database_url)
    request_id = str(uuid.uuid4())
    job, created = store.admit(request_id, _config(), 1)
    assert created is True
    duplicate, created = store.admit(request_id, _config(), 1)
    assert created is False
    assert duplicate["id"] == job["id"]
    with pytest.raises(EvaluationRequestConflictError):
        store.admit(request_id, {**_config(), "k": 5}, 1)
    with pytest.raises(EvaluationBusyError):
        store.admit(str(uuid.uuid4()), _config(), 1)
    store.fail(job["id"], "EvalPreflightError")
    retry, created = store.admit(request_id, _config(), 1)
    assert created is False
    assert retry["status"] == "failed"
    assert retry["id"] == job["id"]


@pytest.mark.parametrize("same_token", [False, True])
def test_concurrent_admission_never_starts_two_jobs(
    pool: Pool, database_url: str, same_token: bool
) -> None:
    store = JobStore(pool, database_url)
    barrier = threading.Barrier(2)
    first_token = str(uuid.uuid4())
    tokens = [first_token, first_token if same_token else str(uuid.uuid4())]

    def admit(token: str) -> str:
        barrier.wait(timeout=5)
        try:
            job, _ = store.admit(token, _config(), 1)
        except EvaluationBusyError:
            return "busy"
        return str(job["id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(admit, tokens))
    if same_token:
        assert results[0] == results[1]
        assert "busy" not in results
    else:
        assert results.count("busy") == 1
    assert len(store.list_jobs()) == 1


def test_owner_lock_is_exclusive_and_does_not_occupy_the_pool(
    pool: Pool, database_url: str
) -> None:
    with ConnectionPool(database_url, min_size=1, max_size=1) as tiny_pool:
        first = JobStore(tiny_pool, database_url)
        second = JobStore(pool, database_url)
        try:
            assert first.acquire_owner() is True
            with tiny_pool.connection(timeout=0.5) as conn:
                assert conn.execute("SELECT 1").fetchone() == (1,)
            assert second.acquire_owner() is False
            first.release_owner()
            assert second.acquire_owner() is True
        finally:
            first.release_owner()
            second.release_owner()


def test_worker_persists_progress_and_only_executes_an_idempotent_request_once(
    pool: Pool, database_url: str
) -> None:
    evaluator = BlockingEvaluator()
    runs = EvalStore(pool)
    jobs = JobStore(pool, database_url)
    worker = EvaluationJobs(evaluator, runs, jobs)
    try:
        assert worker.start() is True
        job = _submit(worker)
        assert evaluator.entered.wait(5)
        progress = worker.get_job(job["id"])
        assert progress is not None
        assert progress["status"] == "running"
        assert progress["total"] == 1
        assert progress["completed"] == 0
        assert progress["case_id"] == "q1"
        assert progress["stage"] == "retrieval"
        checkpoint = runs.get_run(progress["run_id"])
        assert checkpoint is not None
        assert checkpoint["evaluation"]["retrieval_complete"] is False
        assert checkpoint["evaluation"]["status"] == "running"
        assert _submit(worker, request_id=job["request_id"])["id"] == job["id"]
        with pytest.raises(EvaluationBusyError):
            _submit(worker)
        evaluator.release.set()
        final = worker.wait(job["id"])
        assert final["status"] == "completed"
        assert final["completed"] == 1
        assert final["stage"] == "completed"
        saved = runs.get_run(final["run_id"])
        assert saved is not None
        assert saved["evaluation"]["retrieval_complete"] is True
        assert len(saved["cases"]) == 1
        assert saved["cases"][0]["answer"] == "retained"
        assert worker.cancel(job["id"])["status"] == "completed"  # type: ignore[index]
        worker.close()
        assert _submit(worker, request_id=job["request_id"])["id"] == job["id"]
        assert evaluator.calls == 1
    finally:
        evaluator.release.set()
        worker.close()


@pytest.mark.parametrize("error", [False, True])
def test_cancellation_and_failure_keep_checkpointed_cost_and_results(
    pool: Pool, database_url: str, error: bool
) -> None:
    evaluator = BlockingEvaluator(error=error)
    runs = EvalStore(pool)
    worker = EvaluationJobs(evaluator, runs, JobStore(pool, database_url))
    try:
        worker.start()
        job = _submit(worker, with_answers=True)
        assert evaluator.entered.wait(5)
        if not error:
            cancelling = worker.cancel(job["id"])
            assert cancelling is not None
            assert cancelling["status"] == "cancelling"
        evaluator.release.set()
        final = worker.wait(job["id"])
        assert final["status"] == ("failed" if error else "cancelled")
        assert final["error_type"] == ("RuntimeError" if error else None)
        saved = runs.get_run(final["run_id"])
        assert saved is not None
        assert saved["evaluation"]["retrieval_complete"] is False
        assert saved["cost_usd"] == pytest.approx(0.002)
        assert saved["cases"][0]["answer"] == "retained"
        assert saved["evaluation"]["status"] == final["status"]
        if error:
            assert saved["cost_complete"] is False
        assert "private provider" not in str(final)
    finally:
        evaluator.release.set()
        worker.close()


def test_shutdown_keeps_owner_lock_until_inflight_work_finishes(
    pool: Pool, database_url: str
) -> None:
    evaluator = BlockingEvaluator()
    jobs = JobStore(pool, database_url)
    worker = EvaluationJobs(evaluator, EvalStore(pool), jobs)
    other = JobStore(pool, database_url)
    worker.start()
    job = _submit(worker)
    assert evaluator.entered.wait(5)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing = executor.submit(worker.close)
            _wait_status(jobs, job["id"], {"cancelling"})
            assert closing.done() is False
            assert other.acquire_owner() is False
            evaluator.release.set()
            closing.result(timeout=5)
        assert other.acquire_owner() is True
        final = jobs.get_job(job["id"])
        assert final is not None
        assert final["status"] == "cancelled"
    finally:
        evaluator.release.set()
        worker.close()
        other.release_owner()


def test_restart_marks_checkpoint_failed_without_replaying_paid_work(
    pool: Pool, database_url: str
) -> None:
    jobs = JobStore(pool, database_url)
    runs = EvalStore(pool)
    job, _ = jobs.admit(str(uuid.uuid4()), _config(with_answers=True), 1)
    checkpoint = replace(
        _run(),
        status="running",
        retrieval_complete=False,
        with_answers=True,
        cost_usd=0.004,
    )
    jobs.save_checkpoint(job["id"], checkpoint, runs, 0, "judging", "q1")
    evaluator = BlockingEvaluator()
    worker = EvaluationJobs(evaluator, runs, jobs)
    try:
        assert worker.start() is True
        recovered = worker.get_job(job["id"])
        assert recovered is not None
        assert recovered["status"] == "failed"
        assert recovered["error_type"] == "WorkerInterrupted"
        saved = runs.get_run(checkpoint.id)
        assert saved is not None
        assert saved["evaluation"]["status"] == "failed"
        assert saved["evaluation"]["retrieval_complete"] is False
        assert saved["cost_complete"] is False
        assert saved["cost_usd"] == pytest.approx(0.004)
        assert len(saved["cases"]) == 1
        assert evaluator.calls == 0
    finally:
        evaluator.release.set()
        worker.close()


def test_checkpoint_transaction_rolls_back_run_and_progress_together(
    pool: Pool, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = JobStore(pool, database_url)
    runs = EvalStore(pool)
    job, _ = jobs.admit(str(uuid.uuid4()), _config(), 1)
    original_save = runs.save_run

    def fail_after_run_write(
        run: EvalRun,
        config: dict[str, Any],
        git_sha: str | None = None,
        *,
        connection: Connection[Any] | None = None,
    ) -> None:
        original_save(run, config, git_sha, connection=connection)
        raise RuntimeError("checkpoint interrupted before job write")

    monkeypatch.setattr(runs, "save_run", fail_after_run_write)
    with pytest.raises(RuntimeError, match="checkpoint interrupted"):
        jobs.save_checkpoint(job["id"], _run(), runs, 1, "case_complete", "q1")
    assert runs.get_run("checkpoint-run") is None
    after = jobs.get_job(job["id"])
    assert after is not None
    assert after["run_id"] is None
    assert after["completed"] == 0
    assert after["status"] == "queued"


def test_preflight_failure_has_no_run_and_does_not_stop_the_worker(
    pool: Pool, database_url: str
) -> None:
    class PreflightEvaluator(BlockingEvaluator):
        def run(
            self,
            cases: list[EvalCase],
            strategy: Strategy,
            k: int,
            with_answers: bool = False,
            *,
            control: EvaluationControl | None = None,
        ) -> EvalRun:
            if k == 10:
                raise EvalPreflightError("private corpus inconsistency")
            return super().run(cases, strategy, k, with_answers, control=control)

    evaluator = PreflightEvaluator()
    evaluator.release.set()
    worker = EvaluationJobs(evaluator, EvalStore(pool), JobStore(pool, database_url))
    try:
        worker.start()
        first = _submit(worker)
        failed = worker.wait(first["id"])
        assert failed["status"] == "failed"
        assert failed["run_id"] is None
        assert failed["error_type"] == "EvalPreflightError"
        assert "private corpus" not in str(failed)
        second = worker.submit(
            _cases(), Strategy.HYBRID, 5, False, str(uuid.uuid4()), {**_config(), "k": 5}
        )
        assert worker.wait(second["id"])["status"] == "completed"
    finally:
        worker.close()


def test_start_failure_rolls_back_ownership_and_close_is_idempotent(
    pool: Pool, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = EvaluationJobs(
        BlockingEvaluator(), EvalStore(pool), JobStore(pool, database_url)
    )
    other = JobStore(pool, database_url)

    def start_failure(thread: threading.Thread) -> None:
        raise RuntimeError("thread cannot start")

    monkeypatch.setattr(threading.Thread, "start", start_failure)
    try:
        with pytest.raises(RuntimeError, match="thread cannot start"):
            worker.start()
        assert other.acquire_owner() is True
        worker.close()
        worker.close()
    finally:
        other.release_owner()


def test_cancel_accepted_before_final_answer_failure_cannot_restore_full_coverage(
    pool: Pool, database_url: str
) -> None:
    jobs = JobStore(pool, database_url)
    runs = EvalStore(pool)
    job, _ = jobs.admit(str(uuid.uuid4()), _config(with_answers=True), 1)
    jobs.request_cancel(job["id"])
    final = replace(
        _run(),
        status="failed",
        retrieval_complete=True,
        with_answers=True,
        answer_attempted=1,
        answer_failed=1,
    )
    stored_job = jobs.save_checkpoint(
        job["id"], final, runs, 1, "completed", None, terminal=True
    )
    assert stored_job["status"] == "cancelled"
    saved = runs.get_run(final.id)
    assert saved is not None
    assert saved["evaluation"]["status"] == "cancelled"
    assert saved["evaluation"]["retrieval_complete"] is False


def test_stored_interruption_with_all_case_rows_still_cannot_compare(pool: Pool) -> None:
    store = EvalStore(pool)
    baseline = replace(
        _run("baseline"),
        manifest={
            "schema_version": 2,
            "dataset_sha256": "dataset",
            "case_set_sha256": "cases",
            "case_count": 1,
            "split": "test",
            "language": "en",
            "k": 10,
            "retrieval_level": "chunk",
            "evidence_metric_version": "span-union-v1",
            "ranking_metric_version": "binary-v1",
            "index_id": "idx-baseline",
        },
        evidence_metrics={"recall": 1.0},
        case_results=[
            replace(
                _run().case_results[0],
                split="test",
                evidence_recall=1.0,
                evidence_ids=["e1"],
                group_id="g1",
            )
        ],
    )
    store.save_run(baseline, {})
    before = store.get_run(baseline.id)
    assert before is not None
    assert compare_runs(before, before)["compatible"] is True
    interrupted = replace(
        baseline,
        id="candidate",
        status="failed",
        retrieval_complete=False,
        error_type="WorkerInterrupted",
    )
    store.save_run(interrupted, {})
    after = store.get_run(interrupted.id)
    assert after is not None
    assert len(after["cases"]) == after["config"]["case_count"]
    report = compare_runs(before, after)
    assert report["compatible"] is False
    assert report["metrics"] == {}
    assert report["cases"] == []
    assert report["failures"] == ["candidate: complete retrieval coverage is required"]


def test_restart_waits_for_the_previous_shutdown_to_release_ownership(
    pool: Pool, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = JobStore(pool, database_url)
    worker = EvaluationJobs(BlockingEvaluator(), EvalStore(pool), jobs)
    releasing = threading.Event()
    allow_release = threading.Event()
    restart_entered = threading.Event()
    restart_returned = threading.Event()
    original_release = jobs.release_owner

    def delayed_release() -> None:
        releasing.set()
        assert allow_release.wait(5)
        original_release()

    def restart() -> bool:
        restart_entered.set()
        try:
            return worker.start()
        finally:
            restart_returned.set()

    monkeypatch.setattr(jobs, "release_owner", delayed_release)
    assert worker.start() is True
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            closing = executor.submit(worker.close)
            try:
                assert releasing.wait(5)
                restarting = executor.submit(restart)
                assert restart_entered.wait(5)
                assert restart_returned.wait(0.1) is False
                allow_release.set()
                closing.result(timeout=5)
                assert restarting.result(timeout=5) is True
                assert worker.available is True
            finally:
                allow_release.set()
    finally:
        allow_release.set()
        worker.close()


def test_shutdown_still_joins_inflight_work_when_cancel_write_fails(
    pool: Pool, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = JobStore(pool, database_url)
    evaluator = BlockingEvaluator()
    worker = EvaluationJobs(evaluator, EvalStore(pool), jobs)
    cancel_attempted = threading.Event()
    close_returned = threading.Event()

    def failing_cancel(job_id: str) -> dict[str, Any] | None:
        cancel_attempted.set()
        raise RuntimeError("cancellation write failed")

    def close() -> None:
        try:
            worker.close()
        finally:
            close_returned.set()

    worker.start()
    job = _submit(worker)
    assert evaluator.entered.wait(5)
    monkeypatch.setattr(jobs, "request_cancel", failing_cancel)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing = executor.submit(close)
            try:
                assert cancel_attempted.wait(5)
                assert close_returned.wait(0.1) is False
                evaluator.release.set()
                closing.result(timeout=5)
            finally:
                evaluator.release.set()
        assert worker.available is False
        assert worker.wait(job["id"])["status"] == "cancelled"
    finally:
        evaluator.release.set()
        monkeypatch.undo()
        worker.close()


def test_admission_uses_client_identity_and_retains_original_runtime_provenance(
    pool: Pool, database_url: str
) -> None:
    jobs = JobStore(pool, database_url)
    request_id = str(uuid.uuid4())
    request = {**_config(), "split": None}
    config = {**_config(), "index_id": "original-index", "request": request}
    original, _ = jobs.admit(request_id, config, 1)
    jobs.fail(original["id"], "WorkerInterrupted")
    changed_runtime = {**config, "index_id": "new-index", "split": "dev"}
    retried, created = jobs.admit(request_id, changed_runtime, 2)
    assert created is False
    assert retried["id"] == original["id"]
    assert retried["config"] == config
    assert retried["total"] == 1
    worker = EvaluationJobs(BlockingEvaluator(), EvalStore(pool), jobs)
    assert worker.get_request(request_id) == retried
    assert (
        worker.submit(_cases(), Strategy.HYBRID, 10, False, request_id, changed_runtime)
        == retried
    )
    with pytest.raises(EvaluationRequestConflictError):
        jobs.admit(request_id, {**config, "request": {**request, "k": 5}}, 1)


def test_racing_retries_share_client_identity_across_runtime_changes(
    pool: Pool, database_url: str
) -> None:
    jobs = JobStore(pool, database_url)
    request_id = str(uuid.uuid4())
    request = {**_config(), "split": None}
    barrier = threading.Barrier(2)

    def admit(index_id: str) -> tuple[dict[str, Any], bool]:
        barrier.wait(timeout=5)
        return jobs.admit(
            request_id, {**_config(), "index_id": index_id, "request": request}, 1
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(admit, ["original-index", "new-index"]))
    assert results[0][0]["id"] == results[1][0]["id"]
    assert results[0][0]["config"] == results[1][0]["config"]
    assert sum(created for _, created in results) == 1
