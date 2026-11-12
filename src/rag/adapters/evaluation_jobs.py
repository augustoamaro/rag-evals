from __future__ import annotations

import logging
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Protocol

from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.job_store import JobStore
from rag.domain.entities import CaseResult, EvalCase, EvalRun, Strategy
from rag.domain.evaluation import (
    EvaluationControl,
    EvaluationRequestConflictError,
    EvaluationUnavailableError,
    same_evaluation_request,
)

_logger = logging.getLogger(__name__)
_ACTIVE = frozenset({"queued", "running", "cancelling"})


class _Evaluator(Protocol):
    def run(
        self,
        cases: list[EvalCase],
        strategy: Strategy,
        k: int,
        with_answers: bool = False,
        *,
        control: EvaluationControl | None = None,
    ) -> EvalRun: ...


@dataclass(frozen=True)
class _Work:
    job_id: str
    cases: list[EvalCase]
    strategy: Strategy
    k: int
    with_answers: bool


class EvaluationJobs:
    """A bounded local worker; PostgreSQL persists admission and every checkpoint."""

    def __init__(
        self, evaluator: _Evaluator, eval_store: EvalStore, job_store: JobStore
    ) -> None:
        self._evaluator = evaluator
        self._eval_store = eval_store
        self._jobs = job_store
        self._condition = threading.Condition()
        self._lifecycle = threading.Lock()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: _Work | None = None
        self._active_id: str | None = None

    @property
    def available(self) -> bool:
        return (
            not self._shutdown.is_set()
            and self._thread is not None
            and self._thread.is_alive()
            and self._jobs.owner_available
        )

    def start(self) -> bool:
        with self._lifecycle, self._condition:
            if self._thread is not None and self._thread.is_alive():
                return self.available
            if not self._jobs.acquire_owner():
                return False
            try:
                self._jobs.recover_interrupted()
                self._shutdown.clear()
                self._thread = threading.Thread(
                    target=self._work_loop, name="evaluation-worker", daemon=True
                )
                self._thread.start()
            except Exception:
                self._thread = None
                self._jobs.release_owner()
                raise
            return True

    def close(self) -> None:
        """Stop admitting work, finish the current call, then release ownership."""
        # Keep another start from acquiring this session while the previous
        # worker has stopped but its owner connection has not yet been closed.
        with self._lifecycle:
            with self._condition:
                self._shutdown.set()
                active = self._active_id or (
                    self._pending.job_id if self._pending is not None else None
                )
                if active is not None:
                    try:
                        self._jobs.request_cancel(active)
                    except Exception:
                        # The local event still cancels the evaluator. An I/O
                        # failure must not skip join or release ownership early.
                        _logger.exception("Unable to persist shutdown cancellation")
                self._condition.notify_all()
                thread = self._thread
            if thread is not None:
                thread.join()
            self._jobs.release_owner()

    def submit(
        self,
        cases: list[EvalCase],
        strategy: Strategy,
        k: int,
        with_answers: bool,
        request_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        with self._condition:
            existing = self._jobs.get_request(request_id)
            if existing is not None:
                if not same_evaluation_request(existing["config"], config):
                    raise EvaluationRequestConflictError(
                        "request_id was already used for a different evaluation request"
                    )
                return existing
            if not self.available:
                raise EvaluationUnavailableError(
                    "the local evaluation worker is unavailable"
                )
            # Copy inputs before admission: a preparation failure cannot leave
            # a queued record which no thread was given responsibility to run.
            selected = deepcopy(cases)
            job, created = self._jobs.admit(request_id, config, len(selected))
            if created:
                self._pending = _Work(job["id"], selected, strategy, k, with_answers)
                self._condition.notify_all()
            return job

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._jobs.list_jobs()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get_job(job_id)

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        return self._jobs.get_request(request_id)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.request_cancel(job_id)

    def wait(self, job_id: str) -> dict[str, Any]:
        """Wait for the synchronous compatibility endpoint without another queue."""
        while True:
            job = self._jobs.get_job(job_id)
            if job is None:
                raise LookupError("evaluation job does not exist")
            if job["status"] not in _ACTIVE:
                return job
            with self._condition:
                self._condition.wait(timeout=0.1)

    def _work_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._shutdown.is_set()
                )
                if self._pending is None:
                    return
                work, self._pending = self._pending, None
                self._active_id = work.job_id
            try:
                self._execute(work)
            except Exception:
                # If persistence itself is unavailable, leave the last durable
                # checkpoint for restart recovery rather than retrying model calls.
                _logger.exception("Unable to persist evaluation job outcome")
            finally:
                with self._condition:
                    self._active_id = None
                    self._condition.notify_all()

    def _execute(self, work: _Work) -> None:
        control = _Control(self._jobs, self._eval_store, work.job_id, self._shutdown)
        try:
            self._jobs.mark_running(work.job_id)
            run = self._evaluator.run(
                work.cases,
                work.strategy,
                work.k,
                work.with_answers,
                control=control,
            )
            if run.id != control.run_id:
                raise ValueError("evaluation returned a different run ID")
            completed = len(work.cases) if run.retrieval_complete else control.completed
            if self._shutdown.is_set() and run.error_type is None:
                run = replace(run, status="cancelled", retrieval_complete=False)
            self._jobs.save_checkpoint(
                work.job_id,
                run,
                self._eval_store,
                completed,
                run.status,
                None,
                terminal=True,
            )
        except Exception as exc:
            if control.last_run is None:
                self._jobs.fail(work.job_id, type(exc).__name__)
                return
            failed = replace(
                control.last_run,
                status="failed",
                retrieval_complete=False,
                error_type=type(exc).__name__,
                cost_complete=control.last_run.cost_complete and not work.with_answers,
            )
            self._jobs.save_checkpoint(
                work.job_id,
                failed,
                self._eval_store,
                control.completed,
                "failed",
                None,
                terminal=True,
            )


class _Control:
    def __init__(
        self,
        jobs: JobStore,
        runs: EvalStore,
        job_id: str,
        shutdown: threading.Event,
    ) -> None:
        self.run_id = uuid.uuid4().hex
        self.last_run: EvalRun | None = None
        self.completed = 0
        self._persisted: list[CaseResult] = []
        self._jobs = jobs
        self._runs = runs
        self._job_id = job_id
        self._shutdown = shutdown

    def is_cancelled(self) -> bool:
        if self._shutdown.is_set() or not self._jobs.owner_available:
            return True
        job = self._jobs.get_job(self._job_id)
        return job is None or job["status"] in {"cancelling", "cancelled"}

    def checkpoint(
        self, run: EvalRun, completed: int, stage: str, case_id: str | None
    ) -> None:
        self.last_run, self.completed = run, completed
        # Results are immutable and only appended or replaced in place, so
        # identity finds the rows this checkpoint touched. Rewriting every row at
        # every checkpoint made a job's persistence quadratic in its case count.
        changed = [
            result
            for position, result in enumerate(run.case_results)
            if position >= len(self._persisted) or self._persisted[position] is not result
        ]
        self._jobs.save_checkpoint(
            self._job_id, run, self._runs, completed, stage, case_id, changed=changed
        )
        self._persisted = list(run.case_results)
