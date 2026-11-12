from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import psycopg
from psycopg import Connection
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.pool import Pool
from rag.domain.entities import CaseResult, EvalRun
from rag.domain.evaluation import (
    EvaluationBusyError,
    EvaluationRequestConflictError,
    same_evaluation_request,
)

_WORKER_LOCK_ID = 7_243_319_072_026_091_603
_COLUMNS = (
    "id, run_id, request_id, config, status, total, completed, stage, case_id, "
    "error_type, created_at, updated_at"
)
_ACTIVE = frozenset({"queued", "running", "cancelling"})


class JobStore:
    """Persistent admission and checkpoints for one local evaluation worker."""

    def __init__(self, pool: Pool, database_url: str) -> None:
        self._pool = pool
        self._database_url = database_url
        self._owner: Connection[Any] | None = None
        self._owner_mutex = threading.Lock()

    def acquire_owner(self) -> bool:
        """Keep the advisory lock on a dedicated connection, never a pool lease."""
        with self._owner_mutex:
            if self._owner is not None:
                return True
            conn = psycopg.connect(self._database_url, autocommit=True)
            try:
                row = conn.execute(
                    "SELECT pg_try_advisory_lock(%s)", (_WORKER_LOCK_ID,)
                ).fetchone()
                if row is not None and row[0]:
                    self._owner = conn
                    return True
            except Exception:
                conn.close()
                raise
            conn.close()
            return False

    @property
    def owner_available(self) -> bool:
        with self._owner_mutex:
            if self._owner is None:
                return False
            try:
                self._owner.execute("SELECT 1")
            except psycopg.Error:
                self._owner.close()
                self._owner = None
                return False
            return True

    def release_owner(self) -> None:
        with self._owner_mutex:
            if self._owner is not None:
                # Closing the dedicated session releases the advisory lock even
                # when its last operation failed or the network disappeared.
                self._owner.close()
                self._owner = None

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM eval_jobs WHERE request_id = %s",
                (request_id,),
            ).fetchone()
        return _job_row(row) if row is not None else None

    def admit(
        self, request_id: str, config: dict[str, Any], total: int
    ) -> tuple[dict[str, Any], bool]:
        existing = self.get_request(request_id)
        if existing is not None:
            _check_request(existing, config)
            return existing, False
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "INSERT INTO eval_jobs (id, request_id, config, status, total) "
                    "VALUES (%s, %s, %s, 'queued', %s) "
                    f"ON CONFLICT (request_id) DO NOTHING RETURNING {_COLUMNS}",
                    (uuid.uuid4().hex, request_id, Jsonb(config), total),
                ).fetchone()
            if row is not None:
                return _job_row(row), True
        except UniqueViolation:
            # The active-job index may be checked before request_id's index.
            # Once the competing insert commits, the same token still wins
            # idempotently instead of incorrectly reporting a busy worker.
            pass
        existing = self.get_request(request_id)
        if existing is not None:
            _check_request(existing, config)
            return existing, False
        raise EvaluationBusyError("another evaluation is already active")

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM eval_jobs "
                "ORDER BY created_at DESC, id DESC LIMIT 50"
            ).fetchall()
        return [_job_row(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM eval_jobs WHERE id = %s", (job_id,)
            ).fetchone()
        return _job_row(row) if row is not None else None

    def request_cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "UPDATE eval_jobs SET status = 'cancelling', updated_at = now() "
                "WHERE id = %s AND status IN ('queued', 'running') "
                f"RETURNING {_COLUMNS}",
                (job_id,),
            ).fetchone()
        return _job_row(row) if row is not None else self.get_job(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE eval_jobs SET status = CASE WHEN status = 'cancelling' "
                "THEN status ELSE 'running' END, stage = 'preflight', updated_at = now() "
                "WHERE id = %s AND status IN ('queued', 'running', 'cancelling')",
                (job_id,),
            )

    def save_checkpoint(
        self,
        job_id: str,
        run: EvalRun,
        eval_store: EvalStore,
        completed: int,
        stage: str,
        case_id: str | None,
        *,
        terminal: bool = False,
        changed: Sequence[CaseResult] | None = None,
    ) -> dict[str, Any]:
        """Commit the run, case results and progress as one observable checkpoint.

        ``changed`` lists the case results added or replaced since the previous
        committed checkpoint; ``None`` rewrites every case row.
        """
        with self._pool.connection() as conn:
            previous = conn.execute(
                f"SELECT {_COLUMNS} FROM eval_jobs WHERE id = %s FOR UPDATE", (job_id,)
            ).fetchone()
            if previous is None:
                raise LookupError("evaluation job does not exist")
            job = _job_row(previous)
            if job["status"] not in _ACTIVE:
                return job
            status = "cancelling" if job["status"] == "cancelling" else "running"
            if terminal:
                if run.status == "running":
                    raise ValueError("terminal checkpoint requires a terminal run")
                if status == "cancelling" and run.error_type is None:
                    run = replace(run, status="cancelled", retrieval_complete=False)
                status = run.status
                stage, case_id = status, None
            eval_store.save_run(run, job["config"], connection=conn, case_results=changed)
            row = conn.execute(
                "UPDATE eval_jobs SET run_id = %s, status = %s, completed = %s, "
                "stage = %s, case_id = %s, error_type = %s, updated_at = now() "
                f"WHERE id = %s RETURNING {_COLUMNS}",
                (run.id, status, completed, stage, case_id, run.error_type, job_id),
            ).fetchone()
            assert row is not None
        return _job_row(row)

    def fail(self, job_id: str, error_type: str) -> None:
        """Record an error before the first evaluator checkpoint exists."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE eval_jobs SET status = 'failed', stage = 'failed', "
                "error_type = %s, updated_at = now() "
                "WHERE id = %s AND status IN ('queued', 'running', 'cancelling')",
                (error_type, job_id),
            )

    def recover_interrupted(self) -> None:
        """Fail interrupted work after ownership is acquired; never replay calls."""
        if not self.owner_available:
            raise RuntimeError("worker ownership is required for recovery")
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE eval_runs SET evaluation = "
                "COALESCE(evaluation, '{}'::jsonb) || "
                '\'{"status":"failed","retrieval_complete":false,'
                '"error_type":"WorkerInterrupted"}\'::jsonb, '
                "retrieval_complete = false, error_type = 'WorkerInterrupted', "
                "cost_complete = CASE WHEN evaluation->>'with_answers' = 'true' "
                "THEN false ELSE cost_complete END "
                "WHERE evaluation->>'status' = 'running' OR id IN ("
                "SELECT run_id FROM eval_jobs "
                "WHERE status IN ('queued', 'running', 'cancelling'))"
            )
            conn.execute(
                "UPDATE eval_jobs SET status = 'failed', stage = 'failed', "
                "error_type = 'WorkerInterrupted', updated_at = now() "
                "WHERE status IN ('queued', 'running', 'cancelling')"
            )


def _check_request(job: dict[str, Any], config: dict[str, Any]) -> None:
    if not same_evaluation_request(job["config"], config):
        raise EvaluationRequestConflictError(
            "request_id was already used for a different evaluation request"
        )


def _job_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "run_id": row[1],
        "request_id": str(row[2]),
        "config": row[3],
        "status": row[4],
        "total": row[5],
        "completed": row[6],
        "stage": row[7],
        "case_id": row[8],
        "error_type": row[9],
        "created_at": row[10].isoformat(),
        "updated_at": row[11].isoformat(),
    }
