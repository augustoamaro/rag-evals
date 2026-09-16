from __future__ import annotations

from typing import Any, Protocol

from rag.domain.entities import EvalCase, EvalRun, Strategy


class EvaluationControl(Protocol):
    """Cooperative cancellation and durable observations, supplied by a runner."""

    @property
    def run_id(self) -> str: ...

    def is_cancelled(self) -> bool: ...

    def checkpoint(
        self, run: EvalRun, completed: int, stage: str, case_id: str | None
    ) -> None: ...


class EvaluationBusyError(RuntimeError):
    """The bounded evaluation worker already has an active job."""


class EvaluationUnavailableError(RuntimeError):
    """This process does not own an available evaluation worker."""


class EvaluationRequestConflictError(ValueError):
    """An idempotency token was reused with a different request."""


def same_evaluation_request(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Match client inputs independently of the runtime selected at admission."""
    original, requested = first.get("request"), second.get("request")
    if isinstance(original, dict) and isinstance(requested, dict):
        return original == requested
    # Older direct worker callers use their complete config as the request.
    return first == second


class EvaluationJobsPort(Protocol):
    @property
    def available(self) -> bool: ...

    def submit(
        self,
        cases: list[EvalCase],
        strategy: Strategy,
        k: int,
        with_answers: bool,
        request_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]: ...

    def list_jobs(self) -> list[dict[str, Any]]: ...

    def get_job(self, job_id: str) -> dict[str, Any] | None: ...

    def get_request(self, request_id: str) -> dict[str, Any] | None: ...

    def cancel(self, job_id: str) -> dict[str, Any] | None: ...

    def wait(self, job_id: str) -> dict[str, Any]: ...
