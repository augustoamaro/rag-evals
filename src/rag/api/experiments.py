from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException

from rag.api.schemas import AnnotationRequest, JobRequest, RunRequest
from rag.domain.annotations import AnnotationConflictError
from rag.domain.entities import EvalCase, Strategy
from rag.domain.evaluation import (
    EvaluationBusyError,
    EvaluationJobsPort,
    EvaluationRequestConflictError,
    EvaluationUnavailableError,
    same_evaluation_request,
)
from rag.services.run_comparison import compare_runs

if TYPE_CHECKING:
    from rag.api.app import AppDeps


def _worker(deps: AppDeps) -> EvaluationJobsPort:
    if deps.jobs is None:
        raise HTTPException(503, "evaluation worker is unavailable")
    return deps.jobs


def submit_evaluation(
    deps: AppDeps,
    req: RunRequest,
    request_id: str,
    select: Callable[[RunRequest], list[EvalCase]],
) -> dict[str, Any]:
    worker = _worker(deps)
    request = req.model_dump(exclude={"request_id"})
    previous = worker.get_request(request_id)
    if previous is not None:
        if not same_evaluation_request(previous["config"], {"request": request}):
            raise HTTPException(
                409, "request_id was already used for a different evaluation request"
            )
        return previous
    try:
        strategy = Strategy(req.strategy)
    except ValueError as exc:
        raise HTTPException(400, "invalid retrieval strategy") from exc
    try:
        cases = select(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not cases:
        raise HTTPException(400, "no evaluation cases match this selection")
    reason = _answer_unavailable(deps, cases)
    if req.with_answers and reason:
        raise HTTPException(400, reason)
    config = {
        "strategy": strategy.value,
        "k": req.k,
        "with_answers": req.with_answers,
        "split": cases[0].split,
        "language": req.language,
        "index_id": deps.index_id,
        "request": request,
    }
    try:
        return worker.submit(cases, strategy, req.k, req.with_answers, request_id, config)
    except (EvaluationBusyError, EvaluationRequestConflictError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except EvaluationUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc


def _answer_unavailable(deps: AppDeps, cases: list[EvalCase]) -> str | None:
    if any(not c.reference_answer.strip() for c in cases):
        return "answer evaluation requires reference answers for every selected case"
    if not deps.answers_available:
        return "answer track unavailable — set RAG_ANTHROPIC_API_KEY"
    return None


def experiment_router(
    deps: AppDeps, select: Callable[[RunRequest], list[EvalCase]]
) -> APIRouter:
    router = APIRouter(prefix="/evals")

    @router.get("/options")
    def options() -> dict[str, Any]:
        dataset = deps.dataset
        cases = dataset.cases if dataset else deps.cases
        selections = []
        for split, language in sorted({(c.split, c.language) for c in cases}):
            selected = [c for c in cases if (c.split, c.language) == (split, language)]
            reason = _answer_unavailable(deps, selected)
            selections.append(
                {
                    "split": split,
                    "language": language,
                    "case_count": len(selected),
                    "answers_available": reason is None,
                    "answer_unavailable_reason": reason,
                }
            )
        return {
            "dataset": {
                "name": dataset.name,
                "version": dataset.version,
                "index_id": deps.index_id,
            }
            if dataset
            else None,
            "default_split": dataset.default_split if dataset else "smoke",
            "default_language": "en",
            "strategies": [s.value for s in Strategy],
            "selections": selections,
            "worker_available": deps.jobs is not None and deps.jobs.available,
        }

    @router.post("/jobs", status_code=202)
    def launch(req: JobRequest) -> dict[str, Any]:
        return submit_evaluation(deps, req, str(req.request_id), select)

    @router.get("/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return _worker(deps).list_jobs()

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = _worker(deps).get_job(job_id)
        if job is None:
            raise HTTPException(404, "evaluation job not found")
        return job

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        job = _worker(deps).cancel(job_id)
        if job is None:
            raise HTTPException(404, "evaluation job not found")
        return job

    @router.get("/compare")
    def compare(baseline: str, candidate: str) -> dict[str, Any]:
        old, new = deps.eval_store.get_run(baseline), deps.eval_store.get_run(candidate)
        if old is None or new is None:
            raise HTTPException(404, "evaluation run not found")
        return compare_runs(old, new)

    @router.get("/runs/{run_id}/annotations")
    def annotations(run_id: str) -> list[dict[str, Any]]:
        if deps.annotations is None:
            raise HTTPException(503, "annotations are unavailable")
        try:
            return deps.annotations.list_annotations(run_id)
        except KeyError as exc:
            raise HTTPException(404, "evaluation run not found") from exc

    @router.put("/runs/{run_id}/annotations")
    def save_annotation(run_id: str, req: AnnotationRequest) -> dict[str, Any]:
        if deps.annotations is None:
            raise HTTPException(503, "annotations are unavailable")
        try:
            return deps.annotations.save_annotation(
                run_id, req.case_id, req.note, req.revision
            )
        except KeyError as exc:
            raise HTTPException(404, "evaluation run or case not found") from exc
        except AnnotationConflictError as exc:
            raise HTTPException(
                409, "note changed; reload its latest revision before saving"
            ) from exc

    return router
