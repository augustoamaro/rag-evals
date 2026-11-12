from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import Lifespan

from rag.adapters.datasets.loader import select_cases
from rag.api.experiments import experiment_router, submit_evaluation
from rag.api.schemas import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    RetrievedChunkOut,
    RunIdResponse,
    RunRequest,
)
from rag.domain.annotations import AnnotationStorePort
from rag.domain.datasets import Dataset
from rag.domain.entities import Answer, EvalCase, EvalRun, Strategy
from rag.domain.errors import EvalPreflightError
from rag.domain.evaluation import EvaluationJobsPort
from rag.domain.ports import Generator, Retriever

logger = logging.getLogger(__name__)


class EvalRunner(Protocol):
    """What the API needs from an eval service."""

    def run(
        self, cases: list[EvalCase], strategy: Strategy, k: int, with_answers: bool
    ) -> EvalRun: ...


class RunStore(Protocol):
    """What the API needs from run persistence."""

    def save_run(self, run: EvalRun, config: dict[str, Any]) -> None: ...

    def list_runs(self) -> list[dict[str, Any]]: ...

    def get_run(self, run_id: str) -> dict[str, Any] | None: ...


@dataclass
class AppDeps:
    retriever: Retriever
    generator: Generator | None
    eval_service: EvalRunner
    eval_store: RunStore
    cases: list[EvalCase]
    answers_available: bool = False
    dataset: Dataset | None = None
    index_id: str | None = None
    jobs: EvaluationJobsPort | None = None
    annotations: AnnotationStorePort | None = None


def create_app(deps: AppDeps, lifespan: Lifespan[FastAPI] | None = None) -> FastAPI:
    app = FastAPI(title="rag-evals", version="0.0.0", lifespan=lifespan)

    def select(req: RunRequest) -> list[EvalCase]:
        return _evaluation_cases(deps, req)

    app.include_router(experiment_router(deps, select))
    # Deliberately open: this is a local/demo deployment where the dashboard
    # origin is not fixed. Lock allow_origins down for anything internet-facing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        strategy = _parse_strategy(req.strategy)
        chunks = deps.retriever.retrieve(req.question, req.k, strategy)
        generation_error: str | None = None
        if deps.generator is not None:
            try:
                answer = deps.generator.answer(req.question, chunks)
            except Exception as exc:
                # Retrieval already succeeded, so its result is still returned.
                # Provider messages may contain prompts, keys or other private
                # data: report the error type only.
                generation_error = type(exc).__name__
                logger.warning("answer generation failed (%s)", generation_error)
                answer = Answer(text="")
        else:
            answer = Answer(
                text="Set RAG_ANTHROPIC_API_KEY to generate grounded answers."
            )
        return QueryResponse(
            answer=answer.text,
            citations=[CitationOut(chunk_id=c.chunk_id) for c in answer.citations],
            chunks=[
                RetrievedChunkOut(
                    id=rc.chunk.id,
                    text=rc.chunk.text,
                    score=rc.score,
                    rank=rc.rank,
                    score_type=rc.score_type,
                    score_components=rc.score_components,
                    document_id=rc.chunk.document_id,
                    word_start=rc.chunk.word_start,
                    word_end=rc.chunk.word_end,
                )
                for rc in chunks
            ],
            generation_error=generation_error,
        )

    @app.post("/evals/run", response_model=RunIdResponse)
    def run_eval(req: RunRequest) -> RunIdResponse:
        if deps.jobs is not None:
            job = submit_evaluation(deps, req, str(uuid4()), select)
            result = deps.jobs.wait(job["id"])
            if result.get("run_id") is None:
                raise HTTPException(409, "evaluation failed before producing results")
            return RunIdResponse(id=result["run_id"])
        strategy = _parse_strategy(req.strategy)
        if req.with_answers and not deps.answers_available:
            raise HTTPException(
                status_code=400,
                detail="answer track unavailable — set RAG_ANTHROPIC_API_KEY",
            )
        try:
            cases = _evaluation_cases(deps, req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            run = deps.eval_service.run(cases, strategy, req.k, req.with_answers)
        except EvalPreflightError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        deps.eval_store.save_run(run, run.manifest)
        return RunIdResponse(id=run.id)

    @app.get("/evals/runs")
    def list_runs() -> list[dict[str, Any]]:
        return deps.eval_store.list_runs()

    @app.get("/evals/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = deps.eval_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    return app


def _evaluation_cases(deps: AppDeps, req: RunRequest) -> list[EvalCase]:
    if deps.dataset is not None:
        cases = select_cases(deps.dataset, req.split)
    else:
        cases = [
            case for case in deps.cases if req.split is None or case.split == req.split
        ]
    selected = [case for case in cases if case.language == req.language]
    # Existing injected services may own their own empty/default fixture behavior.
    # Any requested selection or dataset-backed run must have matching cases.
    legacy_empty_default = (
        deps.dataset is None
        and not deps.cases
        and req.split is None
        and req.language == "en"
    )
    if not selected and not legacy_empty_default:
        raise ValueError(
            f"no evaluation cases match split {req.split or 'default'!r} "
            f"and language {req.language!r}"
        )
    return selected


def _parse_strategy(value: str) -> Strategy:
    try:
        return Strategy(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid strategy {value!r}"
        ) from exc


def build_app() -> FastAPI:
    """Compose the app with real adapters from settings (used by uvicorn --factory)."""
    from rag.adapters.db.annotation_store import AnnotationStore
    from rag.adapters.db.eval_store import EvalStore
    from rag.adapters.db.job_store import JobStore
    from rag.adapters.db.migrations import apply_migrations
    from rag.adapters.db.pool import make_pool
    from rag.adapters.evaluation_jobs import EvaluationJobs
    from rag.composition import build_runtime
    from rag.config import get_settings

    settings = get_settings()
    apply_migrations(settings.database_url)
    pool = make_pool(settings.database_url)
    try:
        runtime = build_runtime(settings, pool)
    except Exception:
        pool.close()
        raise
    eval_store = EvalStore(pool)
    jobs = EvaluationJobs(
        runtime.evaluator, eval_store, JobStore(pool, settings.database_url)
    )
    deps = AppDeps(
        retriever=runtime.retriever,
        generator=runtime.generator,
        eval_service=runtime.evaluator,
        eval_store=eval_store,
        cases=runtime.cases,
        answers_available=runtime.generator is not None,
        dataset=runtime.dataset,
        index_id=runtime.index_id,
        jobs=jobs,
        annotations=AnnotationStore(pool),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            jobs.start()
            yield
        finally:
            jobs.close()
            pool.close()

    return create_app(deps, lifespan=lifespan)
