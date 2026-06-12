from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import Lifespan

from rag.api.schemas import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    RetrievedChunkOut,
    RunIdResponse,
    RunRequest,
)
from rag.domain.entities import Answer, EvalCase, EvalRun, Strategy
from rag.domain.ports import Generator, Retriever


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


def create_app(deps: AppDeps, lifespan: Lifespan[FastAPI] | None = None) -> FastAPI:
    app = FastAPI(title="rag-evals", version="0.0.0", lifespan=lifespan)
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
        if deps.generator is not None:
            answer = deps.generator.answer(req.question, chunks)
        else:
            answer = Answer(
                text="Set RAG_ANTHROPIC_API_KEY to generate grounded answers."
            )
        return QueryResponse(
            answer=answer.text,
            citations=[CitationOut(chunk_id=c.chunk_id) for c in answer.citations],
            chunks=[
                RetrievedChunkOut(
                    id=rc.chunk.id, text=rc.chunk.text, score=rc.score, rank=rc.rank
                )
                for rc in chunks
            ],
        )

    @app.post("/evals/run", response_model=RunIdResponse)
    def run_eval(req: RunRequest) -> RunIdResponse:
        strategy = _parse_strategy(req.strategy)
        if req.with_answers and not deps.answers_available:
            raise HTTPException(
                status_code=400,
                detail="answer track unavailable — set RAG_ANTHROPIC_API_KEY",
            )
        run = deps.eval_service.run(deps.cases, strategy, req.k, req.with_answers)
        deps.eval_store.save_run(run, {"strategy": req.strategy, "k": req.k})
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


def _parse_strategy(value: str) -> Strategy:
    try:
        return Strategy(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid strategy {value!r}"
        ) from exc


def build_app() -> FastAPI:
    """Compose the app with real adapters from settings (used by uvicorn --factory)."""
    from rag.adapters.db.chunk_store import PgChunkStore
    from rag.adapters.db.eval_store import EvalStore
    from rag.adapters.db.pool import make_pool
    from rag.adapters.embedding.fastembed_embedder import FastEmbedEmbedder
    from rag.adapters.llm.claude_generator import ClaudeGenerator
    from rag.adapters.llm.claude_judge import ClaudeJudge
    from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
    from rag.adapters.retrieval.reranker import FastEmbedReranker
    from rag.config import get_settings
    from rag.services.corpus_loader import load_golden
    from rag.services.eval_service import EvalService

    settings = get_settings()
    pool = make_pool(settings.database_url)
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
    retriever = PgVectorRetriever(pool, embedder, reranker=FastEmbedReranker())
    store = PgChunkStore(pool)

    generator: Generator | None = None
    judge = None
    if settings.anthropic_api_key:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        generator = ClaudeGenerator(client, settings.llm_model)
        judge = ClaudeJudge(client, settings.llm_model)

    deps = AppDeps(
        retriever=retriever,
        generator=generator,
        eval_service=EvalService(retriever, store, generator, judge, settings.llm_model),
        eval_store=EvalStore(pool),
        cases=load_golden(),
        answers_available=generator is not None,
    )
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        pool.close()

    return create_app(deps, lifespan=lifespan)
