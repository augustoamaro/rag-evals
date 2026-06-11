from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag.api.schemas import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    RetrievedChunkOut,
    RunIdResponse,
    RunRequest,
)
from rag.domain.entities import Answer, EvalCase, Strategy
from rag.domain.ports import Generator, Retriever


@dataclass
class AppDeps:
    retriever: Retriever
    generator: Generator | None
    eval_service: Any  # EvalService (Any keeps the API layer decoupled from its shape)
    eval_store: Any  # EvalStore
    cases: list[EvalCase]
    answers_available: bool = False


def create_app(deps: AppDeps) -> FastAPI:
    app = FastAPI(title="rag-evals", version="0.0.0")
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
            answer = Answer(text="Set ANTHROPIC_API_KEY to generate grounded answers.")
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
        runs: list[dict[str, Any]] = deps.eval_store.list_runs()
        return runs

    @app.get("/evals/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run: dict[str, Any] | None = deps.eval_store.get_run(run_id)
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
    return create_app(deps)
