from __future__ import annotations

import typer

from rag.adapters.db.chunk_store import PgChunkStore
from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.migrations import apply_migrations
from rag.adapters.db.pool import make_pool
from rag.adapters.embedding.fastembed_embedder import FastEmbedEmbedder
from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
from rag.config import get_settings
from rag.domain.entities import Strategy
from rag.services.corpus_loader import load_documents, load_golden
from rag.services.eval_service import EvalService
from rag.services.gate import check_thresholds
from rag.services.ingestion_service import IngestionService

app = typer.Typer(help="RAG service + eval harness CLI.", no_args_is_help=True)


@app.command()
def ingest() -> None:
    """Apply migrations and ingest the bundled corpus into Postgres."""
    settings = get_settings()
    apply_migrations(settings.database_url)
    pool = make_pool(settings.database_url)
    try:
        embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
        store = PgChunkStore(pool)
        docs = load_documents()
        count = IngestionService(embedder, store).ingest(docs)
        typer.echo(f"Ingested {count} chunks from {len(docs)} documents.")
    finally:
        pool.close()


@app.command(name="eval")
def run_eval(
    strategy: str = "hybrid",
    k: int = 10,
    gate: bool = False,
    min_recall: float = 0.7,
    min_ndcg: float = 0.6,
) -> None:
    """Run the retrieval eval over the golden set; optionally gate on thresholds."""
    settings = get_settings()
    pool = make_pool(settings.database_url)
    try:
        embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
        retriever = PgVectorRetriever(pool, embedder)
        store = PgChunkStore(pool)
        run = EvalService(retriever, store).run_retrieval(
            load_golden(), Strategy(strategy), k
        )
        EvalStore(pool).save_run(
            run,
            {"strategy": strategy, "k": k, "embedding_model": settings.embedding_model},
        )
        m = run.retrieval_metrics
        typer.echo(
            f"strategy={strategy} k={k}  recall={m.recall:.3f} "
            f"precision={m.precision:.3f} mrr={m.mrr:.3f} ndcg={m.ndcg:.3f}  "
            f"p95={run.latency_p95_ms}ms"
        )
        if gate:
            failures = check_thresholds(m, min_recall, min_ndcg)
            if failures:
                typer.echo("GATE FAILED: " + "; ".join(failures))
                raise typer.Exit(code=1)
            typer.echo("GATE PASSED")
    finally:
        pool.close()


if __name__ == "__main__":
    app()
