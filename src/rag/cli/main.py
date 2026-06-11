from __future__ import annotations

import typer

from rag.adapters.db.chunk_store import PgChunkStore
from rag.adapters.db.migrations import apply_migrations
from rag.adapters.db.pool import make_pool
from rag.adapters.embedding.fastembed_embedder import FastEmbedEmbedder
from rag.config import get_settings
from rag.services.corpus_loader import load_documents
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


if __name__ == "__main__":
    app()
