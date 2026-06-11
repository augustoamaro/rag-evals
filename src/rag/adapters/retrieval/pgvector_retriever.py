from __future__ import annotations

from pgvector import Vector as PgVector

from rag.adapters.db.pool import Pool
from rag.domain.entities import Chunk, RetrievedChunk, Strategy
from rag.domain.ports import Embedder, Reranker
from rag.metrics.fusion import rrf_fuse


class PgVectorRetriever:
    """Hybrid retriever over Postgres + pgvector.

    The only place retrieval SQL lives: dense via pgvector cosine (`<=>`, HNSW),
    sparse via Postgres full-text search, hybrid via Reciprocal Rank Fusion, and
    an optional cross-encoder rerank stage.
    """

    def __init__(
        self,
        pool: Pool,
        embedder: Embedder,
        reranker: Reranker | None = None,
        pool_k: int = 50,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._reranker = reranker
        self._pool_k = pool_k  # candidate depth per arm before fusion / rerank

    def retrieve(self, query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]:
        if strategy is Strategy.HYBRID_RERANK and self._reranker is None:
            # Never degrade silently: a run labelled hybrid_rerank must have
            # actually reranked, or the persisted metrics lie about the config.
            raise ValueError("hybrid_rerank requested but no reranker is configured")
        if strategy is Strategy.DENSE:
            chunks = self._load(self._dense_ids(query, k))
        elif strategy is Strategy.SPARSE:
            chunks = self._load(self._sparse_ids(query, k))
        else:
            dense = self._dense_ids(query, self._pool_k)
            sparse = self._sparse_ids(query, self._pool_k)
            fused = rrf_fuse([dense, sparse])
            if strategy is Strategy.HYBRID_RERANK and self._reranker is not None:
                candidates = self._load(fused[: self._pool_k])
                chunks = self._reranker.rerank(query, candidates)[:k]
            else:
                chunks = self._load(fused[:k])
        return [
            RetrievedChunk(chunk=c, score=1.0 / (i + 1), rank=i + 1)
            for i, c in enumerate(chunks)
        ]

    def _dense_ids(self, query: str, k: int) -> list[str]:
        vec = self._embedder.embed([query])[0]
        with self._pool.connection() as conn:
            # pgvector's HNSW default ef_search is 40; asking for more
            # candidates than that silently truncates the dense arm once the
            # index is actually used. Keep the search width >= the request.
            conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)", (str(max(k, 40)),)
            )
            rows = conn.execute(
                "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (PgVector(vec), k),
            ).fetchall()
        return [r[0] for r in rows]

    def _sparse_ids(self, query: str, k: int) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id FROM chunks "
                "WHERE tsv @@ plainto_tsquery('english', %s) "
                "ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC "
                "LIMIT %s",
                (query, query, k),
            ).fetchall()
        return [r[0] for r in rows]

    def _load(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, document_id, ordinal, text, token_count "
                "FROM chunks WHERE id = ANY(%s)",
                (ids,),
            ).fetchall()
        by_id = {
            r[0]: Chunk(
                id=r[0], document_id=r[1], ordinal=r[2], text=r[3], token_count=r[4]
            )
            for r in rows
        }
        return [by_id[i] for i in ids if i in by_id]
