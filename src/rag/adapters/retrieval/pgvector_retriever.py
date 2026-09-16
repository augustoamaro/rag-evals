from __future__ import annotations

import math

from pgvector import Vector as PgVector

from rag.adapters.db.pool import Pool
from rag.domain.entities import Chunk, RetrievedChunk, Strategy
from rag.domain.ports import Embedder, Reranker, ScoredReranker
from rag.metrics.fusion import rrf_fuse_scored


class PgVectorRetriever:
    """Hybrid retriever over Postgres + pgvector.

    Dense retrieval uses pgvector cosine (`<=>`): exact within an immutable
    snapshot, with HNSW available on the legacy table. Sparse retrieval uses
    Postgres full-text search; hybrid uses RRF and optional cross-encoder reranking.
    """

    def __init__(
        self,
        pool: Pool,
        embedder: Embedder,
        reranker: Reranker | None = None,
        pool_k: int = 50,
        *,
        index_id: str | None = None,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._reranker = reranker
        self._pool_k = pool_k  # candidate depth per arm before fusion / rerank
        self._index_id = index_id
        if index_id is not None:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "SELECT embedding->>'dimension' FROM index_snapshots WHERE id = %s",
                    (index_id,),
                ).fetchone()
            if row is None:
                raise ValueError(f"index snapshot {index_id!r} does not exist")
            if row[0] != str(embedder.dim) or embedder.dim != 384:
                raise ValueError("embedding dimension does not match index snapshot")

    def retrieve(self, query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]:
        if strategy is Strategy.HYBRID_RERANK and self._reranker is None:
            # Never degrade silently: a run labelled hybrid_rerank must have
            # actually reranked, or the persisted metrics lie about the config.
            raise ValueError("hybrid_rerank requested but no reranker is configured")
        scores: dict[str, float | None]
        if strategy is Strategy.DENSE:
            dense = dict(self._dense_scores(query, k))
            chunks = self._load(list(dense))
            scores = dict(dense)
            score_type = "cosine_similarity"
            components = {id_: {"dense": score} for id_, score in dense.items()}
        elif strategy is Strategy.SPARSE:
            sparse = dict(self._sparse_scores(query, k))
            chunks = self._load(list(sparse))
            scores = dict(sparse)
            score_type = "ts_rank_cd"
            components = {id_: {"sparse": score} for id_, score in sparse.items()}
        else:
            dense = dict(self._dense_scores(query, self._pool_k))
            sparse = dict(self._sparse_scores(query, self._pool_k))
            fused = dict(rrf_fuse_scored([list(dense), list(sparse)]))
            scores = dict(fused)
            score_type = "rrf"
            components = {id_: {"rrf": score} for id_, score in fused.items()}
            for id_, values in components.items():
                if id_ in dense:
                    values["dense"] = dense[id_]
                if id_ in sparse:
                    values["sparse"] = sparse[id_]
            if strategy is Strategy.HYBRID_RERANK and self._reranker is not None:
                candidates = self._load(list(fused)[: self._pool_k])
                if isinstance(self._reranker, ScoredReranker):
                    reranked = self._reranker.rerank_with_scores(query, candidates)[:k]
                    chunks = [chunk for chunk, _ in reranked]
                    scores = {chunk.id: score for chunk, score in reranked}
                    score_type = "cross_encoder"
                    for chunk, score in reranked:
                        components[chunk.id]["rerank"] = score
                else:
                    chunks = self._reranker.rerank(query, candidates)[:k]
                    scores = {chunk.id: None for chunk in chunks}
                    score_type = "rank_only"
            else:
                chunks = self._load(list(fused)[:k])
        return [
            RetrievedChunk(
                chunk=c,
                score=scores[c.id],
                rank=i + 1,
                score_type=score_type,
                score_components=components[c.id],
            )
            for i, c in enumerate(chunks)
        ]

    def _dense_scores(self, query: str, k: int) -> list[tuple[str, float]]:
        vec = self._embedder.embed([query])[0]
        if len(vec) != self._embedder.dim or not all(math.isfinite(v) for v in vec):
            raise ValueError("query embedding has invalid dimension or non-finite values")
        if not any(vec):
            raise ValueError("query embedding must not be zero for cosine retrieval")
        with self._pool.connection() as conn:
            # pgvector's HNSW default ef_search is 40; asking for more
            # candidates than that silently truncates the dense arm once the
            # index is actually used. Keep the search width >= the request.
            conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)", (str(max(k, 40)),)
            )
            if self._index_id is None:
                rows = conn.execute(
                    "SELECT id, 1 - (embedding <=> %s) AS score FROM chunks "
                    "ORDER BY embedding <=> %s, id LIMIT %s",
                    (PgVector(vec), PgVector(vec), k),
                ).fetchall()
            else:
                # Filter before ranking. A global ANN index with a post-filter can
                # omit candidates when unrelated historical snapshots dominate it.
                rows = conn.execute(
                    "WITH scoped AS MATERIALIZED (SELECT id, embedding "
                    "FROM snapshot_chunks WHERE index_id = %s) "
                    "SELECT id, 1 - (embedding <=> %s) AS score FROM scoped "
                    "ORDER BY embedding <=> %s, id LIMIT %s",
                    (self._index_id, PgVector(vec), PgVector(vec), k),
                ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def _sparse_scores(self, query: str, k: int) -> list[tuple[str, float]]:
        with self._pool.connection() as conn:
            if self._index_id is None:
                rows = conn.execute(
                    "SELECT id, ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score "
                    "FROM chunks "
                    "WHERE tsv @@ plainto_tsquery('english', %s) "
                    "ORDER BY score DESC, id LIMIT %s",
                    (query, query, k),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score "
                    "FROM snapshot_chunks WHERE index_id = %s "
                    "AND tsv @@ plainto_tsquery('english', %s) "
                    "ORDER BY score DESC, id LIMIT %s",
                    (query, self._index_id, query, k),
                ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def _load(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        with self._pool.connection() as conn:
            if self._index_id is None:
                rows = conn.execute(
                    "SELECT id, document_id, ordinal, text, token_count "
                    "FROM chunks WHERE id = ANY(%s)",
                    (ids,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, document_id, ordinal, text, token_count, word_start, "
                    "word_end FROM snapshot_chunks WHERE index_id = %s AND id = ANY(%s)",
                    (self._index_id, ids),
                ).fetchall()
        by_id = {r[0]: Chunk(*r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]
