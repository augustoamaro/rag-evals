from __future__ import annotations

import time
import uuid

from rag.domain.entities import (
    CaseResult,
    EvalCase,
    EvalRun,
    RunMetrics,
    Strategy,
)
from rag.domain.ports import ChunkStore, Retriever
from rag.metrics.relevance import relevant_chunk_ids
from rag.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from rag.metrics.stats import mean, percentile


class EvalService:
    def __init__(self, retriever: Retriever, store: ChunkStore) -> None:
        self._retriever = retriever
        self._store = store

    def run_retrieval(
        self, cases: list[EvalCase], strategy: Strategy, k: int
    ) -> EvalRun:
        chunks = self._store.all_chunks()
        results: list[CaseResult] = []
        for case in cases:
            relevant = relevant_chunk_ids(chunks, case.relevant_snippets)
            started = time.perf_counter()
            retrieved = self._retriever.retrieve(case.question, k, strategy)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            ranked = [rc.chunk.id for rc in retrieved]
            results.append(
                CaseResult(
                    case_id=case.id,
                    strategy=strategy.value,
                    retrieved_ids=ranked,
                    recall=recall_at_k(ranked, relevant, k),
                    precision=precision_at_k(ranked, relevant, k),
                    mrr=mrr(ranked, relevant),
                    ndcg=ndcg_at_k(ranked, relevant, k),
                    latency_ms=elapsed_ms,
                )
            )
        latencies = [float(r.latency_ms) for r in results]
        return EvalRun(
            id=uuid.uuid4().hex,
            strategy=strategy,
            k=k,
            retrieval_metrics=_aggregate(results),
            case_results=results,
            latency_p50_ms=int(percentile(latencies, 50)),
            latency_p95_ms=int(percentile(latencies, 95)),
        )


def _aggregate(results: list[CaseResult]) -> RunMetrics:
    return RunMetrics(
        recall=mean([r.recall for r in results]),
        precision=mean([r.precision for r in results]),
        mrr=mean([r.mrr for r in results]),
        ndcg=mean([r.ndcg for r in results]),
    )
