from __future__ import annotations

import logging
import math
import time
import uuid

from rag.adapters.llm.pricing import cost_usd
from rag.domain.entities import (
    CaseResult,
    EvalCase,
    EvalRun,
    RunMetrics,
    Strategy,
)
from rag.domain.ports import ChunkStore, Generator, Judge, Retriever
from rag.metrics.relevance import relevant_chunk_ids
from rag.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from rag.metrics.stats import mean, percentile

logger = logging.getLogger(__name__)


class EvalService:
    def __init__(
        self,
        retriever: Retriever,
        store: ChunkStore,
        generator: Generator | None = None,
        judge: Judge | None = None,
        model: str = "claude-opus-4-8",
    ) -> None:
        self._retriever = retriever
        self._store = store
        self._generator = generator
        self._judge = judge
        self._model = model

    def run(
        self,
        cases: list[EvalCase],
        strategy: Strategy,
        k: int,
        with_answers: bool = False,
    ) -> EvalRun:
        do_answers = (
            with_answers and self._generator is not None and self._judge is not None
        )
        chunks = self._store.all_chunks()
        results: list[CaseResult] = []
        total_cost = 0.0
        for case in cases:
            relevant = relevant_chunk_ids(chunks, case.relevant_snippets)
            # Retrieval latency only — LLM time is tracked separately so the
            # run-level percentiles stay comparable across judged/unjudged runs.
            started = time.perf_counter()
            retrieved = self._retriever.retrieve(case.question, k, strategy)
            retrieval_ms = _elapsed_ms(started)
            ranked = [rc.chunk.id for rc in retrieved]

            answer_text: str | None = None
            judge_scores = None
            case_cost = 0.0
            answer_ms = 0
            if do_answers:
                assert self._generator is not None and self._judge is not None
                # One bad API call (rate limit, refusal, truncation) must not
                # throw away the whole run: record the case as unanswered,
                # keep its retrieval metrics, and continue.
                answer_started = time.perf_counter()
                try:
                    answer = self._generator.answer(case.question, retrieved)
                    judge_scores, judge_usage = self._judge.score(
                        case.question, answer, retrieved, case.reference_answer
                    )
                    answer_text = answer.text
                    case_cost = cost_usd(self._model, answer.usage) + cost_usd(
                        self._model, judge_usage
                    )
                    total_cost += case_cost
                except Exception:
                    logger.exception("answer track failed for case %s", case.id)
                answer_ms = _elapsed_ms(answer_started)

            results.append(
                CaseResult(
                    case_id=case.id,
                    strategy=strategy.value,
                    retrieved_ids=ranked,
                    recall=recall_at_k(ranked, relevant, k),
                    precision=precision_at_k(ranked, relevant, k),
                    mrr=mrr(ranked, relevant),
                    ndcg=ndcg_at_k(ranked, relevant, k),
                    answer=answer_text,
                    judge_scores=judge_scores,
                    cost_usd=case_cost,
                    latency_ms=retrieval_ms,
                    answer_ms=answer_ms,
                )
            )

        latencies = [float(r.latency_ms) for r in results]
        return EvalRun(
            id=uuid.uuid4().hex,
            strategy=strategy,
            k=k,
            retrieval_metrics=_aggregate_retrieval(results),
            case_results=results,
            answer_metrics=_aggregate_answers(results) if do_answers else None,
            cost_usd=total_cost,
            latency_p50_ms=int(percentile(latencies, 50)),
            latency_p95_ms=int(percentile(latencies, 95)),
        )


def _elapsed_ms(started: float) -> int:
    """Wall-clock ms since `started`, rounded up so sub-ms never displays as 0."""
    return max(1, math.ceil((time.perf_counter() - started) * 1000))


def _aggregate_retrieval(results: list[CaseResult]) -> RunMetrics:
    return RunMetrics(
        recall=mean([r.recall for r in results]),
        precision=mean([r.precision for r in results]),
        mrr=mean([r.mrr for r in results]),
        ndcg=mean([r.ndcg for r in results]),
    )


def _aggregate_answers(results: list[CaseResult]) -> dict[str, float]:
    scored = [r.judge_scores for r in results if r.judge_scores is not None]
    if not scored:
        return {}
    return {
        "faithfulness": mean([s.faithfulness for s in scored]),
        "relevance": mean([s.relevance for s in scored]),
        "citation_correctness": mean([s.citation_correctness for s in scored]),
    }
