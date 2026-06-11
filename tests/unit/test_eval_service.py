from __future__ import annotations

from rag.domain.entities import Chunk, EvalCase, RetrievedChunk, Strategy
from rag.metrics.retrieval import mrr, recall_at_k
from rag.metrics.stats import mean
from rag.services.eval_service import EvalService
from rag.services.gate import check_thresholds


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, document_id="d", ordinal=0, text=text, token_count=0)


CHUNKS = [_chunk("c1", "alpha"), _chunk("c2", "beta"), _chunk("c3", "gamma")]


class FakeStore:
    def add_document(self, doc: object, chunks: object) -> None:  # pragma: no cover
        raise NotImplementedError

    def all_chunks(self) -> list[Chunk]:
        return list(CHUNKS)


class ScriptedRetriever:
    """Returns a fixed ranking of chunk ids per question."""

    def __init__(self, scripts: dict[str, list[str]]) -> None:
        self._scripts = scripts

    def retrieve(self, query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]:
        by_id = {c.id: c for c in CHUNKS}
        ranked = self._scripts[query][:k]
        return [
            RetrievedChunk(chunk=by_id[cid], score=1.0 / (i + 1), rank=i + 1)
            for i, cid in enumerate(ranked)
        ]


def test_run_retrieval_aggregates_metrics() -> None:
    cases = [
        EvalCase(
            id="q1", question="q1", relevant_snippets=["alpha"], reference_answer="a"
        ),
        EvalCase(
            id="q2", question="q2", relevant_snippets=["beta"], reference_answer="b"
        ),
    ]
    # q1: alpha (c1) retrieved at rank 1; q2: beta (c2) retrieved at rank 2.
    retriever = ScriptedRetriever({"q1": ["c1", "c3"], "q2": ["c3", "c2"]})
    run = EvalService(retriever, FakeStore()).run_retrieval(cases, Strategy.HYBRID, k=3)

    assert run.retrieval_metrics.recall == mean(
        [recall_at_k(["c1", "c3"], {"c1"}, 3), recall_at_k(["c3", "c2"], {"c2"}, 3)]
    )
    assert run.retrieval_metrics.mrr == mean(
        [mrr(["c1", "c3"], {"c1"}), mrr(["c3", "c2"], {"c2"})]
    )
    assert run.retrieval_metrics.mrr == mean([1.0, 0.5])
    assert len(run.case_results) == 2
    assert run.strategy is Strategy.HYBRID


def test_gate_reports_failures_below_threshold() -> None:
    from rag.domain.entities import RunMetrics

    good = RunMetrics(recall=0.9, precision=0.5, mrr=0.8, ndcg=0.85)
    assert check_thresholds(good, min_recall=0.7, min_ndcg=0.6) == []
    bad = RunMetrics(recall=0.5, precision=0.5, mrr=0.4, ndcg=0.4)
    failures = check_thresholds(bad, min_recall=0.7, min_ndcg=0.6)
    assert len(failures) == 2
