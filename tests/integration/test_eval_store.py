from __future__ import annotations

import pytest

from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.pool import Pool
from rag.domain.entities import CaseResult, EvalRun, RunMetrics, Strategy

pytestmark = pytest.mark.integration


def _run() -> EvalRun:
    return EvalRun(
        id="run-1",
        strategy=Strategy.HYBRID,
        k=10,
        retrieval_metrics=RunMetrics(recall=0.8, precision=0.5, mrr=0.7, ndcg=0.75),
        case_results=[
            CaseResult(
                case_id="q1",
                strategy="hybrid",
                retrieved_ids=["c1", "c2"],
                recall=1.0,
                precision=0.5,
                mrr=1.0,
                ndcg=1.0,
                latency_ms=12,
                answer_ms=345,
            )
        ],
        latency_p50_ms=12,
        latency_p95_ms=12,
    )


def test_save_and_read_back_run(pool: Pool) -> None:
    store = EvalStore(pool)
    store.save_run(_run(), {"strategy": "hybrid", "k": 10})

    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["id"] == "run-1"
    assert runs[0]["retrieval_metrics"]["recall"] == 0.8

    detail = store.get_run("run-1")
    assert detail is not None
    assert detail["config"]["strategy"] == "hybrid"
    assert len(detail["cases"]) == 1
    assert detail["cases"][0]["retrieved_ids"] == ["c1", "c2"]
    assert detail["cases"][0]["answer_ms"] == 345

    assert store.get_run("missing") is None
