from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any

from psycopg.types.json import Jsonb

from rag.adapters.db.pool import Pool
from rag.domain.entities import EvalRun


class EvalStore:
    """Persistence for evaluation runs and per-case results."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    def save_run(self, run: EvalRun, config: dict[str, Any]) -> None:
        answer_metrics = Jsonb(run.answer_metrics) if run.answer_metrics else None
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO eval_runs (id, config, retrieval_metrics, answer_metrics, "
                "cost_usd, latency_p50_ms, latency_p95_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    run.id,
                    Jsonb(config),
                    Jsonb(asdict(run.retrieval_metrics)),
                    answer_metrics,
                    run.cost_usd,
                    run.latency_p50_ms,
                    run.latency_p95_ms,
                ),
            )
            for r in run.case_results:
                judge = Jsonb(asdict(r.judge_scores)) if r.judge_scores else None
                conn.execute(
                    "INSERT INTO eval_case_results (id, run_id, case_id, strategy, "
                    "retrieved_ids, recall, precision, mrr, ndcg, answer, judge_scores, "
                    "cost_usd, latency_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        uuid.uuid4().hex,
                        run.id,
                        r.case_id,
                        r.strategy,
                        Jsonb(r.retrieved_ids),
                        r.recall,
                        r.precision,
                        r.mrr,
                        r.ndcg,
                        r.answer,
                        judge,
                        r.cost_usd,
                        r.latency_ms,
                    ),
                )
            conn.commit()

    def list_runs(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, created_at, config, retrieval_metrics, answer_metrics, "
                "cost_usd, latency_p50_ms, latency_p95_ms "
                "FROM eval_runs ORDER BY created_at DESC"
            ).fetchall()
        return [_run_row(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, created_at, config, retrieval_metrics, answer_metrics, "
                "cost_usd, latency_p50_ms, latency_p95_ms "
                "FROM eval_runs WHERE id = %s",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            cases = conn.execute(
                "SELECT case_id, strategy, retrieved_ids, recall, precision, mrr, ndcg, "
                "answer, judge_scores, cost_usd, latency_ms "
                "FROM eval_case_results WHERE run_id = %s ORDER BY case_id",
                (run_id,),
            ).fetchall()
        run = _run_row(row)
        run["cases"] = [
            {
                "case_id": c[0],
                "strategy": c[1],
                "retrieved_ids": c[2],
                "recall": c[3],
                "precision": c[4],
                "mrr": c[5],
                "ndcg": c[6],
                "answer": c[7],
                "judge_scores": c[8],
                "cost_usd": float(c[9]),
                "latency_ms": c[10],
            }
            for c in cases
        ]
        return run


def _run_row(r: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": r[0],
        "created_at": r[1].isoformat(),
        "config": r[2],
        "retrieval_metrics": r[3],
        "answer_metrics": r[4],
        "cost_usd": float(r[5]),
        "latency_p50_ms": r[6],
        "latency_p95_ms": r[7],
    }
