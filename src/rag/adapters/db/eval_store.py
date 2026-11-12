from __future__ import annotations

import uuid
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from rag.adapters.db.pool import Pool
from rag.domain.entities import CaseResult, EvalRun


class EvalStore:
    """Persistence for evaluation runs and per-case results."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    def save_run(
        self,
        run: EvalRun,
        config: dict[str, Any],
        git_sha: str | None = None,
        *,
        connection: Connection[Any] | None = None,
        case_results: Sequence[CaseResult] | None = None,
    ) -> None:
        """Upsert the run row and its case rows.

        ``case_results`` narrows the case writes to rows added or replaced since
        this run was last saved; ``None`` writes every case.
        """
        answer_metrics = Jsonb(run.answer_metrics) if run.answer_metrics else None
        evidence_metrics = (
            Jsonb(run.evidence_metrics) if run.evidence_metrics is not None else None
        )
        manifest = run.manifest or config
        revision = manifest.get("git_sha") or git_sha
        if revision is not None:
            manifest = {**manifest, "git_sha": revision}
        evaluation = {
            "with_answers": run.with_answers,
            "status": run.status,
            "attempted": run.answer_attempted,
            "scored": run.answer_scored,
            "failed": run.answer_failed,
            "retrieval_complete": run.retrieval_complete,
            "error_type": run.error_type,
        }
        transaction = (
            nullcontext(connection) if connection is not None else self._pool.connection()
        )
        with transaction as conn:
            conn.execute(
                "INSERT INTO eval_runs (id, config, retrieval_metrics, answer_metrics, "
                "cost_usd, latency_p50_ms, latency_p95_ms, git_sha, evaluation, "
                "cost_complete, evidence_metrics, slices, "
                "retrieval_complete, error_type) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "config = EXCLUDED.config, "
                "retrieval_metrics = EXCLUDED.retrieval_metrics, "
                "answer_metrics = EXCLUDED.answer_metrics, cost_usd = EXCLUDED.cost_usd, "
                "latency_p50_ms = EXCLUDED.latency_p50_ms, "
                "latency_p95_ms = EXCLUDED.latency_p95_ms, git_sha = EXCLUDED.git_sha, "
                "evaluation = EXCLUDED.evaluation, "
                "cost_complete = EXCLUDED.cost_complete, "
                "evidence_metrics = EXCLUDED.evidence_metrics, slices = EXCLUDED.slices, "
                "retrieval_complete = EXCLUDED.retrieval_complete, "
                "error_type = EXCLUDED.error_type",
                (
                    run.id,
                    Jsonb(manifest),
                    Jsonb(asdict(run.retrieval_metrics)),
                    answer_metrics,
                    run.cost_usd,
                    run.latency_p50_ms,
                    run.latency_p95_ms,
                    revision,
                    Jsonb(evaluation),
                    run.cost_complete,
                    evidence_metrics,
                    Jsonb(run.slices) if run.evidence_metrics is not None else None,
                    run.retrieval_complete,
                    run.error_type,
                ),
            )
            for r in run.case_results if case_results is None else case_results:
                judge = Jsonb(asdict(r.judge_scores)) if r.judge_scores else None
                evidence = {
                    "evidence_recall": r.evidence_recall,
                    "covered_evidence_ids": r.covered_evidence_ids,
                    "evidence_ids": r.evidence_ids,
                    "split": r.split,
                    "tags": r.tags,
                    "language": r.language,
                    "answerable": r.answerable,
                    "group_id": r.group_id,
                }
                conn.execute(
                    "INSERT INTO eval_case_results (id, run_id, case_id, strategy, "
                    "retrieved_ids, recall, precision, mrr, ndcg, answer, judge_scores, "
                    "cost_usd, latency_ms, answer_ms, generation, judging, "
                    "cost_complete, question, reference_answer, retrieved_context, "
                    "evidence, invalid_citations) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    "%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (run_id, case_id) DO UPDATE SET "
                    "strategy = EXCLUDED.strategy, "
                    "retrieved_ids = EXCLUDED.retrieved_ids, "
                    "recall = EXCLUDED.recall, precision = EXCLUDED.precision, "
                    "mrr = EXCLUDED.mrr, ndcg = EXCLUDED.ndcg, answer = EXCLUDED.answer, "
                    "judge_scores = EXCLUDED.judge_scores, cost_usd = EXCLUDED.cost_usd, "
                    "latency_ms = EXCLUDED.latency_ms, answer_ms = EXCLUDED.answer_ms, "
                    "generation = EXCLUDED.generation, judging = EXCLUDED.judging, "
                    "cost_complete = EXCLUDED.cost_complete, "
                    "question = EXCLUDED.question, "
                    "reference_answer = EXCLUDED.reference_answer, "
                    "retrieved_context = EXCLUDED.retrieved_context, "
                    "evidence = EXCLUDED.evidence, "
                    "invalid_citations = EXCLUDED.invalid_citations",
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
                        r.answer_ms,
                        Jsonb(asdict(r.generation)),
                        Jsonb(asdict(r.judging)),
                        r.cost_complete,
                        r.question,
                        r.reference_answer,
                        Jsonb([asdict(rc) for rc in r.retrieved_context]),
                        Jsonb(evidence) if run.evidence_metrics is not None else None,
                        Jsonb(r.invalid_citations)
                        if r.invalid_citations is not None
                        else None,
                    ),
                )
            if connection is None:
                conn.commit()

    def list_runs(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, created_at, config, retrieval_metrics, answer_metrics, "
                "cost_usd, latency_p50_ms, latency_p95_ms, git_sha, evaluation, "
                "cost_complete, evidence_metrics, slices, retrieval_complete, error_type "
                "FROM eval_runs ORDER BY created_at DESC"
            ).fetchall()
        return [_run_row(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, created_at, config, retrieval_metrics, answer_metrics, "
                "cost_usd, latency_p50_ms, latency_p95_ms, git_sha, evaluation, "
                "cost_complete, evidence_metrics, slices, retrieval_complete, error_type "
                "FROM eval_runs WHERE id = %s",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            cases = conn.execute(
                "SELECT case_id, strategy, retrieved_ids, recall, precision, mrr, ndcg, "
                "answer, judge_scores, cost_usd, latency_ms, answer_ms, generation, "
                "judging, cost_complete, question, reference_answer, retrieved_context, "
                "evidence, invalid_citations "
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
                "answer_ms": c[11],
                "generation": c[12],
                "judging": c[13],
                "cost_complete": c[14],
                "question": c[15],
                "reference_answer": c[16],
                "retrieved_context": c[17],
                **_evidence_row(c[18]),
                "invalid_citations": c[19],
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
        "git_sha": r[8],
        "evaluation": {
            **r[9],
            "retrieval_complete": r[13],
            "error_type": r[14],
        }
        if r[9] is not None
        else None,
        "cost_complete": r[10],
        "evidence_metrics": r[11],
        "slices": r[12],
    }


def _evidence_row(evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Expose historical missing metadata as unknown rather than inferred labels."""
    return {
        key: evidence.get(key) if evidence is not None else None
        for key in (
            "evidence_recall",
            "covered_evidence_ids",
            "evidence_ids",
            "split",
            "tags",
            "language",
            "answerable",
            "group_id",
        )
    }
