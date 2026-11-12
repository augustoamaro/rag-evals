from __future__ import annotations

import math
import time
import uuid
from dataclasses import replace
from typing import Any, Literal

from rag.domain.datasets import Dataset, EvidenceUnit
from rag.domain.entities import (
    Answer,
    CaseResult,
    Chunk,
    EvalCase,
    EvalRun,
    RetrievedChunk,
    RunMetrics,
    StageResult,
    Strategy,
)
from rag.domain.errors import EvalPreflightError
from rag.domain.evaluation import EvaluationControl
from rag.domain.ports import ChunkReader, Generator, Judge, Retriever
from rag.metrics.evidence import EvidenceScore, score_evidence
from rag.metrics.relevance import relevant_chunk_ids
from rag.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from rag.metrics.stats import mean, percentile
from rag.services.answer_evaluation import evaluate_answer
from rag.services.comparison import validate_baseline_selection
from rag.services.manifest import build_manifest


class EvalService:
    def __init__(
        self,
        retriever: Retriever,
        store: ChunkReader,
        generator: Generator | None = None,
        judge: Judge | None = None,
        model: str = "claude-opus-4-8",
        *,
        expected_chunks: list[Chunk] | None = None,
        runtime: dict[str, object] | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        self._retriever = retriever
        self._store = store
        self._generator = generator
        self._judge = judge
        self._model = model
        self._expected_chunks = expected_chunks
        self._runtime = runtime
        self._dataset = dataset

    def run(
        self,
        cases: list[EvalCase],
        strategy: Strategy,
        k: int,
        with_answers: bool = False,
        *,
        baseline: dict[str, Any] | None = None,
        control: EvaluationControl | None = None,
    ) -> EvalRun:
        if with_answers and (self._generator is None or self._judge is None):
            raise ValueError("answer track requires both a generator and a judge")
        if not cases or len({c.id for c in cases}) != len(cases):
            raise EvalPreflightError("eval requires nonempty cases with unique IDs")
        if not 1 <= k <= 50:
            raise EvalPreflightError("k must be between 1 and 50")
        chunks = self._store.all_chunks()
        evidence = self._case_evidence(cases, with_answers)
        document_level = bool(evidence) and all(
            unit.word_start is None for units in evidence.values() for unit in units
        )
        manifest = build_manifest(
            cases,
            chunks,
            strategy,
            k,
            with_answers,
            self._model,
            runtime=self._runtime,
            expected_chunks=self._expected_chunks,
            dataset=self._dataset,
            document_level=document_level,
        )
        if manifest["index_matches_expected"] is False:
            raise EvalPreflightError(
                "index does not match the expected corpus; "
                "run rag ingest before evaluating"
            )
        relevance = {
            c.id: _relevant(chunks, evidence[c.id], document_level)
            if self._dataset is not None
            else relevant_chunk_ids(chunks, c.relevant_snippets)
            for c in cases
        }
        missing = [c.id for c in cases if c.answerable and not relevance[c.id]]
        if missing:
            raise EvalPreflightError("unanchored golden cases: " + ", ".join(missing))
        if baseline is not None:
            try:
                validate_baseline_selection(baseline, manifest, cases)
            except ValueError as exc:
                raise EvalPreflightError(str(exc)) from exc
        results: list[CaseResult] = []
        run_id = control.run_id if control is not None else uuid.uuid4().hex
        completed = 0
        error_type: str | None = None

        def cancelled() -> bool:
            return control is not None and control.is_cancelled()

        def snapshot(*, running: bool = True) -> EvalRun:
            return _build_run(
                run_id,
                strategy,
                k,
                list(results),
                manifest,
                with_answers,
                versioned=self._dataset is not None,
                running=running,
                # Checkpoint creation must not do cancellation I/O before the
                # runner receives newly observed answers and paid-stage usage.
                cancelled=cancelled() if not running else False,
                error_type=error_type,
            )

        def checkpoint(stage: str, case_id: str | None) -> None:
            if control is not None:
                control.checkpoint(snapshot(), completed, stage, case_id)

        checkpoint("preflight", None)
        for case in cases:
            if cancelled():
                break
            checkpoint("retrieval", case.id)
            if cancelled():
                break
            relevant = relevance[case.id]
            # Retrieval latency excludes persistence and all model calls.
            started = time.perf_counter()
            try:
                retrieved = self._retriever.retrieve(
                    case.question, 50 if document_level else k, strategy
                )
                if len({rc.chunk.id for rc in retrieved}) != len(retrieved):
                    raise EvalPreflightError("retriever returned duplicate chunk IDs")
                retrieved = (
                    _unique_documents(retrieved, k) if document_level else retrieved[:k]
                )
            except Exception as exc:
                if control is None:
                    raise
                error_type = type(exc).__name__
                break
            retrieval_ms = _elapsed_ms(started)
            ranked = [
                rc.chunk.document_id if document_level else rc.chunk.id
                for rc in retrieved
            ]
            evidence_score = (
                score_evidence(retrieved, evidence[case.id])
                if self._dataset is not None
                else EvidenceScore(None, [])
            )
            results.append(
                CaseResult(
                    case_id=case.id,
                    strategy=strategy.value,
                    retrieved_ids=[rc.chunk.id for rc in retrieved],
                    recall=recall_at_k(ranked, relevant, k),
                    precision=precision_at_k(ranked, relevant, k),
                    mrr=mrr(ranked, relevant),
                    ndcg=ndcg_at_k(ranked, relevant, k),
                    latency_ms=retrieval_ms,
                    question=case.question,
                    reference_answer=case.reference_answer,
                    retrieved_context=retrieved,
                    evidence_recall=evidence_score.recall,
                    covered_evidence_ids=evidence_score.covered_ids,
                    evidence_ids=sorted(case.evidence_ids),
                    split=case.split,
                    tags=sorted(case.tags),
                    language=case.language,
                    answerable=case.answerable,
                    group_id=case.group_id,
                )
            )
            if with_answers:
                assert self._generator is not None and self._judge is not None
                checkpoint("generation", case.id)

                def generated(
                    answer: Answer | None, generation: StageResult, case_id: str = case.id
                ) -> None:
                    results[-1] = replace(
                        results[-1],
                        answer=answer.text if answer else None,
                        invalid_citations=answer.invalid_citations if answer else None,
                        generation=generation,
                        cost_usd=generation.cost_usd or 0.0,
                        cost_complete=generation.cost_complete,
                        answer_ms=generation.latency_ms,
                    )
                    checkpoint(
                        "judging" if answer is not None else "case_complete", case_id
                    )

                answer, scores, generation, judging = evaluate_answer(
                    case,
                    retrieved,
                    self._generator,
                    self._judge,
                    self._model,
                    is_cancelled=cancelled,
                    on_generation=generated,
                )
                results[-1] = replace(
                    results[-1],
                    answer=answer.text if answer is not None else None,
                    invalid_citations=(
                        answer.invalid_citations if answer is not None else None
                    ),
                    judge_scores=scores,
                    generation=generation,
                    judging=judging,
                    cost_usd=(generation.cost_usd or 0.0) + (judging.cost_usd or 0.0),
                    cost_complete=generation.cost_complete and judging.cost_complete,
                    answer_ms=generation.latency_ms + judging.latency_ms,
                )
            if (
                not with_answers
                or results[-1].generation.status == "failed"
                or results[-1].judging.status != "skipped"
            ):
                completed += 1
            checkpoint("case_complete", case.id)
        return snapshot(running=False)

    def _case_evidence(
        self,
        cases: list[EvalCase],
        with_answers: bool,
    ) -> dict[str, list[EvidenceUnit]]:
        if self._dataset is None:
            return {}
        if len({c.split for c in cases}) != 1 or len({c.language for c in cases}) != 1:
            raise EvalPreflightError("select a single split and language per evaluation")
        original = {c.id: c for c in self._dataset.cases}
        if any(original.get(c.id) != c for c in cases):
            raise EvalPreflightError("cases do not match the indexed dataset")
        if with_answers and any(not c.reference_answer.strip() for c in cases):
            raise EvalPreflightError(
                "answer evaluation requires reference answers for every case"
            )
        by_id = {e.id: e for e in self._dataset.evidence}
        evidence = {}
        for case in cases:
            if bool(case.evidence_ids) != case.answerable:
                raise EvalPreflightError(f"case {case.id} has inconsistent answerability")
            if any(eid not in by_id for eid in case.evidence_ids):
                raise EvalPreflightError(f"case {case.id} refers to unknown evidence")
            evidence[case.id] = [by_id[eid] for eid in case.evidence_ids]
        return evidence


def _build_run(
    run_id: str,
    strategy: Strategy,
    k: int,
    results: list[CaseResult],
    manifest: dict[str, object],
    with_answers: bool,
    *,
    versioned: bool,
    running: bool,
    cancelled: bool,
    error_type: str | None,
) -> EvalRun:
    latencies = [float(r.latency_ms) for r in results]
    attempted = sum(r.generation.status != "skipped" for r in results)
    scored = sum(r.judge_scores is not None for r in results)
    failed = attempted - scored
    status: Literal["running", "completed", "partial", "failed", "cancelled"]
    # Keep cancellation and infrastructure failures distinct from answer failures.
    if running:
        status = "running"
    elif error_type is not None:
        status = "failed"
    elif cancelled:
        status = "cancelled"
    else:
        status = ("failed" if scored == 0 else "partial") if failed else "completed"
    return EvalRun(
        id=run_id,
        strategy=strategy,
        k=k,
        retrieval_metrics=_aggregate_retrieval(results),
        case_results=results,
        answer_metrics=_aggregate_answers(results) if with_answers else None,
        cost_usd=sum(r.cost_usd for r in results),
        latency_p50_ms=int(percentile(latencies, 50)),
        latency_p95_ms=int(percentile(latencies, 95)),
        with_answers=with_answers,
        status=status,
        answer_attempted=attempted,
        answer_scored=scored,
        answer_failed=failed,
        cost_complete=all(r.cost_complete for r in results),
        manifest=manifest,
        evidence_metrics=_aggregate_evidence(results) if versioned else None,
        slices=_slices(results) if versioned else {},
        retrieval_complete=not running and not cancelled and error_type is None,
        error_type=error_type,
    )


def _elapsed_ms(started: float) -> int:
    """Wall-clock ms since `started`, rounded up so sub-ms never displays as 0."""
    return max(1, math.ceil((time.perf_counter() - started) * 1000))


def _aggregate_retrieval(results: list[CaseResult]) -> RunMetrics:
    results = [r for r in results if r.answerable]
    return RunMetrics(
        recall=mean([r.recall for r in results]),
        precision=mean([r.precision for r in results]),
        mrr=mean([r.mrr for r in results]),
        ndcg=mean([r.ndcg for r in results]),
    )


def _relevant(
    chunks: list[Chunk], evidence: list[EvidenceUnit], document_level: bool
) -> set[str]:
    relevant: set[str] = set()
    for chunk in chunks:
        for unit in evidence:
            if chunk.document_id != unit.document_id:
                continue
            if document_level:
                relevant.add(chunk.document_id)
            elif unit.word_start is None:
                relevant.add(chunk.id)
            elif chunk.word_start is None or chunk.word_end is None:
                raise EvalPreflightError("span evidence requires indexed word offsets")
            elif (
                unit.word_end is not None
                and chunk.word_start < unit.word_end
                and chunk.word_end > unit.word_start
            ):
                relevant.add(chunk.id)
    return relevant


def _unique_documents(retrieved: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    seen: set[str] = set()
    selected: list[RetrievedChunk] = []
    for result in retrieved:
        if result.chunk.document_id not in seen:
            seen.add(result.chunk.document_id)
            selected.append(replace(result, rank=len(selected) + 1))
            if len(selected) == k:
                break
    return selected


def _aggregate_evidence(results: list[CaseResult]) -> dict[str, float | int | None]:
    scored = [r.evidence_recall for r in results if r.evidence_recall is not None]
    return {
        "recall": mean(scored) if scored else None,
        "eligible_cases": len(scored),
        "unanswerable_cases": sum(not r.answerable for r in results),
    }


def _slices(results: list[CaseResult]) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[CaseResult]] = {}
    for result in results:
        for label in [
            f"language:{result.language}",
            *[f"tag:{tag}" for tag in result.tags],
        ]:
            groups.setdefault(label, []).append(result)
    output: dict[str, dict[str, float | int | None]] = {}
    for label, group in sorted(groups.items()):
        eligible = [r for r in group if r.answerable]
        metrics = _aggregate_retrieval(group)
        output[label] = {
            "count": len(group),
            "eligible_cases": len(eligible),
            "recall": metrics.recall if eligible else None,
            "precision": metrics.precision if eligible else None,
            "mrr": metrics.mrr if eligible else None,
            "ndcg": metrics.ndcg if eligible else None,
            "evidence_recall": _aggregate_evidence(group)["recall"],
        }
    return output


def _aggregate_answers(results: list[CaseResult]) -> dict[str, float] | None:
    scored = [r.judge_scores for r in results if r.judge_scores is not None]
    if not scored:
        return None
    return {
        "faithfulness": mean([s.faithfulness for s in scored]),
        "relevance": mean([s.relevance for s in scored]),
        "citation_correctness": mean([s.citation_correctness for s in scored]),
    }
