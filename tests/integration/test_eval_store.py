from __future__ import annotations

from dataclasses import replace

import pytest

from rag.adapters.db.chunk_store import PgChunkStore
from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.pool import Pool
from rag.domain.entities import (
    CaseResult,
    Chunk,
    Document,
    EvalRun,
    RetrievedChunk,
    RunMetrics,
    StageResult,
    Strategy,
    Usage,
)

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
    store.save_run(_run(), {"strategy": "hybrid", "k": 10}, git_sha="abc1234")

    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["id"] == "run-1"
    assert runs[0]["retrieval_metrics"]["recall"] == 0.8
    assert runs[0]["git_sha"] == "abc1234"

    detail = store.get_run("run-1")
    assert detail is not None
    assert detail["config"]["strategy"] == "hybrid"
    assert len(detail["cases"]) == 1
    assert detail["cases"][0]["retrieved_ids"] == ["c1", "c2"]
    assert detail["cases"][0]["answer_ms"] == 345
    # A newly saved legacy-adapter run still has valid chunk metrics. Defaulted
    # version-2 labels must not falsely mark its retrieval quality as ineligible.
    assert detail["evidence_metrics"] is None
    assert detail["slices"] is None
    assert detail["cases"][0]["evidence_ids"] is None
    assert detail["cases"][0]["answerable"] is None
    assert detail["cases"][0]["recall"] == 1.0

    assert store.get_run("missing") is None


@pytest.mark.parametrize("scored", [0, 1])
def test_incomplete_run_retains_outcomes_cost_and_evidence(
    pool: Pool, scored: int
) -> None:
    baseline = _run()
    context = RetrievedChunk(Chunk("c1", "doc", 0, "original evidence", 2), 0.9, 1)
    result = replace(
        baseline.case_results[0],
        answer="retained answer",
        question="question?",
        reference_answer="reference",
        retrieved_context=[context],
        cost_usd=0.001,
        cost_complete=False,
        generation=StageResult("succeeded", Usage(100, 20), 0.001, 10),
        judging=StageResult("failed", None, None, 20, "RuntimeError"),
    )
    run = replace(
        baseline,
        case_results=[result],
        with_answers=True,
        status="partial" if scored else "failed",
        answer_attempted=2,
        answer_scored=scored,
        answer_failed=2 - scored,
        cost_complete=False,
        cost_usd=0.001,
        manifest={
            "schema_version": 1,
            "strategy": "hybrid",
            "k": 10,
            "git_sha": "revision",
        },
    )
    store = EvalStore(pool)
    store.save_run(run, {"strategy": "incorrect"})
    detail = store.get_run(run.id)
    assert detail is not None
    assert detail["config"] == run.manifest
    assert detail["git_sha"] == "revision"
    assert detail["evaluation"] == {
        "status": run.status,
        "with_answers": True,
        "attempted": 2,
        "scored": scored,
        "failed": 2 - scored,
        "retrieval_complete": True,
        "error_type": None,
    }
    assert detail["cost_complete"] is False
    assert detail["cost_usd"] == pytest.approx(0.001)
    saved_case = detail["cases"][0]
    assert saved_case["answer"] == "retained answer"
    assert saved_case["generation"]["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert saved_case["judging"]["error_type"] == "RuntimeError"
    assert saved_case["cost_complete"] is False
    assert saved_case["question"] == "question?"
    assert saved_case["reference_answer"] == "reference"
    assert saved_case["retrieved_context"][0]["chunk"]["text"] == "original evidence"
    assert store.list_runs()[0]["evaluation"] == detail["evaluation"]


def test_legacy_rows_have_unknown_coverage_and_cost_completeness(pool: Pool) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO eval_runs (id, config, retrieval_metrics) "
            "VALUES ('legacy', '{}', '{}')"
        )
        conn.execute(
            "INSERT INTO eval_case_results (id, run_id, case_id, strategy, "
            "retrieved_ids, recall, precision, mrr, ndcg) "
            "VALUES ('legacy-case', 'legacy', 'q1', 'hybrid', '[]', 0, 0, 0, 0)"
        )
    detail = EvalStore(pool).get_run("legacy")
    assert detail is not None
    assert detail["evaluation"] is None
    assert detail["cost_complete"] is None
    assert detail["evidence_metrics"] is None
    assert detail["slices"] is None
    saved_case = detail["cases"][0]
    for field in (
        "evidence_recall",
        "covered_evidence_ids",
        "evidence_ids",
        "split",
        "tags",
        "language",
        "answerable",
        "group_id",
    ):
        assert saved_case[field] is None


def test_evidence_scores_and_case_labels_survive_persistence(pool: Pool) -> None:
    baseline = _run()
    answerable = replace(
        baseline.case_results[0],
        evidence_recall=0.5,
        covered_evidence_ids=["span-1"],
        evidence_ids=["span-1", "span-2"],
        split="dev",
        tags=["critical", "paraphrase"],
        language="pt",
        group_id="original-q1",
        retrieved_context=[
            RetrievedChunk(Chunk("c1", "doc", 0, "one two three", 3, 0, 3), 0.9, 1)
        ],
    )
    unanswerable = replace(
        baseline.case_results[0],
        case_id="q2",
        split="dev",
        language="pt",
        answerable=False,
        group_id="original-q2",
    )
    run = replace(
        baseline,
        case_results=[answerable, unanswerable],
        evidence_metrics={"recall": 0.5, "eligible_cases": 1, "unanswerable_cases": 1},
        slices={
            "tag:critical": {"recall": 1.0, "evidence_recall": 0.5, "n": 1},
            "language:pt": {"recall": 1.0, "evidence_recall": 0.5, "n": 2},
        },
        manifest={"schema_version": 2, "split": "dev", "language": "pt"},
    )
    store = EvalStore(pool)
    store.save_run(run, {})
    detail = store.get_run(run.id)
    assert detail is not None
    assert detail["evidence_metrics"] == {
        "recall": 0.5,
        "eligible_cases": 1,
        "unanswerable_cases": 1,
    }
    assert detail["slices"] == {
        "tag:critical": {"recall": 1.0, "evidence_recall": 0.5, "n": 1},
        "language:pt": {"recall": 1.0, "evidence_recall": 0.5, "n": 2},
    }
    saved_answerable, saved_unanswerable = detail["cases"]
    assert saved_answerable["evidence_recall"] == 0.5
    assert saved_answerable["covered_evidence_ids"] == ["span-1"]
    assert saved_answerable["evidence_ids"] == ["span-1", "span-2"]
    assert saved_answerable["split"] == "dev"
    assert saved_answerable["tags"] == ["critical", "paraphrase"]
    assert saved_answerable["language"] == "pt"
    assert saved_answerable["answerable"] is True
    assert saved_answerable["group_id"] == "original-q1"
    assert saved_answerable["retrieved_context"][0]["chunk"]["word_start"] == 0
    assert saved_answerable["retrieved_context"][0]["chunk"]["word_end"] == 3
    assert saved_unanswerable["answerable"] is False
    assert saved_unanswerable["evidence_recall"] is None
    assert saved_unanswerable["covered_evidence_ids"] == []
    assert saved_unanswerable["evidence_ids"] == []
    summary = store.list_runs()[0]
    assert summary["evidence_metrics"] == detail["evidence_metrics"]
    assert summary["slices"] == detail["slices"]


@pytest.mark.parametrize("replace_corpus", [False, True])
def test_legacy_chunk_store_retains_known_and_unknown_offsets(
    pool: Pool, replace_corpus: bool
) -> None:
    doc = Document("doc", "fixture", "Offsets", "one two three four")
    chunks = [
        Chunk("c1", "doc", 0, "one two three", 3, 0, 3),
        Chunk("c2", "doc", 1, "three four", 2),
    ]
    vector = [1.0] + [0.0] * 383
    prepared = [(chunk, vector) for chunk in chunks]
    store = PgChunkStore(pool)
    if replace_corpus:
        store.replace_corpus([(doc, prepared)])
    else:
        store.add_document(doc, prepared)

    restored = store.all_chunks()
    assert [(chunk.id, chunk.word_start, chunk.word_end) for chunk in restored] == [
        ("c1", 0, 3),
        ("c2", None, None),
    ]
