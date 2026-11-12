from __future__ import annotations

import pytest

from rag.domain.entities import (
    Answer,
    AnswerScores,
    Chunk,
    EvalCase,
    RetrievedChunk,
    Strategy,
    Usage,
)
from rag.metrics.retrieval import mrr, recall_at_k
from rag.metrics.stats import mean
from rag.services.eval_service import EvalService
from rag.services.gate import check_thresholds


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, document_id="d", ordinal=0, text=text, token_count=0)


CHUNKS = [_chunk("c1", "alpha"), _chunk("c2", "beta"), _chunk("c3", "gamma")]


class FakeStore:
    def replace_corpus(self, documents: object) -> None:  # pragma: no cover
        raise NotImplementedError

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
    run = EvalService(retriever, FakeStore()).run(cases, Strategy.HYBRID, k=3)

    assert run.retrieval_metrics.recall == mean(
        [recall_at_k(["c1", "c3"], {"c1"}, 3), recall_at_k(["c3", "c2"], {"c2"}, 3)]
    )
    assert run.retrieval_metrics.mrr == mean(
        [mrr(["c1", "c3"], {"c1"}), mrr(["c3", "c2"], {"c2"})]
    )
    assert run.retrieval_metrics.mrr == mean([1.0, 0.5])
    assert len(run.case_results) == 2
    assert run.strategy is Strategy.HYBRID
    assert run.answer_metrics is None  # no answer track without judge


class StubGenerator:
    def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
        return Answer(text="grounded answer", citations=[], usage=Usage(100, 20))


class StubJudge:
    def score(
        self,
        question: str,
        answer: Answer,
        context: list[RetrievedChunk],
        reference_answer: str,
    ) -> tuple[AnswerScores, Usage]:
        scores = AnswerScores(
            faithfulness=0.9, relevance=0.8, citation_correctness=0.7, rationale="ok"
        )
        return scores, Usage(50, 10)


def test_answer_track_aggregates_scores_and_cost() -> None:
    cases = [
        EvalCase(
            id="q1", question="q1", relevant_snippets=["alpha"], reference_answer="a"
        )
    ]
    service = EvalService(
        ScriptedRetriever({"q1": ["c1"]}),
        FakeStore(),
        generator=StubGenerator(),
        judge=StubJudge(),
        model="claude-opus-4-8",
    )
    run = service.run(cases, Strategy.HYBRID, k=3, with_answers=True)
    assert run.answer_metrics == {
        "faithfulness": 0.9,
        "relevance": 0.8,
        "citation_correctness": 0.7,
    }
    assert run.case_results[0].answer == "grounded answer"
    # cost = generator (100 in, 20 out) + judge (50 in, 10 out) at Opus 4.8 pricing
    assert run.cost_usd > 0.0


class ExplodingGenerator:
    """Fails on one specific question — a rate limit mid-run, for example."""

    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on

    def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
        if question == self._fail_on:
            raise RuntimeError("rate limited")
        return Answer(text="ok", citations=[], usage=Usage(100, 20))


def test_answer_track_survives_per_case_failures() -> None:
    cases = [
        EvalCase(
            id="q1", question="q1", relevant_snippets=["alpha"], reference_answer="a"
        ),
        EvalCase(
            id="q2", question="q2", relevant_snippets=["beta"], reference_answer="b"
        ),
    ]
    service = EvalService(
        ScriptedRetriever({"q1": ["c1"], "q2": ["c2"]}),
        FakeStore(),
        generator=ExplodingGenerator(fail_on="q1"),
        judge=StubJudge(),
        model="claude-opus-4-8",
    )
    run = service.run(cases, Strategy.HYBRID, k=3, with_answers=True)

    # The run completes: the failed case is recorded without an answer, the
    # healthy case is judged, and aggregates cover only the scored cases.
    by_id = {r.case_id: r for r in run.case_results}
    assert by_id["q1"].answer is None
    assert by_id["q1"].judge_scores is None
    assert by_id["q2"].answer == "ok"
    assert run.answer_metrics == {
        "faithfulness": 0.9,
        "relevance": 0.8,
        "citation_correctness": 0.7,
    }
    # retrieval metrics still cover ALL cases
    assert len(run.case_results) == 2


def test_gate_reports_failures_below_threshold() -> None:
    from rag.domain.entities import RunMetrics

    good = RunMetrics(recall=0.9, precision=0.5, mrr=0.8, ndcg=0.85)
    assert check_thresholds(good, min_recall=0.7, min_ndcg=0.6) == []
    bad = RunMetrics(recall=0.5, precision=0.5, mrr=0.4, ndcg=0.4)
    failures = check_thresholds(bad, min_recall=0.7, min_ndcg=0.6)
    assert len(failures) == 2


class FailingJudge:
    def score(
        self,
        question: str,
        answer: Answer,
        context: list[RetrievedChunk],
        reference_answer: str,
    ) -> tuple[AnswerScores, Usage]:
        raise RuntimeError("provider failed with a potentially sensitive message")


def test_failed_judge_preserves_generated_answer_and_known_cost() -> None:
    case = EvalCase("q1", "q1", ["alpha"], "a")
    run = EvalService(
        ScriptedRetriever({"q1": ["c1"]}), FakeStore(), StubGenerator(), FailingJudge()
    ).run([case], Strategy.HYBRID, 3, with_answers=True)

    result = run.case_results[0]
    assert result.answer == "grounded answer"
    assert result.cost_usd == pytest.approx(0.001)
    assert result.generation.usage == Usage(100, 20)
    assert result.generation.status == "succeeded"
    assert result.judging.status == "failed"
    assert result.judging.error_type == "RuntimeError"
    assert result.judging.usage is None
    assert not result.cost_complete
    assert run.status == "failed"
    assert run.answer_metrics is None
    assert run.answer_attempted == 1
    assert run.answer_scored == 0
    assert run.answer_failed == 1
    assert not run.cost_complete


def test_single_success_out_of_thirty_is_partial_and_preserves_spending() -> None:
    class OneSuccessJudge(FailingJudge):
        def score(
            self,
            question: str,
            answer: Answer,
            context: list[RetrievedChunk],
            reference_answer: str,
        ) -> tuple[AnswerScores, Usage]:
            if question == "q0":
                return AnswerScores(1.0, 1.0, 1.0, "ok"), Usage(50, 10)
            return super().score(question, answer, context, reference_answer)

    cases = [EvalCase(f"q{i}", f"q{i}", ["alpha"], "a") for i in range(30)]
    run = EvalService(
        ScriptedRetriever({c.question: ["c1"] for c in cases}),
        FakeStore(),
        StubGenerator(),
        OneSuccessJudge(),
    ).run(cases, Strategy.HYBRID, 3, with_answers=True)

    assert run.cost_usd == pytest.approx(0.0305)
    assert run.status == "partial"
    assert (run.answer_attempted, run.answer_scored, run.answer_failed) == (30, 1, 29)
    assert run.answer_metrics == {
        "faithfulness": 1.0,
        "relevance": 1.0,
        "citation_correctness": 1.0,
    }


def test_answer_track_requires_both_adapters() -> None:
    with pytest.raises(ValueError, match="answer track"):
        EvalService(ScriptedRetriever({"q1": ["c1"]}), FakeStore()).run(
            [EvalCase("q1", "q1", ["alpha"], "a")],
            Strategy.HYBRID,
            3,
            with_answers=True,
        )


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [EvalCase("q1", "q1", ["alpha"], "a"), EvalCase("q1", "q2", ["beta"], "b")],
        [EvalCase("q1", "q1", ["absent evidence"], "a")],
    ],
)
def test_invalid_eval_data_fails_before_retrieval(cases: list[EvalCase]) -> None:
    # Empty scripts would raise KeyError if invalid data reaches retrieval.
    with pytest.raises(ValueError):
        EvalService(ScriptedRetriever({}), FakeStore()).run(cases, Strategy.HYBRID, 3)


def test_index_mismatch_fails_before_retrieval() -> None:
    from rag.domain.errors import EvalPreflightError

    with pytest.raises(EvalPreflightError, match="index.*corpus"):
        EvalService(ScriptedRetriever({}), FakeStore(), expected_chunks=CHUNKS[:2]).run(
            [EvalCase("q1", "q1", ["alpha"], "a")], Strategy.HYBRID, 3
        )


def test_failed_billable_call_preserves_usage_and_known_cost() -> None:
    from rag.domain.errors import ModelCallError

    class TruncatedGenerator(StubGenerator):
        def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
            raise ModelCallError("truncated", usage=Usage(100, 20))

    run = EvalService(
        ScriptedRetriever({"q1": ["c1"]}), FakeStore(), TruncatedGenerator(), StubJudge()
    ).run([EvalCase("q1", "q1", ["alpha"], "a")], Strategy.HYBRID, 3, True)
    assert run.status == "failed"
    assert run.cost_usd == pytest.approx(0.001)
    assert run.cost_complete
    assert run.case_results[0].generation.usage == Usage(100, 20)
    assert run.case_results[0].judging.status == "skipped"


def test_successful_unknown_model_has_unknown_cost() -> None:
    run = EvalService(
        ScriptedRetriever({"q1": ["c1"]}),
        FakeStore(),
        StubGenerator(),
        StubJudge(),
        model="unpriced-model",
    ).run([EvalCase("q1", "q1", ["alpha"], "a")], Strategy.HYBRID, 3, True)
    assert run.status == "completed"
    assert not run.cost_complete
    assert run.case_results[0].generation.cost_usd is None
    assert run.case_results[0].judging.cost_usd is None


def test_run_preserves_case_evidence_and_fingerprints() -> None:
    case = EvalCase("q1", "q1", ["alpha"], "reference")
    run = EvalService(
        ScriptedRetriever({"q1": ["c1"]}), FakeStore(), expected_chunks=CHUNKS
    ).run([case], Strategy.HYBRID, 3)
    assert run.manifest["schema_version"] == 1
    assert run.manifest["index_matches_expected"] is True
    assert run.manifest["with_answers"] is False
    result = run.case_results[0]
    assert result.question == case.question
    assert result.reference_answer == "reference"
    assert result.retrieved_context[0].chunk.text == "alpha"


def test_case_results_record_citations_of_passages_that_were_not_retrieved() -> None:
    class CitingGenerator:
        def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
            invalid = ["doc:0"] if question == "q1" else []
            return Answer(text="a", usage=Usage(1, 1), invalid_citations=invalid)

    cases = [
        EvalCase(id=q, question=q, relevant_snippets=[text], reference_answer="r")
        for q, text in (("q1", "alpha"), ("q2", "beta"))
    ]
    retriever = ScriptedRetriever({"q1": ["c1"], "q2": ["c2"]})
    service = EvalService(retriever, FakeStore(), CitingGenerator(), StubJudge())
    judged = service.run(cases, Strategy.HYBRID, k=3, with_answers=True)
    assert [r.invalid_citations for r in judged.case_results] == [["doc:0"], []]
    # Without an answer there is nothing to check: unknown, not "no problems".
    retrieval_only = service.run(cases, Strategy.HYBRID, k=3)
    assert [r.invalid_citations for r in retrieval_only.case_results] == [None, None]
