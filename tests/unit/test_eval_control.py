from __future__ import annotations

from collections.abc import Callable

import pytest

from rag.domain.entities import (
    Answer,
    AnswerScores,
    EvalCase,
    EvalRun,
    RetrievedChunk,
    Strategy,
    Usage,
)
from rag.services.eval_service import EvalService
from tests.unit.test_eval_service import (
    FakeStore,
    ScriptedRetriever,
    StubGenerator,
    StubJudge,
)

CASES = [EvalCase(f"q{i}", f"q{i}", ["alpha"], "reference") for i in range(2)]


class Control:
    run_id = "controlled-run"

    def __init__(self) -> None:
        self.cancelled = False
        self.saved: list[tuple[EvalRun, int, str, str | None]] = []
        self.hook: Callable[[str], None] = lambda _: None

    def is_cancelled(self) -> bool:
        return self.cancelled

    def checkpoint(
        self, run: EvalRun, completed: int, stage: str, case_id: str | None
    ) -> None:
        self.saved.append((run, completed, stage, case_id))
        self.hook(stage)


def _service(generator: StubGenerator | None = None) -> EvalService:
    return EvalService(
        ScriptedRetriever({c.id: ["c1"] for c in CASES}),
        FakeStore(),
        generator or StubGenerator(),
        StubJudge(),
    )


def test_control_checkpoints_have_stable_identity_and_immutable_results() -> None:
    control = Control()
    run = _service().run(CASES, Strategy.HYBRID, 3, True, control=control)
    assert run.id == control.run_id
    assert run.status == "completed" and run.retrieval_complete
    assert control.saved[0][0].case_results == []
    generation_done = [record for record in control.saved if record[2] == "judging"]
    assert len(generation_done) == 2
    first = generation_done[0][0]
    assert len(first.case_results) == 1
    assert first.case_results[0].answer == "grounded answer"
    assert first.case_results[0].judging.status == "skipped"
    assert first.cost_usd == pytest.approx(0.001)
    assert all(row[0].id == run.id for row in control.saved)


def test_cancel_before_first_case_does_no_retrieval() -> None:
    control = Control()
    control.cancelled = True
    service = EvalService(ScriptedRetriever({}), FakeStore())
    run = service.run(CASES, Strategy.DENSE, 3, control=control)
    assert run.status == "cancelled"
    assert not run.retrieval_complete
    assert run.case_results == [] and run.cost_usd == 0


def test_cancel_at_generation_boundary_prevents_paid_call() -> None:
    control = Control()

    class MustNotGenerate(StubGenerator):
        def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
            raise AssertionError("generation must never start")

    def cancel(stage: str) -> None:
        if stage == "generation":
            control.cancelled = True

    control.hook = cancel
    run = _service(MustNotGenerate()).run(
        CASES, Strategy.HYBRID, 3, True, control=control
    )
    assert run.status == "cancelled" and run.answer_attempted == 0
    assert run.case_results[0].generation.status == "skipped"
    assert run.cost_usd == 0 and run.cost_complete


def test_cancel_during_generation_retains_answer_cost_without_judging() -> None:
    control = Control()

    class CancellingGenerator(StubGenerator):
        def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
            control.cancelled = True
            return super().answer(question, context)

    run = _service(CancellingGenerator()).run(
        CASES, Strategy.HYBRID, 3, True, control=control
    )
    assert run.status == "cancelled" and not run.retrieval_complete
    assert len(run.case_results) == 1
    assert run.answer_attempted == 1 and run.answer_scored == 0
    assert run.case_results[0].answer == "grounded answer"
    assert run.case_results[0].judging.status == "skipped"
    assert run.cost_usd == pytest.approx(0.001) and run.cost_complete


def test_cancel_at_judge_boundary_retains_generation() -> None:
    control = Control()

    def cancel(stage: str) -> None:
        if stage == "judging":
            control.cancelled = True

    control.hook = cancel
    run = _service().run(CASES, Strategy.HYBRID, 3, True, control=control)
    assert run.status == "cancelled" and run.answer_scored == 0
    assert run.case_results[0].judging.status == "skipped"
    assert run.cost_usd == pytest.approx(0.001)


def test_retrieval_failure_retains_previous_paid_case_and_safe_error() -> None:
    control = Control()
    service = EvalService(
        ScriptedRetriever({"q0": ["c1"]}), FakeStore(), StubGenerator(), StubJudge()
    )
    run = service.run(CASES, Strategy.HYBRID, 3, True, control=control)
    assert run.status == "failed" and run.error_type == "KeyError"
    assert not run.retrieval_complete
    assert len(run.case_results) == 1 and run.answer_scored == 1
    assert run.cost_usd == pytest.approx(0.0015) and run.cost_complete


def test_checkpoint_failure_stops_before_any_paid_stage() -> None:
    control = Control()

    def unavailable(stage: str) -> None:
        if stage == "generation":
            raise ConnectionError("checkpoint unavailable")

    control.hook = unavailable
    with pytest.raises(ConnectionError):
        _service().run(CASES, Strategy.HYBRID, 3, True, control=control)
    assert all(record[0].cost_usd == 0 for record in control.saved)


def test_cancelling_in_flight_retrieval_counts_its_finished_case() -> None:
    control = Control()

    class CancelDuringRetrieval(ScriptedRetriever):
        def retrieve(
            self, query: str, k: int, strategy: Strategy
        ) -> list[RetrievedChunk]:
            control.cancelled = True
            return super().retrieve(query, k, strategy)

    service = EvalService(CancelDuringRetrieval({"q0": ["c1"]}), FakeStore())
    run = service.run(CASES, Strategy.HYBRID, 3, control=control)
    assert run.status == "cancelled" and not run.retrieval_complete
    assert len(run.case_results) == 1
    assert control.saved[-1][1] == 1


def test_cancellation_lookup_failure_after_judging_preserves_observed_usage() -> None:
    class LookupFailureControl(Control):
        fail_reads = False

        def is_cancelled(self) -> bool:
            if self.fail_reads:
                raise ConnectionError("cancellation lookup unavailable")
            return super().is_cancelled()

    control = LookupFailureControl()

    class CompletedJudge(StubJudge):
        def score(
            self,
            question: str,
            answer: Answer,
            context: list[RetrievedChunk],
            reference_answer: str,
        ) -> tuple[AnswerScores, Usage]:
            observed = super().score(question, answer, context, reference_answer)
            control.fail_reads = True
            return observed

    service = EvalService(
        ScriptedRetriever({"q0": ["c1"]}),
        FakeStore(),
        StubGenerator(),
        CompletedJudge(),
    )
    with pytest.raises(ConnectionError, match="cancellation lookup"):
        service.run(CASES, Strategy.HYBRID, 3, True, control=control)

    retained, completed, _, _ = control.saved[-1]
    assert retained.cost_usd == pytest.approx(0.0015)
    assert retained.cost_complete
    assert retained.case_results[0].judge_scores is not None
    assert retained.case_results[0].judging.usage == Usage(50, 10)
    assert completed == 1
