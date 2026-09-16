from dataclasses import replace
from typing import Any

import pytest

from rag.chunking.chunker import chunk_document
from rag.domain.datasets import Dataset, EvidenceUnit
from rag.domain.entities import Chunk, Document, EvalCase, RetrievedChunk, Strategy
from rag.domain.errors import EvalPreflightError
from rag.services.comparison import make_baseline
from rag.services.eval_service import EvalService


class MemoryIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls = 0

    def all_chunks(self) -> list[Chunk]:
        return self.chunks

    def retrieve(self, query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]:
        self.calls += 1
        return [RetrievedChunk(c, 1.0, i + 1) for i, c in enumerate(self.chunks[:k])]


def fixture() -> tuple[Dataset, MemoryIndex]:
    doc = Document("d", "d.md", "D", "zero one two three four five")
    evidence = EvidenceUnit("fact", "d", 2, 5, "two three four")
    case = EvalCase(
        "q", "Question?", [], "Answer", ["fact"], tags=["critical"], group_id="q"
    )
    dataset = Dataset("unit", "1", [doc], [evidence], [case])
    return dataset, MemoryIndex(chunk_document(doc, target_words=3, overlap_words=0))


def test_span_recall_combines_chunks_and_records_stable_identity() -> None:
    dataset, index = fixture()
    run = EvalService(index, index, dataset=dataset, runtime={"index_id": "idx_1"}).run(
        dataset.cases, Strategy.HYBRID, 2
    )
    assert run.case_results[0].evidence_recall == 1
    assert run.case_results[0].covered_evidence_ids == ["fact"]
    assert run.evidence_metrics == {
        "recall": 1,
        "eligible_cases": 1,
        "unanswerable_cases": 0,
    }
    assert run.slices["tag:critical"]["evidence_recall"] == 1
    assert run.manifest["schema_version"] == 2
    assert run.manifest["index_id"] == "idx_1"
    assert run.manifest["index_consistency"] == "immutable_snapshot"
    assert run.manifest["retrieval_level"] == "chunk"


def test_unanswerable_is_excluded_from_retrieval_mean() -> None:
    dataset, index = fixture()
    missing = EvalCase(
        "missing", "Unknown?", [], "Not present", answerable=False, group_id="missing"
    )
    dataset = replace(dataset, cases=[*dataset.cases, missing])
    run = EvalService(index, index, dataset=dataset).run(
        dataset.cases, Strategy.HYBRID, 2
    )
    assert run.retrieval_metrics.recall == 1
    assert run.case_results[1].evidence_recall is None
    assert run.evidence_metrics == {
        "recall": 1,
        "eligible_cases": 1,
        "unanswerable_cases": 1,
    }


def test_document_qrels_deduplicate_chunks_before_document_cutoff() -> None:
    dataset, index = fixture()
    doc2 = Document("other", "other.md", "Other", "last document")
    dataset = replace(
        dataset,
        documents=[*dataset.documents, doc2],
        evidence=[EvidenceUnit("fact", "other")],
    )
    index.chunks.extend(chunk_document(doc2))
    run = EvalService(index, index, dataset=dataset).run(
        dataset.cases, Strategy.HYBRID, 2
    )
    case = run.case_results[0]
    assert case.recall == 1
    assert case.mrr == 0.5
    assert case.precision == 0.5
    assert case.evidence_recall == 1
    assert len(case.retrieved_context) == 2
    assert run.manifest["retrieval_level"] == "document"
    assert run.manifest["retrieval_candidate_k"] == 50


@pytest.mark.parametrize("change", ["language", "stale", "reference"])
def test_dataset_preflight_rejects_ambiguous_or_unjudgable_cases(change: str) -> None:
    dataset, index = fixture()
    cases = dataset.cases
    kwargs: dict[str, Any] = {}
    if change == "language":
        cases = [*cases, replace(cases[0], id="pt", language="pt")]
        dataset = replace(dataset, cases=cases)
    elif change == "stale":
        cases = [replace(cases[0], question="Changed after indexing")]
    else:
        cases = [replace(cases[0], reference_answer="")]
        dataset = replace(dataset, cases=cases)
        from tests.unit.test_eval_service import StubGenerator, StubJudge

        kwargs = {"generator": StubGenerator(), "judge": StubJudge()}
    with pytest.raises(EvalPreflightError):
        EvalService(index, index, dataset=dataset, **kwargs).run(
            cases, Strategy.HYBRID, 2, with_answers=change == "reference"
        )
    assert index.calls == 0


@pytest.mark.parametrize("group", [None, ""])
def test_loaded_optional_group_can_complete_evaluation_and_export(
    group: str | None,
) -> None:
    from rag.adapters.datasets.loader import dataset_from_payload
    from rag.domain.datasets import dataset_payload

    dataset, index = fixture()
    payload = dataset_payload(dataset)
    if group is None:
        payload["cases"][0].pop("group_id")
    else:
        payload["cases"][0]["group_id"] = group
    loaded = dataset_from_payload(payload)
    run = EvalService(index, index, dataset=loaded).run(loaded.cases, Strategy.HYBRID, 2)
    assert make_baseline(run)["cases"][0]["group_id"] == "q"


def test_baseline_selection_rejects_changed_k_before_retrieval() -> None:
    dataset, index = fixture()
    service = EvalService(index, index, dataset=dataset)
    baseline = make_baseline(service.run(dataset.cases, Strategy.HYBRID, 2))
    index.calls = 0
    with pytest.raises(EvalPreflightError, match="incompatible manifest field: k"):
        service.run(dataset.cases, Strategy.HYBRID, 3, baseline=baseline)
    assert index.calls == 0


def test_equivalent_label_order_does_not_break_paired_comparison() -> None:
    dataset, index = fixture()
    dataset = replace(dataset, cases=[replace(dataset.cases[0], tags=["z", "a"])])
    before = EvalService(index, index, dataset=dataset).run(
        dataset.cases, Strategy.HYBRID, 2
    )
    dataset = replace(dataset, cases=[replace(dataset.cases[0], tags=["a", "z"])])
    after = EvalService(index, index, dataset=dataset).run(
        dataset.cases, Strategy.HYBRID, 2, baseline=make_baseline(before)
    )
    assert before.manifest["case_set_sha256"] == after.manifest["case_set_sha256"]
