from __future__ import annotations

from rag.chunking.chunker import chunk_document
from rag.domain.datasets import EvidenceUnit
from rag.domain.entities import Chunk, Document, RetrievedChunk
from rag.metrics.evidence import score_evidence


def _retrieved(doc: str, start: int, end: int, ordinal: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        Chunk(f"{doc}:{ordinal}", doc, ordinal, "words", end - start, start, end),
        1.0,
        ordinal + 1,
    )


def test_evidence_denominator_is_stable_across_chunk_boundaries() -> None:
    doc = Document("doc", "s", "t", "zero one two three four five six seven")
    unit = EvidenceUnit("claim", "doc", 2, 7, "two three four five six")
    for target, overlap in [(8, 0), (4, 1), (3, 0)]:
        chunks = chunk_document(doc, target_words=target, overlap_words=overlap)
        result = score_evidence(
            [RetrievedChunk(c, 1, i + 1) for i, c in enumerate(chunks)], [unit]
        )
        assert result.recall == 1.0
        assert result.covered_ids == ["claim"]


def test_overlap_copies_do_not_count_as_extra_evidence() -> None:
    units = [EvidenceUnit("one", "doc", 2, 4), EvidenceUnit("two", "doc", 10, 12)]
    result = score_evidence([_retrieved("doc", 0, 5), _retrieved("doc", 1, 6, 1)], units)
    assert result.recall == 0.5
    assert result.covered_ids == ["one"]


def test_partial_span_or_same_text_from_wrong_document_does_not_cover_evidence() -> None:
    unit = EvidenceUnit("one", "doc", 2, 8)
    result = score_evidence([_retrieved("doc", 0, 5), _retrieved("other", 5, 10)], [unit])
    assert result.recall == 0.0
    assert result.covered_ids == []


def test_document_qrels_count_once_despite_many_chunks() -> None:
    result = score_evidence(
        [_retrieved("doc", 0, 5), _retrieved("doc", 4, 10, 1)],
        [EvidenceUnit("d", "doc"), EvidenceUnit("x", "absent")],
    )
    assert result.recall == 0.5
    assert result.covered_ids == ["d"]


def test_unanswerable_has_no_retrieval_quality_denominator() -> None:
    result = score_evidence([_retrieved("doc", 0, 5)], [])
    assert result.recall is None
    assert result.covered_ids == []
