from __future__ import annotations

from dataclasses import dataclass

from rag.domain.datasets import EvidenceUnit
from rag.domain.entities import RetrievedChunk


@dataclass(frozen=True)
class EvidenceScore:
    recall: float | None
    covered_ids: list[str]


def score_evidence(
    retrieved: list[RetrievedChunk], evidence: list[EvidenceUnit]
) -> EvidenceScore:
    """Full evidence coverage over the union of retrieved word intervals.

    Word offsets refer to the canonical document's whitespace-split tokens.
    Document-level qrels use units without offsets. No positives means the
    retrieval-quality score is not applicable, rather than a synthetic zero/one.
    """
    if len({e.id for e in evidence}) != len(evidence):
        raise ValueError("duplicate evidence IDs")
    covered = sorted(e.id for e in evidence if _covered(e, retrieved))
    return EvidenceScore(len(covered) / len(evidence) if evidence else None, covered)


def _covered(evidence: EvidenceUnit, retrieved: list[RetrievedChunk]) -> bool:
    chunks = [
        rc.chunk for rc in retrieved if rc.chunk.document_id == evidence.document_id
    ]
    if evidence.word_start is None and evidence.word_end is None:
        return bool(chunks)
    if evidence.word_start is None or evidence.word_end is None:
        raise ValueError("evidence requires both word offsets")
    intervals: list[tuple[int, int]] = []
    for chunk in chunks:
        if chunk.word_start is None or chunk.word_end is None:
            raise ValueError("span evidence requires indexed chunk word offsets")
        intervals.append((chunk.word_start, chunk.word_end))
    position = evidence.word_start
    for start, end in sorted(intervals):
        if end <= position:
            continue
        if start > position:
            break
        position = max(position, end)
        if position >= evidence.word_end:
            return True
    return False
