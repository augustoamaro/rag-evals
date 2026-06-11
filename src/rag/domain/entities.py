from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

Vector = list[float]


class Strategy(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    title: str
    content: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    token_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    relevant_snippets: list[str]
    reference_answer: str


@dataclass(frozen=True)
class AnswerScores:
    faithfulness: float
    relevance: float
    citation_correctness: float
    rationale: str
