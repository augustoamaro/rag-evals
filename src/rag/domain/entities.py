from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

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
    word_start: int | None = None
    word_end: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float | None
    rank: int
    score_type: str | None = None
    score_components: dict[str, float] = field(default_factory=dict)


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
    evidence_ids: list[str] = field(default_factory=list)
    split: str = "smoke"
    tags: list[str] = field(default_factory=list)
    language: str = "en"
    answerable: bool = True
    group_id: str = ""


@dataclass(frozen=True)
class AnswerScores:
    faithfulness: float
    relevance: float
    citation_correctness: float
    rationale: str


@dataclass(frozen=True)
class RunMetrics:
    recall: float
    precision: float
    mrr: float
    ndcg: float


@dataclass(frozen=True)
class StageResult:
    status: Literal["skipped", "succeeded", "failed"] = "skipped"
    usage: Usage | None = None
    cost_usd: float | None = 0.0
    latency_ms: int = 0
    error_type: str | None = None

    @property
    def cost_complete(self) -> bool:
        return self.status == "skipped" or self.cost_usd is not None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    strategy: str
    retrieved_ids: list[str]
    recall: float
    precision: float
    mrr: float
    ndcg: float
    answer: str | None = None
    judge_scores: AnswerScores | None = None
    cost_usd: float = 0.0
    latency_ms: int = 0  # retrieval only
    answer_ms: int = 0  # generation + judging (0 when the answer track is off)
    generation: StageResult = field(default_factory=StageResult)
    judging: StageResult = field(default_factory=StageResult)
    cost_complete: bool = True
    question: str = ""
    reference_answer: str = ""
    retrieved_context: list[RetrievedChunk] = field(default_factory=list)
    evidence_recall: float | None = None
    covered_evidence_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    split: str = "smoke"
    tags: list[str] = field(default_factory=list)
    language: str = "en"
    answerable: bool = True
    group_id: str = ""


@dataclass(frozen=True)
class EvalRun:
    id: str
    strategy: Strategy
    k: int
    retrieval_metrics: RunMetrics
    case_results: list[CaseResult]
    answer_metrics: dict[str, float] | None = None
    cost_usd: float = 0.0
    latency_p50_ms: int = 0
    latency_p95_ms: int = 0
    with_answers: bool = False
    status: Literal["running", "completed", "partial", "failed", "cancelled"] = (
        "completed"
    )
    answer_attempted: int = 0
    answer_scored: int = 0
    answer_failed: int = 0
    cost_complete: bool = True
    manifest: dict[str, object] = field(default_factory=dict)
    evidence_metrics: dict[str, float | int | None] | None = None
    slices: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    retrieval_complete: bool = True
    error_type: str | None = None
