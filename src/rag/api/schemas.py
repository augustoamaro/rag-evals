from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    chunk_id: str


class RetrievedChunkOut(BaseModel):
    id: str
    text: str
    score: float | None
    rank: int
    score_type: str | None = None
    score_components: dict[str, float] = Field(default_factory=dict)
    document_id: str | None = None
    word_start: int | None = None
    word_end: int | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=50)
    strategy: str = "hybrid"


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    chunks: list[RetrievedChunkOut]


class RunRequest(BaseModel):
    """Evaluate cases against the immutable index bound to this service at startup."""

    strategy: str = "hybrid"
    k: int = Field(default=10, ge=1, le=50)
    with_answers: bool = False
    split: str | None = Field(
        default=None,
        description=(
            "Split in the stored dataset; omitted uses its default split. The service "
            "is bound to one immutable index. Select an index with RAG_INDEX_ID "
            "at startup."
        ),
    )
    language: str = Field(
        default="en",
        min_length=1,
        pattern=r"\S",
        description=(
            "Exact case language to evaluate within the selected split. Defaults to "
            "English (en); use pt for the separate Portuguese diagnostic slice."
        ),
    )


class RunIdResponse(BaseModel):
    id: str


class JobRequest(RunRequest):
    request_id: UUID


class AnnotationRequest(BaseModel):
    case_id: str | None = Field(default=None, min_length=1)
    note: str = Field(max_length=4000)
    revision: int = Field(ge=0)
