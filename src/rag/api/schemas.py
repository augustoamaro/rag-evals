from __future__ import annotations

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    chunk_id: str


class RetrievedChunkOut(BaseModel):
    id: str
    text: str
    score: float
    rank: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=50)
    strategy: str = "hybrid"


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    chunks: list[RetrievedChunkOut]


class RunRequest(BaseModel):
    strategy: str = "hybrid"
    k: int = Field(default=10, ge=1, le=50)
    with_answers: bool = False


class RunIdResponse(BaseModel):
    id: str
