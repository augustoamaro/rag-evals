from __future__ import annotations

from pydantic import BaseModel


class CitationOut(BaseModel):
    chunk_id: str


class RetrievedChunkOut(BaseModel):
    id: str
    text: str
    score: float
    rank: int


class QueryRequest(BaseModel):
    question: str
    k: int = 10
    strategy: str = "hybrid"


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    chunks: list[RetrievedChunkOut]


class RunRequest(BaseModel):
    strategy: str = "hybrid"
    k: int = 10
    with_answers: bool = False


class RunIdResponse(BaseModel):
    id: str
