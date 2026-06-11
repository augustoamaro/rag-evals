from __future__ import annotations

from typing import Protocol

from rag.domain.entities import (
    Answer,
    AnswerScores,
    Chunk,
    Document,
    RetrievedChunk,
    Strategy,
    Usage,
    Vector,
)


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[Vector]: ...


class ChunkStore(Protocol):
    def add_document(self, doc: Document, chunks: list[tuple[Chunk, Vector]]) -> None: ...

    def all_chunks(self) -> list[Chunk]: ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...


class Generator(Protocol):
    def answer(self, question: str, context: list[RetrievedChunk]) -> Answer: ...


class Judge(Protocol):
    def score(
        self,
        question: str,
        answer: Answer,
        context: list[RetrievedChunk],
        reference_answer: str,
    ) -> tuple[AnswerScores, Usage]: ...
