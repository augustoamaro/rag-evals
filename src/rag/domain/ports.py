from __future__ import annotations

from typing import Protocol, runtime_checkable

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


class ChunkReader(Protocol):
    def all_chunks(self) -> list[Chunk]: ...


class ChunkStore(ChunkReader, Protocol):
    def add_document(self, doc: Document, chunks: list[tuple[Chunk, Vector]]) -> None: ...

    def replace_corpus(
        self, documents: list[tuple[Document, list[tuple[Chunk, Vector]]]]
    ) -> None: ...


class Retriever(Protocol):
    def retrieve(
        self, query: str, k: int, strategy: Strategy
    ) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...


@runtime_checkable
class ScoredReranker(Reranker, Protocol):
    """Optional access to cross-encoder scores in the same reranked order."""

    def rerank_with_scores(
        self, query: str, chunks: list[Chunk]
    ) -> list[tuple[Chunk, float]]: ...


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
