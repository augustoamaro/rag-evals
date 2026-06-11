from __future__ import annotations

from rag.domain.entities import Answer, RetrievedChunk, Strategy
from rag.domain.ports import Generator, Retriever


class QueryService:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(
        self, question: str, k: int, strategy: Strategy
    ) -> tuple[Answer, list[RetrievedChunk]]:
        context = self._retriever.retrieve(question, k, strategy)
        return self._generator.answer(question, context), context
