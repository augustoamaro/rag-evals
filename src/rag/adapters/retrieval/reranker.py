from __future__ import annotations

from functools import cached_property

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.domain.entities import Chunk


class FastEmbedReranker:
    """Local cross-encoder reranker (no API key) reordering candidates by relevance."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name

    @cached_property
    def _model(self) -> TextCrossEncoder:
        return TextCrossEncoder(model_name=self._model_name)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        scores = list(self._model.rerank(query, [c.text for c in chunks]))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order]
