from __future__ import annotations

from functools import cached_property

from fastembed import TextEmbedding

from rag.domain.entities import Vector


class FastEmbedEmbedder:
    """Local, key-free embeddings via fastembed (ONNX). Default: bge-small (384-dim)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        self._model_name = model_name
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @cached_property
    def _model(self) -> TextEmbedding:
        return TextEmbedding(model_name=self._model_name)

    def embed(self, texts: list[str]) -> list[Vector]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]
