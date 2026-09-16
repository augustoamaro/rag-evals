from __future__ import annotations

from dataclasses import dataclass

from rag.domain.datasets import Dataset


@dataclass(frozen=True)
class IndexSnapshot:
    id: str
    dataset: Dataset
    embedding: dict[str, object]
    chunking: dict[str, object]
