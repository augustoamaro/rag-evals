from __future__ import annotations

from collections.abc import Iterable

import pytest

from rag.adapters.retrieval.reranker import FastEmbedReranker
from rag.domain.entities import Chunk


class FixedCrossEncoder:
    def rerank(self, query: str, documents: Iterable[str]) -> Iterable[float]:
        scores = {"first": -1.5, "second": 4.25, "third": 4.25}
        return (scores[text] for text in documents)


def test_fastembed_exposes_raw_scores_and_stable_ties_through_both_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reranker = FastEmbedReranker()
    monkeypatch.setattr(
        FastEmbedReranker, "_model", property(lambda _: FixedCrossEncoder())
    )
    chunks = [
        Chunk(str(i), "doc", i, text, 1)
        for i, text in enumerate(["first", "second", "third"])
    ]

    scored = reranker.rerank_with_scores("question", chunks)

    assert [(chunk.id, score) for chunk, score in scored] == [
        ("1", 4.25),
        ("2", 4.25),
        ("0", -1.5),
    ]
    assert [chunk.id for chunk in reranker.rerank("question", chunks)] == ["1", "2", "0"]


def test_empty_candidates_do_not_load_the_cross_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_model(self: FastEmbedReranker) -> None:
        raise AssertionError("empty candidates do not need a model")

    monkeypatch.setattr(FastEmbedReranker, "_model", property(unavailable_model))
    reranker = FastEmbedReranker()

    assert reranker.rerank_with_scores("question", []) == []
    assert reranker.rerank("question", []) == []
