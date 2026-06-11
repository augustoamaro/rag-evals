import math

import pytest

from rag.adapters.embedding.fastembed_embedder import FastEmbedEmbedder

pytestmark = pytest.mark.integration


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_embeds_to_fixed_dim_and_is_deterministic() -> None:
    embedder = FastEmbedEmbedder()
    vecs = embedder.embed(["hello world", "hello world", "a different sentence"])
    assert len(vecs) == 3
    assert all(len(v) == embedder.dim for v in vecs)
    assert vecs[0] == vecs[1]  # deterministic for identical input
    assert _cosine(vecs[0], vecs[1]) == pytest.approx(1.0, abs=1e-5)


def test_related_text_is_closer_than_unrelated() -> None:
    embedder = FastEmbedEmbedder()
    cat, kitten, finance = embedder.embed(
        ["a small domestic cat", "a tiny kitten", "quarterly financial report"]
    )
    assert _cosine(cat, kitten) > _cosine(cat, finance)
