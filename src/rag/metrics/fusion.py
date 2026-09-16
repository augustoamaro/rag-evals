from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(rankings: Sequence[Sequence[str]], k_const: int = 60) -> list[str]:
    """Reciprocal Rank Fusion.

    Each item scores ``sum(1 / (k_const + rank))`` across the input rankings
    (rank is 1-based). Items appearing high in several rankings rise to the top;
    items are returned best-first, ties broken by id for determinism.
    """
    return [item for item, _ in rrf_fuse_scored(rankings, k_const)]


def rrf_fuse_scored(
    rankings: Sequence[Sequence[str]], k_const: int = 60
) -> list[tuple[str, float]]:
    """Return the RRF rank sums, best-first with deterministic ties by id."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_const + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
