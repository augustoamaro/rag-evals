import pytest

from rag.metrics import fusion


def test_rrf_exposes_the_same_rank_sum_that_orders_results() -> None:
    # b has ranks 1 and 2; a has ranks 2 and 1. Their equal sums tie by ID.
    scored = fusion.rrf_fuse_scored([["b", "a", "c"], ["a", "b", "d"]])

    assert [item for item, _ in scored] == ["a", "b", "c", "d"]
    assert [score for _, score in scored] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61 + 1 / 62, 1 / 63, 1 / 63]
    )
    assert fusion.rrf_fuse([["b", "a", "c"], ["a", "b", "d"]]) == [
        "a",
        "b",
        "c",
        "d",
    ]
