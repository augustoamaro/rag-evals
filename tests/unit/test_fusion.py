from rag.metrics.fusion import rrf_fuse


def test_rrf_rewards_agreement_across_rankings() -> None:
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = rrf_fuse([dense, sparse], k_const=60)
    assert set(fused[:2]) == {"a", "b"}
    assert fused[-1] in {"c", "d"}


def test_rrf_single_ranking_preserves_order() -> None:
    assert rrf_fuse([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_empty() -> None:
    assert rrf_fuse([]) == []
