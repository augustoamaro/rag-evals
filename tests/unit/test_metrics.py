import math

import pytest

from rag.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k

RANKED = ["a", "b", "c", "d"]  # ranked retrieved ids (best first)
REL = {"b", "d", "z"}  # z is relevant but not retrieved


def test_recall_at_k() -> None:
    # top-3 = a,b,c → relevant retrieved = {b}; total relevant = 3
    assert recall_at_k(RANKED, REL, 3) == 1 / 3
    assert recall_at_k(RANKED, REL, 4) == 2 / 3


def test_recall_empty_relevant_is_zero() -> None:
    assert recall_at_k(RANKED, set(), 3) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(RANKED, REL, 4) == 2 / 4


def test_mrr_first_relevant_at_rank_2() -> None:
    assert mrr(RANKED, REL) == 1 / 2


def test_mrr_no_relevant_is_zero() -> None:
    assert mrr(RANKED, {"q"}) == 0.0


def test_ndcg_at_k() -> None:
    # rel vector over a,b,c,d = [0,1,0,1] → hits at 0-indexed ranks 1 and 3.
    # |REL| = 3 (b, d, z) so the ideal ranking has 3 relevant items at the front.
    dcg = 1 / math.log2(1 + 2) + 1 / math.log2(3 + 2)
    idcg = sum(1 / math.log2(i + 2) for i in range(3))
    assert ndcg_at_k(RANKED, REL, 4) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["b", "d"], {"b", "d"}, 2) == pytest.approx(1.0)
