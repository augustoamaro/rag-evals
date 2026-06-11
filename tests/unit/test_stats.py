from rag.metrics.stats import mean, percentile


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0


def test_percentile() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 20.0
    assert percentile(values, 95) == 40.0
    assert percentile(values, 100) == 40.0


def test_percentile_empty() -> None:
    assert percentile([], 50) == 0.0
