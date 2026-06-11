from __future__ import annotations

from rag.domain.entities import RunMetrics


def check_thresholds(
    metrics: RunMetrics, min_recall: float, min_ndcg: float
) -> list[str]:
    """Return human-readable failures when retrieval metrics fall below baselines.

    An empty list means the run passed the regression gate.
    """
    failures: list[str] = []
    if metrics.recall < min_recall:
        failures.append(f"recall {metrics.recall:.3f} < min {min_recall:.3f}")
    if metrics.ndcg < min_ndcg:
        failures.append(f"ndcg {metrics.ndcg:.3f} < min {min_ndcg:.3f}")
    return failures
