from __future__ import annotations

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (``p`` in [0, 100]). Returns 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]
