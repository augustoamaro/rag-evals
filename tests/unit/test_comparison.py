from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from rag.services.comparison import compare_baseline


def _artifact(values: list[float], *, critical: bool = False) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest": {
            "schema_version": 2,
            "dataset_sha256": "dataset",
            "case_set_sha256": "cases",
            "split": "test",
            "language": "en",
            "k": 3,
            "retrieval_level": "chunk",
            "evidence_metric_version": "span-union-v1",
            "ranking_metric_version": "binary-v1",
            "index_id": "idx-baseline",
        },
        "cases": [
            {
                "case_id": f"q{i}",
                "recall": v,
                "precision": v,
                "mrr": v,
                "ndcg": v,
                "evidence_recall": v,
                "tags": ["critical"] if critical and i == 0 else ["ordinary"],
                "language": "en",
                "split": "test",
                "answerable": True,
                "evidence_ids": [f"e{i}"],
                "group_id": f"g{i}",
            }
            for i, v in enumerate(values)
        ],
    }


def test_regression_above_absolute_floor_fails_paired_gate() -> None:
    report = compare_baseline(
        _artifact([0.99] * 10), _artifact([0.91] * 10), max_drop=0.02
    )
    assert report.compatible
    assert not report.passed
    assert report.metrics["recall"]["mean_delta"] == pytest.approx(-0.08)
    assert report.metrics["recall"]["ci_low"] == pytest.approx(-0.08)
    assert report.metrics["recall"]["ci_high"] == pytest.approx(-0.08)


def test_small_slice_cannot_hide_behind_improved_aggregate() -> None:
    baseline = _artifact([1, 0, 0, 0])
    candidate = _artifact([0, 1, 1, 1])
    baseline["cases"][0]["tags"] = candidate["cases"][0]["tags"] = ["hard"]
    report = compare_baseline(baseline, candidate, max_drop=0.02)
    assert report.metrics["recall"]["mean_delta"] > 0
    assert not report.passed
    assert any("tag:hard" in failure for failure in report.failures)


def test_critical_case_fails_even_with_permissive_aggregate_tolerance() -> None:
    report = compare_baseline(
        _artifact([1, 1], critical=True),
        _artifact([0.5, 1], critical=True),
        max_drop=1,
        slice_max_drop=1,
    )
    assert not report.passed
    assert any("protected case q0" in failure for failure in report.failures)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "nan", "out_of_range", "labels", "dataset", "k", "legacy"],
)
def test_invalid_or_incompatible_artifacts_fail_closed(mutation: str) -> None:
    baseline = _artifact([1, 0.5])
    candidate = deepcopy(baseline)
    if mutation == "missing":
        candidate["cases"].pop()
    elif mutation == "duplicate":
        candidate["cases"].append(candidate["cases"][0])
    elif mutation == "nan":
        candidate["cases"][0]["recall"] = float("nan")
    elif mutation == "out_of_range":
        candidate["cases"][0]["ndcg"] = 1.63
    elif mutation == "labels":
        candidate["cases"][0]["evidence_ids"] = ["other"]
    elif mutation == "dataset":
        candidate["manifest"]["dataset_sha256"] = "different"
    elif mutation == "k":
        candidate["manifest"]["k"] = 10
    else:
        candidate["manifest"]["schema_version"] = 1
    report = compare_baseline(baseline, candidate)
    assert not report.compatible
    assert not report.passed
    assert report.failures


def test_reordered_identical_cases_with_new_index_are_comparable() -> None:
    baseline = _artifact([1, 0.5])
    candidate = deepcopy(baseline)
    candidate["cases"].reverse()
    candidate["manifest"]["index_id"] = "idx-new-model-or-chunking"
    report = compare_baseline(baseline, candidate)
    assert report.passed
    assert report.metrics["recall"]["mean_delta"] == 0


def test_unknown_protected_case_is_configuration_error() -> None:
    report = compare_baseline(_artifact([1]), _artifact([1]), protected_case_ids={"typo"})
    assert not report.passed
    assert not report.compatible
