from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from rag.services.run_comparison import compare_runs
from tests.unit.test_comparison import _artifact


def stored(values: list[float]) -> dict[str, Any]:
    artifact = _artifact(values)
    return {
        "id": "run",
        "config": {**artifact["manifest"], "case_count": len(values)},
        "cases": artifact["cases"],
        "evaluation": {"status": "completed", "retrieval_complete": True},
    }


def test_comparison_exposes_paired_case_deltas_and_slice_gate() -> None:
    old, new = stored([1, 0.5]), stored([1, 0])
    new["id"] = "candidate"
    report = compare_runs(old, new)
    assert report["compatible"] and not report["passed"]
    assert report["candidate_id"] == "candidate"
    assert report["metrics"]["recall"]["mean_delta"] == -0.25
    assert report["cases"][1]["metrics"]["recall"] == {
        "baseline": 0.5,
        "candidate": 0,
        "delta": -0.5,
    }
    assert report["slices"]["tag:ordinary"]["recall"]["n"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "running",
        "cancelled",
        "prefix",
        "unknown_count",
        "retrieval_failed",
        "language",
        "legacy",
    ],
)
def test_comparison_rejects_incomplete_or_incompatible_run(mutation: str) -> None:
    old = stored([1, 1])
    new = deepcopy(old)
    if mutation in ("running", "cancelled"):
        new["evaluation"]["status"] = mutation
    elif mutation == "prefix":
        # Even two identical prefixes must not compare as a full benchmark.
        old["cases"].pop()
        new["cases"].pop()
    elif mutation == "unknown_count":
        new["config"].pop("case_count")
    elif mutation == "retrieval_failed":
        new["evaluation"]["retrieval_complete"] = False
    elif mutation == "language":
        new["config"]["language"] = "pt"
    else:
        new["config"]["schema_version"] = 1
    report = compare_runs(old, new)
    assert not report["compatible"] and report["failures"]
    assert report["metrics"] == {} and report["cases"] == []


def test_complete_old_runs_and_failed_answer_runs_can_compare_retrieval() -> None:
    old, new = stored([1]), stored([1])
    old["evaluation"].pop("retrieval_complete")
    new["evaluation"]["status"] = "failed"
    assert compare_runs(old, new)["passed"]


def test_unanswerable_case_is_explicitly_not_scored() -> None:
    old, new = stored([1, 0]), stored([1, 0])
    for run in (old, new):
        run["cases"][1].update(answerable=False, evidence_ids=[], evidence_recall=None)
    result = compare_runs(old, new)
    assert result["compatible"]
    assert result["cases"][1]["metrics"] == {}
    assert result["metrics"]["recall"]["n"] == 1
