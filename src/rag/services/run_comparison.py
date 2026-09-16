from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rag.services.comparison import compare_baseline

_METRICS = ("recall", "precision", "mrr", "ndcg", "evidence_recall")


def compare_runs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare stored retrieval observations; never treat a prefix as a full run."""
    response: dict[str, Any] = {
        "baseline_id": baseline["id"],
        "candidate_id": candidate["id"],
        "compatible": False,
        "passed": False,
        "failures": [],
        "metrics": {},
        "slices": {},
        "cases": [],
    }
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        config = run.get("config") or {}
        evaluation = run.get("evaluation") or {}
        count = config.get("case_count")
        if (
            type(count) is not int
            or count < 1
            or count != len(run.get("cases", []))
            or evaluation.get("retrieval_complete") is False
            or evaluation.get("status") in ("running", "cancelled")
            or evaluation.get("error_type") is not None
        ):
            response["failures"].append(
                f"{label}: complete retrieval coverage is required"
            )
    if response["failures"]:
        return response

    report = compare_baseline(
        {"schema_version": 1, "manifest": baseline["config"], "cases": baseline["cases"]},
        {
            "schema_version": 1,
            "manifest": candidate["config"],
            "cases": candidate["cases"],
        },
    )
    response.update(asdict(report), passed=report.passed)
    if report.compatible:
        before = {row["case_id"]: row for row in baseline["cases"]}
        after = {row["case_id"]: row for row in candidate["cases"]}
        response["cases"] = [
            {
                "case_id": cid,
                "question": after[cid].get("question") or "",
                "answerable": after[cid]["answerable"],
                "tags": after[cid]["tags"],
                "metrics": {
                    metric: {
                        "baseline": before[cid][metric],
                        "candidate": after[cid][metric],
                        "delta": after[cid][metric] - before[cid][metric],
                    }
                    for metric in _METRICS
                }
                if after[cid]["answerable"]
                else {},
            }
            for cid in sorted(before)
        ]
    return response
