from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from rag.domain.entities import EvalCase, EvalRun

_METRICS = ("recall", "precision", "mrr", "ndcg", "evidence_recall")
_IDENTITY = (
    "schema_version",
    "dataset_sha256",
    "case_set_sha256",
    "split",
    "language",
    "k",
    "retrieval_level",
    "evidence_metric_version",
    "ranking_metric_version",
)
_LABELS = ("split", "language", "answerable", "group_id", "tags", "evidence_ids")


@dataclass(frozen=True)
class ComparisonReport:
    compatible: bool
    failures: list[str]
    metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)
    slices: dict[str, dict[str, dict[str, float | int]]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.compatible and not self.failures


def make_baseline(run: EvalRun) -> dict[str, Any]:
    artifact = {
        "schema_version": 1,
        "run_id": run.id,
        "manifest": run.manifest,
        "metrics": {
            "retrieval": asdict(run.retrieval_metrics),
            "evidence": run.evidence_metrics,
            "slices": run.slices,
        },
        "cases": [
            {
                key: value
                for key, value in asdict(result).items()
                if key
                in (
                    "case_id",
                    *_METRICS,
                    *_LABELS,
                    "retrieved_ids",
                    "covered_evidence_ids",
                )
            }
            for result in run.case_results
        ],
    }
    _validate(artifact)
    return artifact


def compare_baseline(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_drop: float = 0.02,
    slice_max_drop: float = 0.02,
    protected_case_ids: set[str] | None = None,
) -> ComparisonReport:
    """Fail closed on incomparable data; gate paired drops, not CI significance."""
    try:
        for limit in (max_drop, slice_max_drop):
            if not math.isfinite(limit) or not 0 <= limit <= 1:
                raise ValueError("regression tolerances must be finite and in [0, 1]")
        old, new = _validate(baseline), _validate(candidate)
        for key in _IDENTITY:
            if baseline["manifest"][key] != candidate["manifest"][key]:
                raise ValueError(f"incompatible manifest field: {key}")
        if old.keys() != new.keys():
            raise ValueError("baseline and candidate must contain identical case IDs")
        for cid in old:
            for label in _LABELS:
                if old[cid][label] != new[cid][label]:
                    raise ValueError(f"case {cid} has incompatible {label}")
        protected = set(protected_case_ids or ())
        if protected - old.keys():
            raise ValueError(
                "unknown protected case IDs: " + ", ".join(sorted(protected - old.keys()))
            )
        protected.update(cid for cid, row in old.items() if "critical" in row["tags"])
    except (ValueError, TypeError, KeyError) as exc:
        return ComparisonReport(False, [str(exc)])

    eligible = [cid for cid in sorted(old) if old[cid]["answerable"]]
    if not eligible:
        return ComparisonReport(False, ["no answerable cases to compare"])
    failures: list[str] = []
    metrics = _compare_group(old, new, eligible, max_drop, "overall", failures)
    groups: dict[str, list[str]] = {}
    for cid in eligible:
        for tag in [
            f"language:{old[cid]['language']}",
            *[f"tag:{t}" for t in old[cid]["tags"]],
        ]:
            groups.setdefault(tag, []).append(cid)
    slices = {
        tag: _compare_group(old, new, ids, slice_max_drop, tag, failures)
        for tag, ids in sorted(groups.items())
    }
    for cid in sorted(protected):
        if (
            old[cid]["answerable"]
            and new[cid]["evidence_recall"] < old[cid]["evidence_recall"] - 1e-12
        ):
            failures.append(f"protected case {cid}: evidence recall regressed")
    return ComparisonReport(True, failures, metrics, slices)


def validate_baseline_selection(
    baseline: dict[str, Any],
    manifest: dict[str, object],
    cases: list[EvalCase],
) -> None:
    """Reject incompatible baselines before retrieval or paid model calls."""
    records = _validate(baseline)
    for key in _IDENTITY:
        if baseline["manifest"][key] != manifest.get(key):
            raise ValueError(f"incompatible manifest field: {key}")
    if records.keys() != {c.id for c in cases}:
        raise ValueError("baseline and candidate must contain identical case IDs")
    for case in cases:
        labels = asdict(case)
        labels["tags"] = sorted(case.tags)
        labels["evidence_ids"] = sorted(case.evidence_ids)
        for key in _LABELS:
            if labels[key] != records[case.id][key]:
                raise ValueError(f"case {case.id} has incompatible {key}")


def _validate(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported baseline artifact schema")
    manifest = artifact.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("baseline requires a version 2 evaluation manifest")
    for key in _IDENTITY:
        value = manifest.get(key)
        if key in ("schema_version", "k"):
            if type(value) is not int or value < 1 or (key == "k" and value > 50):
                raise ValueError(f"invalid manifest field: {key}")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"missing manifest field: {key}")
    cases = artifact.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("baseline requires nonempty per-case results")
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid case record")
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid or cid in result:
            raise ValueError("case IDs must be nonempty and unique")
        if type(case.get("answerable")) is not bool:
            raise ValueError(f"case {cid} needs an explicit answerable flag")
        for key in ("tags", "evidence_ids"):
            values = case.get(key)
            if not isinstance(values, list) or not all(
                isinstance(v, str) and v for v in values
            ):
                raise ValueError(f"invalid case {cid} {key}")
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate case {cid} {key}")
        for key in ("split", "language", "group_id"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"missing case {cid} {key}")
        if case["split"] != manifest["split"] or case["language"] != manifest["language"]:
            raise ValueError(f"case {cid} does not match the selected split/language")
        if bool(case["evidence_ids"]) != case["answerable"]:
            raise ValueError(f"case {cid} answerability contradicts its evidence")
        for metric in _METRICS:
            value = case.get(metric)
            if metric == "evidence_recall" and not case["answerable"] and value is None:
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"case {cid} has invalid {metric}")
        result[cid] = {
            **case,
            "tags": sorted(case["tags"]),
            "evidence_ids": sorted(case["evidence_ids"]),
        }
    return result


def _compare_group(
    old: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
    ids: list[str],
    tolerance: float,
    label: str,
    failures: list[str],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for metric in _METRICS:
        before = [float(old[cid][metric]) for cid in ids]
        after = [float(new[cid][metric]) for cid in ids]
        delta = [b - a for a, b in zip(before, after, strict=True)]
        mean_delta = math.fsum(delta) / len(delta)
        lo, hi = _paired_interval(delta)
        metrics[metric] = {
            "n": len(ids),
            "baseline": math.fsum(before) / len(ids),
            "candidate": math.fsum(after) / len(ids),
            "mean_delta": mean_delta,
            "ci_low": lo,
            "ci_high": hi,
        }
        if mean_delta < -tolerance - 1e-12:
            failures.append(
                f"{label} {metric}: delta {mean_delta:+.4f} "
                f"exceeds allowed drop {tolerance:.4f}"
            )
    return metrics


def _paired_interval(deltas: list[float]) -> tuple[float, float]:
    if min(deltas) == max(deltas):
        return deltas[0], deltas[0]
    rng = random.Random(20260916)
    n = len(deltas)
    samples = sorted(math.fsum(rng.choices(deltas, k=n)) / n for _ in range(2000))
    return samples[49], samples[1949]
