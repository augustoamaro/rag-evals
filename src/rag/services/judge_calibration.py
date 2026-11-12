"""Compare the LLM judge with human labels on a fixed calibration set.

A set holds real generated answers and deliberately defective ones. A human rates
each on the judge's three dimensions using three levels; judge scores are mapped
onto the same levels so disagreements have a direction: a judge that gives full
marks where the human saw a problem is lenient.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any

from rag.domain.entities import Answer, Chunk, Citation, RetrievedChunk
from rag.domain.ports import Judge
from rag.metrics.citations import split_citations
from rag.services.pricing import cost_usd

logger = logging.getLogger(__name__)

DIMENSIONS = ("faithfulness", "relevance", "citation_correctness")
LEVELS = (0.0, 0.5, 1.0)


def to_level(score: float) -> float:
    """Map a judge score in [0, 1] onto the three-level human scale."""
    if score < 0.25:
        return 0.0
    if score < 0.75:
        return 0.5
    return 1.0


def validate_set(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported calibration set schema")
    items = artifact.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("calibration set requires nonempty items")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid item record")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("item ids must be nonempty strings")
        if item_id in seen:
            raise ValueError(f"duplicate item id: {item_id!r}")
        seen.add(item_id)
        for key in ("case_id", "question", "answer"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"item {item_id} needs a nonempty {key}")
        if not isinstance(item.get("reference_answer"), str):
            raise ValueError(f"item {item_id} needs a reference_answer string")
        kind, defect = item.get("kind"), item.get("defect")
        generated = kind == "generated" and defect is None
        defective = kind == "defect" and isinstance(defect, str) and bool(defect)
        if not generated and not defective:
            raise ValueError(f"item {item_id} has an inconsistent kind/defect")
        expected = item.get("expected_low")
        if expected is not None and (
            not defective
            or not isinstance(expected, list)
            or not expected
            or any(dimension not in DIMENSIONS for dimension in expected)
            or len(set(expected)) != len(expected)
        ):
            raise ValueError(
                f"item {item_id} expected_low must list distinct dimensions of a defect"
            )
        context = item.get("context")
        if not isinstance(context, list) or not context:
            raise ValueError(f"item {item_id} needs a nonempty context")
        passage_ids: list[str] = []
        for passage in context:
            if not isinstance(passage, dict) or not all(
                isinstance(passage.get(key), str) and passage[key]
                for key in ("id", "document_id", "text")
            ):
                raise ValueError(f"item {item_id} has an invalid context passage")
            passage_ids.append(passage["id"])
        if len(set(passage_ids)) != len(passage_ids):
            raise ValueError(f"item {item_id} has duplicate context ids")
    return items


def merge_labels(artifact: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the set with one complete human label per item."""
    known = {item["id"] for item in validate_set(artifact)}
    if not isinstance(labels, dict):
        raise ValueError("labels must be an object keyed by item id")
    unknown, missing = sorted(labels.keys() - known), sorted(known - labels.keys())
    if unknown:
        raise ValueError("labels for unknown items: " + ", ".join(unknown))
    if missing:
        raise ValueError("items without labels: " + ", ".join(missing))
    merged = deepcopy(artifact)
    for item in merged["items"]:
        label = labels[item["id"]]
        if not isinstance(label, dict):
            raise ValueError(f"item {item['id']} label must be an object")
        human: dict[str, float | str] = {}
        for dimension in DIMENSIONS:
            value = label.get(dimension)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) not in LEVELS
            ):
                raise ValueError(f"item {item['id']} {dimension} must be 0, 0.5 or 1")
            human[dimension] = float(value)
        note = label.get("note", "")
        if not isinstance(note, str):
            raise ValueError(f"item {item['id']} note must be a string")
        human["note"] = note
        item["human"] = human
    return merged


def score_items(artifact: dict[str, Any], judge: Judge, model: str) -> dict[str, Any]:
    """Return a copy of the set with the judge's result recorded on every item."""
    validate_set(artifact)
    scored = deepcopy(artifact)
    for item in scored["items"]:
        context = [
            RetrievedChunk(
                chunk=Chunk(
                    id=passage["id"],
                    document_id=passage["document_id"],
                    ordinal=0,
                    text=passage["text"],
                    token_count=len(passage["text"].split()),
                ),
                score=None,
                rank=rank,
            )
            for rank, passage in enumerate(item["context"], start=1)
        ]
        # Same split the evaluation path applies before the judge sees an answer.
        valid, invalid = split_citations(
            item["answer"], {passage["id"] for passage in item["context"]}
        )
        answer = Answer(
            text=item["answer"],
            citations=[Citation(chunk_id=cid) for cid in valid],
            invalid_citations=invalid,
        )
        try:
            scores, usage = judge.score(
                item["question"], answer, context, item["reference_answer"]
            )
        except Exception as exc:
            # Provider messages may contain prompts, keys or other private data.
            error_type = type(exc).__name__
            logger.warning("judge calibration call failed (%s)", error_type)
            item["judge"] = {"error_type": error_type}
            continue
        item["judge"] = {
            "faithfulness": scores.faithfulness,
            "relevance": scores.relevance,
            "citation_correctness": scores.citation_correctness,
            "rationale": scores.rationale,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost_usd(model, usage),
        }
    return scored


def agreement(artifact: dict[str, Any]) -> dict[str, Any]:
    """Level agreement between judge and human, overall and per item group."""
    items = validate_set(artifact)
    unlabelled = sum(not _complete(item.get("human"), LEVELS) for item in items)
    unscored = sum(not _complete(item.get("judge"), None) for item in items)
    if unlabelled or unscored:
        raise ValueError(
            f"calibration set is incomplete: {unlabelled} without human labels, "
            f"{unscored} without judge scores"
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(_group(item), []).append(item)
    disagreements = [
        {
            "id": item["id"],
            "case_id": item["case_id"],
            "group": _group(item),
            "dimension": dimension,
            "human": item["human"][dimension],
            "judge": float(item["judge"][dimension]),
            "judge_level": to_level(float(item["judge"][dimension])),
        }
        for item in sorted(items, key=lambda item: item["id"])
        for dimension in DIMENSIONS
        if to_level(float(item["judge"][dimension])) != item["human"][dimension]
    ]
    return {
        "items": len(items),
        "overall": _summarize(items),
        "groups": {name: _summarize(members) for name, members in sorted(groups.items())},
        "disagreements": disagreements,
    }


def defect_detection(artifact: dict[str, Any]) -> dict[str, Any]:
    """How many planted defects the judge scored down, without human labels.

    A defect item names the dimensions it was built to lower (``expected_low``).
    It is detected on a dimension when the judge leaves full marks there, i.e.
    its score maps below level 1. Untouched answers scored below level 1 are the
    judge's false alarms.
    """
    items = validate_set(artifact)
    unscored = sum(not _complete(item.get("judge"), None) for item in items)
    if unscored:
        raise ValueError(
            f"calibration set is incomplete: {unscored} without judge scores"
        )
    planted = [item for item in items if item.get("expected_low")]
    if not planted:
        raise ValueError("calibration set has no planted defects with expected_low")
    clean = {item["case_id"]: item for item in items if item["kind"] == "generated"}

    rows: list[dict[str, Any]] = []
    by_defect: dict[str, dict[str, Any]] = {}
    for item in sorted(planted, key=lambda item: item["id"]):
        lowered = [
            dimension
            for dimension in DIMENSIONS
            if to_level(float(item["judge"][dimension])) < 1
        ]
        summary = by_defect.setdefault(
            item["defect"], {"n": 0, "detected": 0, "dimensions": []}
        )
        for dimension in item["expected_low"]:
            score = float(item["judge"][dimension])
            original = clean.get(item["case_id"])
            rows.append(
                {
                    "id": item["id"],
                    "case_id": item["case_id"],
                    "defect": item["defect"],
                    "dimension": dimension,
                    "judge": score,
                    "judge_level": to_level(score),
                    "detected": dimension in lowered,
                    "clean_judge": float(original["judge"][dimension])
                    if original is not None
                    else None,
                    "also_lowered": [
                        other for other in lowered if other not in item["expected_low"]
                    ],
                }
            )
            summary["n"] += 1
            summary["detected"] += dimension in lowered
            if dimension not in summary["dimensions"]:
                summary["dimensions"].append(dimension)
    flagged = [
        {
            "id": item["id"],
            "case_id": item["case_id"],
            "dimension": dimension,
            "judge": float(item["judge"][dimension]),
            "judge_level": to_level(float(item["judge"][dimension])),
        }
        for item in sorted(clean.values(), key=lambda item: item["id"])
        for dimension in DIMENSIONS
        if to_level(float(item["judge"][dimension])) < 1
    ]
    return {
        "defects": len(rows),
        "detected": sum(row["detected"] for row in rows),
        "by_defect": dict(sorted(by_defect.items())),
        "items": rows,
        "clean": {"n": len(clean), "flagged": flagged},
    }


def _group(item: dict[str, Any]) -> str:
    return "generated" if item["kind"] == "generated" else f"defect:{item['defect']}"


def _complete(scores: object, allowed: tuple[float, ...] | None) -> bool:
    if not isinstance(scores, dict):
        return False
    for dimension in DIMENSIONS:
        value = scores.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value) or not 0 <= value <= 1:
            return False
        if allowed is not None and float(value) not in allowed:
            return False
    return True


def _summarize(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        human = [float(item["human"][dimension]) for item in items]
        judge = [float(item["judge"][dimension]) for item in items]
        levels = [to_level(score) for score in judge]
        pairs = list(zip(human, levels, strict=True))
        summary[dimension] = {
            "n": len(items),
            "same_level": sum(h == level for h, level in pairs),
            "mean_abs_diff": math.fsum(
                abs(j - h) for h, j in zip(human, judge, strict=True)
            )
            / len(items),
            "human_mean": math.fsum(human) / len(items),
            "judge_mean": math.fsum(judge) / len(items),
            # A flag is a level below 1: the rater saw a problem.
            "flags": {
                "both": sum(h < 1 and level < 1 for h, level in pairs),
                "human_only": sum(h < 1 and level == 1 for h, level in pairs),
                "judge_only": sum(h == 1 and level < 1 for h, level in pairs),
                "neither": sum(h == 1 and level == 1 for h, level in pairs),
            },
        }
    return summary
