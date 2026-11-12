from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from rag.domain.entities import Answer, AnswerScores, RetrievedChunk, Usage
from rag.services.judge_calibration import (
    agreement,
    defect_detection,
    merge_labels,
    score_items,
    to_level,
    validate_set,
)


def _item(
    item_id: str,
    *,
    kind: str = "generated",
    defect: str | None = None,
    human: tuple[float, float, float] | None = None,
    judge: tuple[float, float, float] | None = None,
    answer: str = "An answer [doc-a:0].",
) -> dict[str, Any]:
    names = ("faithfulness", "relevance", "citation_correctness")
    return {
        "id": item_id,
        "case_id": f"case-{item_id}",
        "kind": kind,
        "defect": defect,
        "question": "What is asked?",
        "reference_answer": "The reference.",
        "context": [
            {"id": "doc-a:0", "document_id": "doc-a", "text": "First passage."},
            {"id": "doc-b:0", "document_id": "doc-b", "text": "Second passage."},
        ],
        "answer": answer,
        "human": dict(zip(names, human, strict=True)) | {"note": ""} if human else None,
        "judge": dict(zip(names, judge, strict=True)) | {"rationale": "r"}
        if judge
        else None,
    }


def _set(*items: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "name": "test-set", "items": list(items)}


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (0.0, 0.0),
        (0.24, 0.0),
        (0.25, 0.5),
        (0.5, 0.5),
        (0.74, 0.5),
        (0.75, 1.0),
        (1.0, 1.0),
    ],
)
def test_judge_scores_map_onto_the_three_level_human_scale(
    score: float, level: float
) -> None:
    assert to_level(score) == level


def test_agreement_counts_levels_flags_and_differences() -> None:
    # Hand-computed. A "flag" is a level below 1: the rater saw a problem.
    report = agreement(
        _set(
            _item("g1", human=(1, 1, 1), judge=(1.0, 0.95, 1.0)),
            _item("g2", human=(1, 0.5, 1), judge=(0.9, 1.0, 0.6)),
            _item(
                "d1",
                kind="defect",
                defect="wrong_passage",
                human=(1, 1, 0),
                judge=(1.0, 1.0, 0.2),
            ),
            _item(
                "d2",
                kind="defect",
                defect="other_question",
                human=(1, 0, 1),
                judge=(1.0, 1.0, 1.0),
            ),
        )
    )

    assert report["items"] == 4
    relevance = report["overall"]["relevance"]
    assert relevance["n"] == 4
    assert relevance["same_level"] == 2
    assert relevance["mean_abs_diff"] == pytest.approx((0.05 + 0.5 + 0.0 + 1.0) / 4)
    assert relevance["human_mean"] == pytest.approx(0.625)
    assert relevance["judge_mean"] == pytest.approx(0.9875)
    # Both misses are the judge giving full marks where the human saw a problem.
    assert relevance["flags"] == {
        "both": 0,
        "human_only": 2,
        "judge_only": 0,
        "neither": 2,
    }

    citations = report["overall"]["citation_correctness"]
    assert citations["same_level"] == 3
    assert citations["mean_abs_diff"] == pytest.approx((0.0 + 0.4 + 0.2 + 0.0) / 4)
    assert citations["flags"] == {
        "both": 1,
        "human_only": 0,
        "judge_only": 1,
        "neither": 2,
    }

    assert report["overall"]["faithfulness"]["same_level"] == 4
    assert set(report["groups"]) == {
        "generated",
        "defect:wrong_passage",
        "defect:other_question",
    }
    assert report["groups"]["generated"]["relevance"]["n"] == 2
    assert report["groups"]["defect:other_question"]["relevance"]["flags"] == {
        "both": 0,
        "human_only": 1,
        "judge_only": 0,
        "neither": 0,
    }
    assert [(d["id"], d["dimension"]) for d in report["disagreements"]] == [
        ("d2", "relevance"),
        ("g2", "relevance"),
        ("g2", "citation_correctness"),
    ]
    assert report["disagreements"][0] == {
        "id": "d2",
        "case_id": "case-d2",
        "group": "defect:other_question",
        "dimension": "relevance",
        "human": 0.0,
        "judge": 1.0,
        "judge_level": 1.0,
    }


def test_agreement_requires_every_item_to_be_labelled_and_scored() -> None:
    unlabelled = _set(_item("g1", judge=(1, 1, 1)), _item("g2", human=(1, 1, 1)))
    with pytest.raises(ValueError, match="1 without human labels, 1 without judge"):
        agreement(unlabelled)


def test_a_failed_judge_call_blocks_agreement_instead_of_counting_as_a_score() -> None:
    item = _item("g1", human=(1, 1, 1))
    item["judge"] = {"error_type": "RateLimitError"}
    with pytest.raises(ValueError, match="1 without judge scores"):
        agreement(_set(item))


def test_merge_labels_fills_every_item_without_mutating_the_input() -> None:
    original = _set(_item("g1"), _item("g2"))
    snapshot = deepcopy(original)
    labels = {
        "g1": {"faithfulness": 1, "relevance": 0.5, "citation_correctness": 0},
        "g2": {
            "faithfulness": 0,
            "relevance": 1,
            "citation_correctness": 1,
            "note": "adds a claim",
        },
    }
    merged = merge_labels(original, labels)

    assert original == snapshot
    assert merged["items"][0]["human"] == {
        "faithfulness": 1.0,
        "relevance": 0.5,
        "citation_correctness": 0.0,
        "note": "",
    }
    assert merged["items"][1]["human"]["note"] == "adds a claim"


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({"g1": {"faithfulness": 1, "relevance": 1, "citation_correctness": 1}}, "g2"),
        (
            {
                "g1": {"faithfulness": 1, "relevance": 1, "citation_correctness": 1},
                "g2": {"faithfulness": 1, "relevance": 1, "citation_correctness": 1},
                "zz": {"faithfulness": 1, "relevance": 1, "citation_correctness": 1},
            },
            "zz",
        ),
        (
            {
                "g1": {"faithfulness": 0.7, "relevance": 1, "citation_correctness": 1},
                "g2": {"faithfulness": 1, "relevance": 1, "citation_correctness": 1},
            },
            "0, 0.5 or 1",
        ),
        (
            {
                "g1": {"faithfulness": 1, "relevance": 1},
                "g2": {"faithfulness": 1, "relevance": 1, "citation_correctness": 1},
            },
            "citation_correctness",
        ),
    ],
)
def test_merge_labels_rejects_incomplete_unknown_or_off_scale_labels(
    labels: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        merge_labels(_set(_item("g1"), _item("g2")), labels)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s["items"].append(deepcopy(s["items"][0])), "duplicate item id"),
        (lambda s: s["items"][0].update(kind="defect", defect=None), "defect"),
        (lambda s: s["items"][0].update(kind="generated", defect="x"), "defect"),
        (lambda s: s["items"][0].update(context=[]), "context"),
        (lambda s: s["items"][0].update(answer=" "), "answer"),
        (lambda s: s.update(schema_version=2), "schema"),
        (lambda s: s.update(items=[]), "items"),
    ],
)
def test_invalid_sets_are_rejected(change: Any, message: str) -> None:
    artifact = _set(_item("g1"))
    change(artifact)
    with pytest.raises(ValueError, match=message):
        validate_set(artifact)


class RecordingJudge:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, Answer, list[str], str]] = []
        self._fail_on = fail_on

    def score(
        self,
        question: str,
        answer: Answer,
        context: list[RetrievedChunk],
        reference_answer: str,
    ) -> tuple[AnswerScores, Usage]:
        self.calls.append(
            (question, answer, [rc.chunk.id for rc in context], reference_answer)
        )
        if self._fail_on is not None and self._fail_on in answer.text:
            raise RuntimeError("provider message with private details")
        return AnswerScores(0.9, 0.8, 0.7, "because"), Usage(1000, 200)


def test_score_items_judges_each_answer_with_its_own_context_and_citations() -> None:
    judge = RecordingJudge()
    artifact = _set(_item("g1", answer="Claim [doc-a:0] and [doc:0]."))
    scored = score_items(artifact, judge, "claude-opus-4-8")

    assert artifact["items"][0]["judge"] is None  # input untouched
    question, answer, context_ids, reference = judge.calls[0]
    assert (question, reference) == ("What is asked?", "The reference.")
    assert context_ids == ["doc-a:0", "doc-b:0"]
    # The judge sees what the evaluation path would give it.
    assert [c.chunk_id for c in answer.citations] == ["doc-a:0"]
    assert answer.invalid_citations == ["doc:0"]
    assert scored["items"][0]["judge"] == {
        "faithfulness": 0.9,
        "relevance": 0.8,
        "citation_correctness": 0.7,
        "rationale": "because",
        "input_tokens": 1000,
        "output_tokens": 200,
        # Opus 4.8: 1000 * $5/M + 200 * $25/M
        "cost_usd": pytest.approx(0.01),
    }


def test_score_items_records_a_failure_and_keeps_judging_the_rest() -> None:
    judge = RecordingJudge(fail_on="broken")
    scored = score_items(
        _set(_item("g1", answer="broken [doc-a:0]"), _item("g2")),
        judge,
        "claude-opus-4-8",
    )
    # Only the error type is kept: provider messages may hold private data.
    assert scored["items"][0]["judge"] == {"error_type": "RuntimeError"}
    assert scored["items"][1]["judge"]["faithfulness"] == 0.9
    assert len(judge.calls) == 2


def _planted(
    item_id: str,
    defect: str,
    expected_low: list[str],
    judge: tuple[float, float, float],
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    item = _item(item_id, kind="defect", defect=defect, judge=judge)
    item["expected_low"] = expected_low
    if case_id is not None:
        item["case_id"] = case_id
    return item


def test_defect_detection_counts_planted_defects_the_judge_scored_down() -> None:
    # No human labels involved: each defect was planted, so the dimension it
    # should lower is known. "Detected" means the judge left full marks (level 1).
    report = defect_detection(
        _set(
            _item("g1", judge=(1.0, 1.0, 1.0)),
            _item("g2", judge=(1.0, 0.5, 1.0)),
            _planted(
                "d1",
                "wrong_passage",
                ["citation_correctness"],
                (1.0, 1.0, 0.3),
                case_id="case-g1",
            ),
            _planted("d2", "wrong_passage", ["citation_correctness"], (1.0, 1.0, 1.0)),
            _planted("d3", "other_question", ["relevance"], (0.6, 0.0, 1.0)),
        )
    )

    assert (report["defects"], report["detected"]) == (3, 2)
    assert report["by_defect"] == {
        "other_question": {"n": 1, "detected": 1, "dimensions": ["relevance"]},
        "wrong_passage": {"n": 2, "detected": 1, "dimensions": ["citation_correctness"]},
    }
    assert report["items"][0] == {
        "id": "d1",
        "case_id": "case-g1",
        "defect": "wrong_passage",
        "dimension": "citation_correctness",
        "judge": 0.3,
        "judge_level": 0.5,
        "detected": True,
        # The same case's untouched answer, on the same dimension.
        "clean_judge": 1.0,
        "also_lowered": [],
    }
    missed, relevance = report["items"][1], report["items"][2]
    assert (missed["id"], missed["detected"], missed["clean_judge"]) == (
        "d2",
        False,
        None,
    )
    # A defect may drag another dimension down too; that is reported, not counted.
    assert (relevance["detected"], relevance["also_lowered"]) == (True, ["faithfulness"])
    # Untouched answers the judge marked down are its false alarms.
    assert report["clean"] == {
        "n": 2,
        "flagged": [
            {
                "id": "g2",
                "case_id": "case-g2",
                "dimension": "relevance",
                "judge": 0.5,
                "judge_level": 0.5,
            }
        ],
    }


def test_defect_detection_needs_scores_and_planted_defects() -> None:
    unscored = _set(_item("g1"), _planted("d1", "x", ["relevance"], (1, 1, 1)))
    with pytest.raises(ValueError, match="1 without judge scores"):
        defect_detection(unscored)
    with pytest.raises(ValueError, match="no planted defects"):
        defect_detection(_set(_item("g1", judge=(1, 1, 1))))


@pytest.mark.parametrize(
    "change",
    [
        lambda item: item.update(expected_low=["not-a-dimension"]),
        lambda item: item.update(expected_low=["relevance", "relevance"]),
        lambda item: item.update(expected_low="relevance"),
        lambda item: item.update(
            kind="generated", defect=None, expected_low=["relevance"]
        ),
    ],
)
def test_invalid_expected_dimensions_are_rejected(change: Any) -> None:
    item = _planted("d1", "other_question", ["relevance"], (1, 1, 1))
    change(item)
    with pytest.raises(ValueError, match="expected_low"):
        validate_set(_set(item))


def test_retained_calibration_set_matches_the_published_judge_check() -> None:
    # Guards docs/judge-2026-11-12.md: re-scoring the set changes these numbers and
    # must be reviewed together with the report.
    import json
    from pathlib import Path

    path = Path(__file__).parents[2] / "benchmarks/judge-calibration/2026-11-12.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    items = validate_set(artifact)
    assert [item["kind"] for item in items].count("generated") == 30
    report = defect_detection(artifact)
    assert (report["defects"], report["detected"]) == (12, 11)
    assert {name: s["detected"] for name, s in report["by_defect"].items()} == {
        "nonexistent_id": 2,
        "other_question": 3,
        "unsupported_claim": 3,
        "wrong_passage": 3,
    }
    assert report["clean"] == {"n": 30, "flagged": []}
    assert all(not row["also_lowered"] for row in report["items"])
