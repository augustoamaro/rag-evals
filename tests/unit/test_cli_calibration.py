from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from rag.cli.main import app
from rag.config import Settings
from rag.domain.entities import Answer, AnswerScores, RetrievedChunk, Usage

runner = CliRunner()
_DIMENSIONS = ("faithfulness", "relevance", "citation_correctness")


def _item(
    item_id: str,
    human: tuple[float, float, float] | None,
    judge: tuple[float, float, float] | None,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "case_id": f"case-{item_id}",
        "kind": "generated",
        "defect": None,
        "question": "What is asked?",
        "reference_answer": "The reference.",
        "context": [{"id": "doc:0", "document_id": "doc", "text": "A passage."}],
        "answer": "An answer [doc:0].",
        "human": dict(zip(_DIMENSIONS, human, strict=True)) | {"note": ""}
        if human
        else None,
        "judge": dict(zip(_DIMENSIONS, judge, strict=True)) | {"rationale": "r"}
        if judge
        else None,
    }


def _write(path: Path, *items: dict[str, Any]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, "name": "t", "items": list(items)}))
    return path


def test_complete_set_prints_agreement_and_exports_the_report(tmp_path: Path) -> None:
    calibration = _write(
        tmp_path / "set.json",
        _item("a", (1, 1, 1), (1.0, 1.0, 1.0)),
        _item("b", (1, 0, 1), (1.0, 1.0, 0.4)),
    )
    report = tmp_path / "report.json"
    result = runner.invoke(
        app, ["calibrate-judge", "--set", str(calibration), "--output", str(report)]
    )

    assert result.exit_code == 0, result.output
    assert "items=2" in result.output
    # Item b: the human saw an irrelevant answer and the judge gave full marks.
    assert "relevance" in result.output and "same level 1/2" in result.output
    assert "human flagged 1: judge also flagged 0, missed 1" in result.output
    assert "disagreements: 2" in result.output
    saved = json.loads(report.read_text())
    assert saved["agreement"]["overall"]["relevance"]["flags"]["human_only"] == 1
    assert saved["defect_detection"] is None  # this set has no planted defects


def test_incomplete_set_reports_what_is_missing_and_exits_nonzero(
    tmp_path: Path,
) -> None:
    calibration = _write(tmp_path / "set.json", _item("a", None, (1, 1, 1)))
    result = runner.invoke(app, ["calibrate-judge", "--set", str(calibration)])
    assert result.exit_code == 1
    assert "CALIBRATION INCOMPLETE" in result.output
    assert "1 without human labels" in result.output


def test_labels_are_merged_into_the_set_file(tmp_path: Path) -> None:
    calibration = _write(tmp_path / "set.json", _item("a", None, (1, 1, 1)))
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {"a": {"faithfulness": 1, "relevance": 0.5, "citation_correctness": 0}}
        )
    )
    result = runner.invoke(
        app, ["calibrate-judge", "--set", str(calibration), "--labels", str(labels)]
    )
    assert result.exit_code == 0, result.output
    assert "Labels merged: 1 items" in result.output
    assert json.loads(calibration.read_text())["items"][0]["human"] == {
        "faithfulness": 1.0,
        "relevance": 0.5,
        "citation_correctness": 0.0,
        "note": "",
    }


def test_invalid_labels_leave_the_set_untouched(tmp_path: Path) -> None:
    calibration = _write(tmp_path / "set.json", _item("a", None, None))
    before = calibration.read_text()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"zz": {}}))
    result = runner.invoke(
        app, ["calibrate-judge", "--set", str(calibration), "--labels", str(labels)]
    )
    assert result.exit_code == 2
    assert "zz" in result.output
    assert calibration.read_text() == before


def test_scoring_requires_a_key_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _write(tmp_path / "set.json", _item("a", (1, 1, 1), None))
    monkeypatch.setattr(
        "rag.cli.main.get_settings",
        lambda: Settings(anthropic_api_key=None, _env_file=None),
    )

    def no_judge(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a missing key must be rejected before building the judge")

    monkeypatch.setattr("rag.cli.main.build_judge", no_judge)
    result = runner.invoke(app, ["calibrate-judge", "--set", str(calibration), "--score"])
    assert result.exit_code == 2
    assert "RAG_ANTHROPIC_API_KEY" in result.output


def test_scoring_saves_judge_results_with_the_model_and_prompt_that_made_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubJudge:
        def score(
            self,
            question: str,
            answer: Answer,
            context: list[RetrievedChunk],
            reference_answer: str,
        ) -> tuple[AnswerScores, Usage]:
            return AnswerScores(1.0, 0.5, 1.0, "partly relevant"), Usage(1000, 200)

    calibration = _write(tmp_path / "set.json", _item("a", (1, 0.5, 1), None))
    monkeypatch.setattr(
        "rag.cli.main.get_settings",
        lambda: Settings(anthropic_api_key="test-key", _env_file=None),
    )
    monkeypatch.setattr("rag.cli.main.build_judge", lambda settings: StubJudge())
    result = runner.invoke(app, ["calibrate-judge", "--set", str(calibration), "--score"])

    assert result.exit_code == 0, result.output
    assert "Judged 1/1 items  failed=0  cost estimate=$0.0100" in result.output
    saved = json.loads(calibration.read_text())
    assert saved["items"][0]["judge"]["relevance"] == 0.5
    assert saved["judge_run"] == {
        "llm_model": "claude-opus-4-8",
        "judge_prompt_version": "structured-scores-v1",
    }
    assert "same level 1/1" in result.output


def test_planted_defects_are_reported_without_human_labels(tmp_path: Path) -> None:
    caught = _item("d1", None, (1.0, 0.2, 1.0))
    caught.update(kind="defect", defect="other_question", expected_low=["relevance"])
    missed = _item("d2", None, (1.0, 1.0, 1.0))
    missed.update(kind="defect", defect="other_question", expected_low=["relevance"])
    calibration = _write(
        tmp_path / "set.json", _item("a", None, (1, 1, 1)), caught, missed
    )
    report = tmp_path / "report.json"
    result = runner.invoke(
        app, ["calibrate-judge", "--set", str(calibration), "--output", str(report)]
    )

    assert result.exit_code == 0, result.output
    assert "planted defects: judge scored down 1/2" in result.output
    assert (
        "other_question" in result.output
        and "clean answers flagged: 0/1" in result.output
    )
    assert "Agreement with human labels not computed" in result.output
    saved = json.loads(report.read_text())
    assert saved["agreement"] is None
    assert saved["defect_detection"]["detected"] == 1
