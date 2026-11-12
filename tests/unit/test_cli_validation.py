from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from rag.cli.main import app
from rag.config import Settings

runner = CliRunner()


def test_eval_rejects_unknown_strategy_cleanly() -> None:
    # Param validation fires before any DB connection is attempted.
    result = runner.invoke(app, ["eval", "--strategy", "bogus"])
    assert result.exit_code == 2
    assert "bogus" in result.output


def test_eval_rejects_out_of_range_k() -> None:
    result = runner.invoke(app, ["eval", "--k", "0"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "option", ["--min-recall", "--min-ndcg", "--max-drop", "--slice-max-drop"]
)
def test_gates_reject_nonfinite_limits_before_database(
    option: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_pool(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid limits must fail before DB access")

    monkeypatch.setattr("rag.cli.main.make_pool", no_pool)
    result = runner.invoke(app, ["eval", option, "nan"])
    assert result.exit_code == 2


def test_cli_refuses_to_overwrite_its_baseline(tmp_path: Any) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"sentinel": true}')
    result = runner.invoke(
        app, ["eval", "--baseline", str(baseline), "--output", str(baseline)]
    )
    assert result.exit_code == 2
    assert "different" in result.output
    assert baseline.read_text() == '{"sentinel": true}'


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "eval" in result.output


def test_judge_without_key_fails_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rag.cli.main.get_settings",
        lambda: Settings(anthropic_api_key=None, _env_file=None),
    )

    def no_pool(*args: Any, **kwargs: Any) -> None:
        pytest.fail("missing answer configuration must reject before DB access")

    monkeypatch.setattr("rag.cli.main.make_pool", no_pool)
    result = runner.invoke(app, ["eval", "--judge"])
    assert result.exit_code == 2
    assert "RAG_ANTHROPIC_API_KEY" in result.output


@pytest.mark.parametrize("scored", [0, 1])
def test_incomplete_judged_eval_is_saved_reported_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    scored: int,
) -> None:
    from types import SimpleNamespace

    from rag.domain.entities import EvalRun, RunMetrics, Strategy

    run = EvalRun(
        "run-1",
        Strategy.HYBRID,
        10,
        RunMetrics(1, 1, 1, 1),
        [],
        answer_metrics={"faithfulness": 1, "relevance": 1, "citation_correctness": 1}
        if scored
        else None,
        with_answers=True,
        status="partial" if scored else "failed",
        answer_attempted=2,
        answer_scored=scored,
        answer_failed=2 - scored,
        cost_usd=0.001,
        cost_complete=False,
        manifest={"schema_version": 1},
    )
    saved: list[tuple[EvalRun, dict[str, Any]]] = []
    closed: list[bool] = []

    class Store:
        def __init__(self, pool: object) -> None:
            pass

        def save_run(self, result: EvalRun, config: dict[str, Any]) -> None:
            saved.append((result, config))

    monkeypatch.setattr(
        "rag.cli.main.get_settings",
        lambda: Settings(anthropic_api_key="test-key", _env_file=None),
    )
    monkeypatch.setattr("rag.cli.main.apply_migrations", lambda _: None)
    monkeypatch.setattr(
        "rag.cli.main.make_pool",
        lambda _: SimpleNamespace(close=lambda: closed.append(True)),
    )
    monkeypatch.setattr("rag.cli.main.EvalStore", Store)
    monkeypatch.setattr(
        "rag.cli.main.build_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            evaluator=SimpleNamespace(run=lambda *args, **kwargs: run),
            cases=[],
        ),
    )
    result = runner.invoke(app, ["eval", "--judge", "--gate"])
    assert result.exit_code == 1
    assert f"scored={scored}/2" in result.output
    assert "incomplete" in result.output
    assert "GATE PASSED" not in result.output
    assert saved == [(run, run.manifest)]
    assert closed == [True]


def test_eval_migrates_first_so_a_missing_index_is_a_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from rag.domain.errors import EvalPreflightError

    calls: list[str] = []

    def migrate(database_url: str) -> None:
        calls.append("migrate")

    def pool(database_url: str) -> SimpleNamespace:
        calls.append("pool")
        return SimpleNamespace(close=lambda: calls.append("close"))

    def no_index(*args: Any, **kwargs: Any) -> None:
        calls.append("runtime")
        raise EvalPreflightError("no index matches this dataset; run rag ingest")

    monkeypatch.setattr("rag.cli.main.get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr("rag.cli.main.apply_migrations", migrate)
    monkeypatch.setattr("rag.cli.main.make_pool", pool)
    monkeypatch.setattr("rag.cli.main.build_runtime", no_index)
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 2
    assert "rag ingest" in result.output
    # The schema must exist before the pool registers pgvector on a connection.
    assert calls == ["migrate", "pool", "runtime", "close"]
