from __future__ import annotations

from typer.testing import CliRunner

from rag.cli.main import app

runner = CliRunner()


def test_eval_rejects_unknown_strategy_cleanly() -> None:
    # Param validation fires before any DB connection is attempted.
    result = runner.invoke(app, ["eval", "--strategy", "bogus"])
    assert result.exit_code == 2
    assert "bogus" in result.output


def test_eval_rejects_out_of_range_k() -> None:
    result = runner.invoke(app, ["eval", "--k", "0"])
    assert result.exit_code == 2


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "eval" in result.output
