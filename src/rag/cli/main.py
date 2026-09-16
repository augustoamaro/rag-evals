from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from rag.adapters.datasets.loader import load_dataset
from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.migrations import apply_migrations
from rag.adapters.db.pool import make_pool
from rag.adapters.db.snapshot_store import SnapshotChunkStore
from rag.adapters.eval_artifacts import load_artifact, write_artifact
from rag.composition import build_runtime, ingest_dataset
from rag.config import get_settings
from rag.domain.entities import Strategy
from rag.services.comparison import compare_baseline, make_baseline
from rag.services.gate import check_thresholds

app = typer.Typer(help="RAG service + eval harness CLI.", no_args_is_help=True)


@app.command()
def ingest(
    dataset: Annotated[
        Path | None, typer.Option(help="Versioned YAML or local BEIR directory.")
    ] = None,
) -> None:
    """Apply migrations and publish a complete immutable dataset/index snapshot."""
    settings = get_settings()
    try:
        loaded = load_dataset(dataset or settings.dataset_path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--dataset") from exc
    apply_migrations(settings.database_url)
    pool = make_pool(settings.database_url)
    try:
        snapshot = ingest_dataset(settings, pool, loaded)
        count = len(SnapshotChunkStore(pool, snapshot.id).all_chunks())
        typer.echo(f"Indexed {count} chunks from {len(loaded.documents)} documents.")
        typer.echo(f"index_id={snapshot.id}")
    except ValueError as exc:
        typer.echo(f"INGEST FAILED: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        pool.close()


@app.command(name="eval")
def run_eval(
    strategy: Strategy = Strategy.HYBRID,
    k: int = typer.Option(10, min=1, max=50),
    judge: bool = False,
    gate: bool = False,
    min_recall: float = typer.Option(0.7, min=0, max=1),
    min_ndcg: float = typer.Option(0.6, min=0, max=1),
    dataset: Annotated[
        Path | None, typer.Option(help="Versioned YAML or local BEIR directory.")
    ] = None,
    index_id: Annotated[
        str | None, typer.Option(help="Evaluate a retained index and its stored dataset.")
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(help="Dataset split; defaults to its declared default_split."),
    ] = None,
    language: str = typer.Option(
        "en", help="Evaluate one language independently (e.g. en or pt)."
    ),
    baseline: Annotated[
        Path | None,
        typer.Option(help="Compare against a versioned per-case JSON baseline."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Export a reviewable per-case JSON artifact.")
    ] = None,
    max_drop: float = typer.Option(0.02, min=0, max=1),
    slice_max_drop: float = typer.Option(0.02, min=0, max=1),
    protect_case: Annotated[
        list[str] | None,
        typer.Option(help="Case ID whose evidence recall must not regress; repeatable."),
    ] = None,
) -> None:
    """Evaluate a dataset split/language; optionally judge, export and gate results."""
    for value in (min_recall, min_ndcg, max_drop, slice_max_drop):
        if not math.isfinite(value):
            raise typer.BadParameter("gate limits must be finite numbers in [0, 1]")
    if not language.strip():
        raise typer.BadParameter("language must not be empty")
    if (
        baseline is not None
        and output is not None
        and baseline.resolve() == output.resolve()
    ):
        raise typer.BadParameter("baseline and output must use different paths")
    previous = None
    if baseline is not None:
        try:
            previous = load_artifact(baseline)
            validation = compare_baseline(
                previous, previous, protected_case_ids=set(protect_case or [])
            )
            if not validation.passed:
                raise ValueError("; ".join(validation.failures))
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--baseline") from exc
    elif protect_case:
        raise typer.BadParameter("--protect-case requires --baseline")
    settings = get_settings()
    if judge and not settings.anthropic_api_key:
        raise typer.BadParameter("answer track requires RAG_ANTHROPIC_API_KEY")
    pool = make_pool(settings.database_url)
    try:
        try:
            runtime = build_runtime(
                settings,
                pool,
                enable_answers=judge,
                dataset_path=dataset,
                index_id=index_id,
                split=split,
                language=language,
            )
            run = runtime.evaluator.run(
                runtime.cases,
                strategy,
                k,
                with_answers=judge,
                baseline=previous,
            )
        except (OSError, ValueError) as exc:
            typer.echo(f"EVAL PREFLIGHT FAILED: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        EvalStore(pool).save_run(run, run.manifest)
        comparison = None
        if output is not None or previous is not None:
            artifact = make_baseline(run)
            if previous is not None:
                comparison = compare_baseline(
                    previous,
                    artifact,
                    max_drop=max_drop,
                    slice_max_drop=slice_max_drop,
                    protected_case_ids=set(protect_case or []),
                )
                artifact["comparison"] = {
                    **asdict(comparison),
                    "passed": comparison.passed,
                }
            if output is not None:
                write_artifact(output, artifact)
                typer.echo(f"Artifact: {output}")
        m = run.retrieval_metrics
        typer.echo(
            f"strategy={strategy.value} k={k}  recall={m.recall:.3f} "
            f"precision={m.precision:.3f} mrr={m.mrr:.3f} ndcg={m.ndcg:.3f}  "
            f"p95={run.latency_p95_ms}ms"
        )
        if run.evidence_metrics is not None:
            evidence_recall = run.evidence_metrics["recall"]
            label = f"{evidence_recall:.3f}" if evidence_recall is not None else "N/A"
            typer.echo(
                f"  split={run.manifest['split']} language={run.manifest['language']} "
                f"evidence_recall={label} "
                f"eligible={run.evidence_metrics['eligible_cases']} "
                f"unanswerable={run.evidence_metrics['unanswerable_cases']}"
            )
        if run.with_answers:
            cost_label = (
                "cost estimate" if run.cost_complete else "known cost (incomplete)"
            )
            typer.echo(
                f"  answers: {run.status} "
                f"scored={run.answer_scored}/{run.answer_attempted} "
                f"failed={run.answer_failed}  {cost_label}=${run.cost_usd:.4f}"
            )
        if run.answer_metrics is not None:
            a = run.answer_metrics
            typer.echo(
                f"  scored cases only: faithfulness={a['faithfulness']:.3f} "
                f"relevance={a['relevance']:.3f} "
                f"citations={a['citation_correctness']:.3f}"
            )
        if run.status != "completed":
            typer.echo(f"EVAL INCOMPLETE: run {run.id} saved with answer failures.")
            raise typer.Exit(code=1)
        if comparison is not None:
            if not comparison.passed:
                typer.echo("BASELINE GATE FAILED: " + "; ".join(comparison.failures))
                raise typer.Exit(code=1)
            typer.echo("BASELINE GATE PASSED")
        if gate:
            if (
                run.evidence_metrics is not None
                and not run.evidence_metrics["eligible_cases"]
            ):
                typer.echo("GATE FAILED: no answerable cases to score")
                raise typer.Exit(code=1)
            failures = check_thresholds(m, min_recall, min_ndcg)
            if failures:
                typer.echo("GATE FAILED: " + "; ".join(failures))
                raise typer.Exit(code=1)
            typer.echo("GATE PASSED")
    finally:
        pool.close()


if __name__ == "__main__":
    app()
