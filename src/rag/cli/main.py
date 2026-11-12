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
from rag.composition import (
    JUDGE_PROMPT_VERSION,
    build_judge,
    build_runtime,
    ingest_dataset,
)
from rag.config import get_settings
from rag.domain.entities import Strategy
from rag.services.comparison import compare_baseline, make_baseline
from rag.services.gate import check_thresholds
from rag.services.judge_calibration import (
    DIMENSIONS,
    agreement,
    merge_labels,
    score_items,
    validate_set,
)

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
    # Same schema guarantee as `rag ingest` and API startup. Without it the pool
    # cannot register pgvector on a fresh database and stalls until it times out,
    # hiding the "run rag ingest" preflight message.
    apply_migrations(settings.database_url)
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
            # Checked in code, not by the judge: a live judge scored citations of
            # passages that were never retrieved as fully correct.
            checked = [r for r in run.case_results if r.invalid_citations is not None]
            flagged = [r.case_id for r in checked if r.invalid_citations]
            listed = f" ({', '.join(flagged)})" if flagged else ""
            typer.echo(
                f"  invalid citations: {len(flagged)} of {len(checked)} answers{listed}"
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


@app.command(name="calibrate-judge")
def calibrate_judge(
    calibration_set: Annotated[
        Path,
        typer.Option("--set", help="Calibration set JSON: items, labels and scores."),
    ],
    labels: Annotated[
        Path | None,
        typer.Option(help="Merge human labels (JSON keyed by item id) into the set."),
    ] = None,
    score: Annotated[
        bool, typer.Option("--score", help="Run the judge on every item (paid calls).")
    ] = False,
    output: Annotated[
        Path | None, typer.Option(help="Export the agreement report as JSON.")
    ] = None,
) -> None:
    """Compare the LLM judge with human labels on a calibration set."""
    try:
        artifact = load_artifact(calibration_set)
        validate_set(artifact)
        if labels is not None:
            artifact = merge_labels(artifact, load_artifact(labels))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--set / --labels") from exc
    settings = get_settings()
    if score and not settings.anthropic_api_key:
        raise typer.BadParameter("--score requires RAG_ANTHROPIC_API_KEY")
    if labels is not None:
        write_artifact(calibration_set, artifact)
        typer.echo(f"Labels merged: {len(artifact['items'])} items")
    if score:
        artifact = score_items(artifact, build_judge(settings), settings.llm_model)
        artifact["judge_run"] = {
            "llm_model": settings.llm_model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
        }
        write_artifact(calibration_set, artifact)
        results = [item["judge"] for item in artifact["items"]]
        scored = [result for result in results if "error_type" not in result]
        costs = [result["cost_usd"] for result in scored]
        cost_label = "cost estimate" if None not in costs else "known cost (incomplete)"
        typer.echo(
            f"Judged {len(scored)}/{len(results)} items  "
            f"failed={len(results) - len(scored)}  "
            f"{cost_label}=${sum(cost or 0.0 for cost in costs):.4f}"
        )
    try:
        report = agreement(artifact)
    except ValueError as exc:
        typer.echo(f"CALIBRATION INCOMPLETE: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"items={report['items']}")
    for dimension in DIMENSIONS:
        summary = report["overall"][dimension]
        flags = summary["flags"]
        typer.echo(
            f"  {dimension:21s} same level {summary['same_level']}/{summary['n']}  "
            f"human flagged {flags['both'] + flags['human_only']}: "
            f"judge also flagged {flags['both']}, missed {flags['human_only']}  "
            f"judge-only flags {flags['judge_only']}  "
            f"mean |diff| {summary['mean_abs_diff']:.3f}"
        )
    for group, summaries in report["groups"].items():
        cells = "  ".join(
            f"{summaries[d]['same_level']}/{summaries[d]['n']}" for d in DIMENSIONS
        )
        typer.echo(f"  {group:28s} same level per dimension: {cells}")
    typer.echo(f"  disagreements: {len(report['disagreements'])}")
    if output is not None:
        write_artifact(output, report)
        typer.echo(f"Report: {output}")


if __name__ == "__main__":
    app()
