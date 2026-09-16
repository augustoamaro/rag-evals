"""Real embeddings + Postgres through versioned ingest, CLI, and API composition."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Never, cast

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import rag.cli.main as cli
from rag.adapters.datasets.loader import load_dataset
from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.pool import Pool
from rag.adapters.eval_artifacts import load_artifact
from rag.adapters.llm.claude_generator import ClaudeGenerator
from rag.adapters.llm.claude_judge import ClaudeJudge
from rag.adapters.retrieval.pgvector_retriever import PgVectorRetriever
from rag.api.app import AppDeps, create_app
from rag.composition import build_runtime, ingest_dataset
from rag.config import Settings
from rag.domain.datasets import Dataset, EvidenceUnit, dataset_digest, dataset_payload
from rag.domain.entities import Document, EvalCase, Strategy
from rag.domain.snapshots import IndexSnapshot

pytestmark = pytest.mark.integration


@pytest.fixture
def published_dataset(
    pool: Pool,
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Settings, Dataset, IndexSnapshot]:
    dataset = Dataset(
        name="versioned-e2e",
        version="1",
        documents=[
            Document(
                "backups",
                "backups.md",
                "Backups",
                "Backups retain seven daily snapshots.",
            ),
            Document(
                "fusion",
                "fusion.md",
                "Fusion",
                "Reciprocal rank fusion combines multiple rankings.",
            ),
        ],
        evidence=[
            EvidenceUnit("backup-retention", "backups", 2, 5, "seven daily snapshots."),
            EvidenceUnit(
                "fusion-method",
                "fusion",
                0,
                6,
                "Reciprocal rank fusion combines multiple rankings.",
            ),
        ],
        cases=[
            EvalCase(
                id="smoke-en-backups",
                question="How many daily backup snapshots are retained?",
                relevant_snippets=[],
                reference_answer="Seven daily snapshots.",
                evidence_ids=["backup-retention"],
                split="smoke",
                tags=["critical"],
                language="en",
                group_id="backup-group",
            ),
            EvalCase(
                id="dev-pt-fusion",
                question="Como Reciprocal rank fusion combina os rankings?",
                relevant_snippets=[],
                reference_answer="Ela combina diversos rankings.",
                evidence_ids=["fusion-method"],
                split="dev",
                tags=["translation"],
                language="pt",
                group_id="fusion-group",
            ),
        ],
    )
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(dataset_payload(dataset)), encoding="utf-8")
    settings = Settings(
        database_url=database_url,
        dataset_path=path,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
        index_id=None,
        anthropic_api_key=None,
        chunk_target_words=6,
        chunk_overlap_words=1,
        _env_file=None,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    loaded = load_dataset(path)
    return settings, loaded, ingest_dataset(settings, pool, loaded)


def _api_run(client: TestClient, split: str, language: str) -> dict[str, Any]:
    response = client.post(
        "/evals/run",
        json={"strategy": "hybrid", "k": 2, "split": split, "language": language},
    )
    assert response.status_code == 200, response.text
    detail = client.get(f"/evals/runs/{response.json()['id']}")
    assert detail.status_code == 200, detail.text
    return cast(dict[str, Any], detail.json())


def _assert_cli_api_agree(artifact: dict[str, Any], detail: dict[str, Any]) -> None:
    for key in (
        "schema_version",
        "dataset_sha256",
        "case_set_sha256",
        "index_id",
        "embedding_identity",
        "embedding_model",
        "embedding_dim",
        "chunker",
        "split",
        "language",
        "k",
        "index_consistency",
        "ranking_metric_version",
        "evidence_metric_version",
    ):
        assert artifact["manifest"][key] == detail["config"][key], key
    assert artifact["metrics"]["retrieval"] == detail["retrieval_metrics"]
    assert artifact["metrics"]["evidence"] == detail["evidence_metrics"]
    assert artifact["metrics"]["slices"] == detail["slices"]
    assert detail["evidence_metrics"] == {
        "recall": 1.0,
        "eligible_cases": 1,
        "unanswerable_cases": 0,
    }
    assert len(artifact["cases"]) == len(detail["cases"]) == 1
    for key, value in artifact["cases"][0].items():
        assert detail["cases"][0][key] == value, key


def test_cli_baseline_and_api_keep_languages_separate_and_share_provenance(
    pool: Pool,
    published_dataset: tuple[Settings, Dataset, IndexSnapshot],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, dataset, snapshot = published_dataset
    runner = CliRunner()
    baseline = tmp_path / "baseline-en.json"
    candidate = tmp_path / "candidate-en.json"
    arguments = [
        "eval",
        "--dataset",
        str(settings.dataset_path),
        "--k",
        "2",
        "--split",
        "smoke",
        "--language",
        "en",
    ]
    first = runner.invoke(cli.app, [*arguments, "--output", str(baseline)])
    assert first.exit_code == 0, first.output
    english = load_artifact(baseline)
    assert english["manifest"]["dataset_sha256"] == dataset_digest(dataset)
    assert english["manifest"]["index_id"] == snapshot.id
    assert english["manifest"]["schema_version"] == 2
    assert english["manifest"]["index_consistency"] == "immutable_snapshot"
    assert english["manifest"]["chunker"] == {
        "algorithm": "word_window_v1",
        "target_words": 6,
        "overlap_words": 1,
    }
    assert len(english["manifest"]["embedding_identity"]["artifact_sha256"]) == 64
    assert [case["case_id"] for case in english["cases"]] == ["smoke-en-backups"]

    paired = runner.invoke(
        cli.app,
        [*arguments, "--baseline", str(baseline), "--output", str(candidate)],
    )
    assert paired.exit_code == 0, paired.output
    assert "BASELINE GATE PASSED" in paired.output
    comparison = load_artifact(candidate)["comparison"]
    assert comparison["passed"] is True
    assert comparison["failures"] == []
    assert comparison["metrics"]["evidence_recall"]["mean_delta"] == 0

    runtime = build_runtime(settings, pool, enable_answers=False)
    app = create_app(
        AppDeps(
            runtime.retriever,
            runtime.generator,
            runtime.evaluator,
            EvalStore(pool),
            runtime.cases,
            dataset=runtime.dataset,
        )
    )
    with TestClient(app) as client:
        english_api = _api_run(client, "smoke", "en")
        _assert_cli_api_agree(english, english_api)
        assert english_api["cases"][0]["reference_answer"] == "Seven daily snapshots."

        portuguese_output = tmp_path / "portuguese.json"
        portuguese_run = runner.invoke(
            cli.app,
            [
                "eval",
                "--index-id",
                snapshot.id,
                "--split",
                "dev",
                "--language",
                "pt",
                "--k",
                "2",
                "--output",
                str(portuguese_output),
            ],
        )
        assert portuguese_run.exit_code == 0, portuguese_run.output
        portuguese = load_artifact(portuguese_output)
        portuguese_api = _api_run(client, "dev", "pt")
        _assert_cli_api_agree(portuguese, portuguese_api)
        assert [case["case_id"] for case in portuguese_api["cases"]] == ["dev-pt-fusion"]
        assert (
            portuguese_api["cases"][0]["reference_answer"]
            == "Ela combina diversos rankings."
        )
        assert (
            english["manifest"]["case_set_sha256"]
            != portuguese["manifest"]["case_set_sha256"]
        )
        assert (
            client.post(
                "/evals/run", json={"split": "smoke", "language": "pt"}
            ).status_code
            == 400
        )

    def forbidden_call(*args: object, **kwargs: object) -> Never:
        raise AssertionError("incompatible baseline reached retrieval or a paid model")

    run_count = len(EvalStore(pool).list_runs())
    rejected_output = tmp_path / "rejected.json"
    with monkeypatch.context() as scoped:
        paid_settings = settings.model_copy(update={"anthropic_api_key": "test-unused"})
        scoped.setattr(cli, "get_settings", lambda: paid_settings)
        scoped.setattr(PgVectorRetriever, "retrieve", forbidden_call)
        scoped.setattr(ClaudeGenerator, "answer", forbidden_call)
        scoped.setattr(ClaudeJudge, "score", forbidden_call)
        rejected = runner.invoke(
            cli.app,
            [
                "eval",
                "--index-id",
                snapshot.id,
                "--k",
                "1",
                "--judge",
                "--baseline",
                str(baseline),
                "--output",
                str(rejected_output),
            ],
        )
    assert rejected.exit_code == 2, rejected.output
    assert "incompatible manifest field: k" in rejected.output
    assert not rejected_output.exists()
    assert len(EvalStore(pool).list_runs()) == run_count


def test_retained_index_uses_its_original_dataset_after_local_republication(
    pool: Pool,
    published_dataset: tuple[Settings, Dataset, IndexSnapshot],
) -> None:
    settings, original, old_snapshot = published_dataset
    before = build_runtime(settings, pool, enable_answers=False)
    old_run = before.evaluator.run(before.cases, Strategy.HYBRID, 2)

    changed = replace(
        original,
        version="2",
        documents=[
            replace(
                original.documents[0], content="Backups retain thirty monthly snapshots."
            ),
            original.documents[1],
        ],
        evidence=[
            replace(original.evidence[0], quote="thirty monthly snapshots."),
            original.evidence[1],
        ],
        cases=[
            replace(case, reference_answer="Thirty monthly snapshots.")
            if case.id == "smoke-en-backups"
            else case
            for case in original.cases
        ],
    )
    settings.dataset_path.write_text(
        yaml.safe_dump(dataset_payload(changed)), encoding="utf-8"
    )
    changed_settings = settings.model_copy(
        update={"chunk_target_words": 9, "chunk_overlap_words": 0}
    )
    new_snapshot = ingest_dataset(
        changed_settings, pool, load_dataset(settings.dataset_path)
    )
    assert new_snapshot.id != old_snapshot.id
    current = build_runtime(changed_settings, pool, enable_answers=False)
    assert current.index_id == new_snapshot.id
    new_run = current.evaluator.run(current.cases, Strategy.HYBRID, 2)
    assert new_run.case_results[0].reference_answer == "Thirty monthly snapshots."
    assert "Backups retain thirty monthly snapshots." in {
        chunk.chunk.text for chunk in new_run.case_results[0].retrieved_context
    }

    # An explicit retained index must ignore the changed default file and window.
    retained = build_runtime(
        changed_settings, pool, enable_answers=False, index_id=old_snapshot.id
    )
    replay = retained.evaluator.run(retained.cases, Strategy.HYBRID, 2)
    assert retained.dataset == original
    assert replay.manifest["index_id"] == old_snapshot.id
    assert replay.manifest["dataset_sha256"] == dataset_digest(original)
    assert replay.manifest["chunker"] == {
        "algorithm": "word_window_v1",
        "target_words": 6,
        "overlap_words": 1,
    }
    assert replay.retrieval_metrics == old_run.retrieval_metrics
    assert replay.evidence_metrics == old_run.evidence_metrics
    assert replay.case_results[0].retrieved_ids == old_run.case_results[0].retrieved_ids
    assert replay.case_results[0].reference_answer == "Seven daily snapshots."
    old_text = {chunk.chunk.text for chunk in replay.case_results[0].retrieved_context}
    assert "Backups retain seven daily snapshots." in old_text
    assert "Backups retain thirty monthly snapshots." not in old_text
