from __future__ import annotations

from dataclasses import replace

from rag.domain.entities import Chunk, EvalCase, Strategy
from rag.services.manifest import build_manifest


def test_manifest_identity_ignores_order_but_detects_text_labels_and_model() -> None:
    chunks = [Chunk("a", "d", 0, "alpha", 1), Chunk("b", "d", 1, "beta", 1)]
    cases = [
        EvalCase("q1", "what?", ["alpha"], "a"),
        EvalCase("q2", "why?", ["beta"], "b"),
    ]
    baseline = build_manifest(cases, chunks, Strategy.HYBRID, 3, False, "model-a")
    reordered = build_manifest(
        cases[::-1], chunks[::-1], Strategy.HYBRID, 3, False, "model-a"
    )
    assert baseline == reordered
    changed_text = build_manifest(
        cases, [replace(chunks[0], text="new")], Strategy.HYBRID, 3, False, "model-a"
    )
    assert changed_text["indexed_chunks_sha256"] != baseline["indexed_chunks_sha256"]
    changed_label = build_manifest(
        [replace(cases[0], reference_answer="new")],
        chunks,
        Strategy.HYBRID,
        3,
        False,
        "model-a",
    )
    assert changed_label["dataset_sha256"] != baseline["dataset_sha256"]
    changed_model = build_manifest(cases, chunks, Strategy.HYBRID, 3, False, "model-b")
    assert changed_model["llm_model"] != baseline["llm_model"]


def test_manifest_excludes_credentials_and_database_urls() -> None:
    import json

    manifest = build_manifest(
        [],
        [],
        Strategy.HYBRID,
        3,
        False,
        "model",
        runtime={
            "database_url": "private-db",
            "anthropic_api_key": "private-key",
            "embedding_model": "embed-v1",
            "strategy": "incorrect",
        },
    )
    assert "private-" not in json.dumps(manifest)
    assert manifest["embedding_model"] == "embed-v1"
    assert manifest["strategy"] == "hybrid"
