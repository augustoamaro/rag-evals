from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from rag.domain.datasets import Dataset, dataset_digest, dataset_payload
from rag.domain.entities import Chunk, EvalCase, Strategy

# Explicitly allow configuration/provenance fields, never credentials or URLs.
_RUNTIME_FIELDS = (
    "git_sha",
    "git_dirty",
    "source_sha256",
    "dependency_lock_sha256",
    "python_version",
    "package_versions",
    "embedding_model",
    "embedding_dim",
    "chunker",
    "retrieval_pool_k",
    "rrf_k",
    "reranker_model",
    "generator_prompt_version",
    "judge_prompt_version",
    "index_id",
    "embedding_identity",
)


def _digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def chunks_digest(chunks: list[Chunk]) -> str:
    return _digest([asdict(c) for c in sorted(chunks, key=lambda c: c.id)])


def build_manifest(
    cases: list[EvalCase],
    chunks: list[Chunk],
    strategy: Strategy,
    k: int,
    with_answers: bool,
    model: str,
    *,
    runtime: dict[str, object] | None = None,
    expected_chunks: list[Chunk] | None = None,
    dataset: Dataset | None = None,
    document_level: bool = False,
) -> dict[str, object]:
    metadata = runtime or {}
    indexed_hash = chunks_digest(chunks)
    expected_hash = (
        chunks_digest(expected_chunks) if expected_chunks is not None else None
    )
    manifest: dict[str, object] = {
        **{key: metadata.get(key) for key in _RUNTIME_FIELDS},
        "schema_version": 1,
        "strategy": strategy.value,
        "k": k,
        "with_answers": with_answers,
        "llm_model": model,
        "dataset_sha256": _digest([asdict(c) for c in sorted(cases, key=lambda c: c.id)]),
        "indexed_chunks_sha256": indexed_hash,
        "expected_chunks_sha256": expected_hash,
        "index_matches_expected": indexed_hash == expected_hash
        if expected_hash
        else None,
        # This records a preflight check, not a historical database snapshot.
        "index_consistency": "preflight_content_check",
        "case_count": len(cases),
        "chunk_count": len(chunks),
    }
    if dataset is not None:
        manifest.update(
            schema_version=2,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_sha256=dataset_digest(dataset),
            case_set_sha256=_digest(
                dataset_payload(replace(dataset, cases=cases))["cases"]
            ),
            split=cases[0].split,
            language=cases[0].language,
            retrieval_level="document" if document_level else "chunk",
            retrieval_candidate_k=50 if document_level else k,
            ranking_metric_version=(
                "document-first-hit-v1" if document_level else "span-overlap-chunks-v1"
            ),
            evidence_metric_version="document-or-word-union-v1",
            index_consistency=(
                "immutable_snapshot"
                if metadata.get("index_id")
                else "preflight_content_check"
            ),
        )
    return manifest
