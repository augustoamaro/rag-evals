"""Dataset values and content identity, independent of storage and file formats."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from rag.domain.entities import Document, EvalCase

DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    document_id: str
    word_start: int | None = None
    word_end: int | None = None
    quote: str = ""


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    documents: list[Document]
    evidence: list[EvidenceUnit]
    cases: list[EvalCase]
    default_split: str = "smoke"
    provenance: dict[str, str] = field(default_factory=dict)


def dataset_payload(dataset: Dataset) -> dict[str, Any]:
    """Return a self-contained snapshot with deterministic record/set ordering."""
    cases = []
    for case in sorted(dataset.cases, key=lambda item: item.id):
        record = asdict(case)
        record["evidence_ids"] = sorted(case.evidence_ids)
        record["relevant_snippets"] = sorted(case.relevant_snippets)
        record["tags"] = sorted(case.tags)
        cases.append(record)
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "name": dataset.name,
        "version": dataset.version,
        "default_split": dataset.default_split,
        "provenance": dict(sorted(dataset.provenance.items())),
        "documents": [
            asdict(document)
            for document in sorted(dataset.documents, key=lambda item: item.id)
        ],
        "evidence": [
            asdict(unit) for unit in sorted(dataset.evidence, key=lambda item: item.id)
        ],
        "cases": cases,
    }


def dataset_digest(dataset: Dataset) -> str:
    """SHA-256 of all dataset content and metadata, independent of input ordering."""
    encoded = json.dumps(
        dataset_payload(dataset),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
