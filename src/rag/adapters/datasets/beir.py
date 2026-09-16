"""Import the binary relevance subset of BEIR without altering published splits."""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from rag.adapters.datasets.loader import _object, _string, dataset_from_payload
from rag.domain.datasets import DATASET_SCHEMA_VERSION, Dataset


def _json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            label = f"{path.name} line {number}"
            record = _object(json.loads(line, object_pairs_hook=_json_object), label)
            record_id = _string(record.get("_id"), f"{label} _id")
            if record_id in records:
                raise ValueError(f"duplicate {path.stem} id: {record_id!r}")
            _object(record.get("metadata", {}), f"{label} metadata")
            records[record_id] = record
    if not records:
        raise ValueError(f"{path.name} must contain records")
    return records


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON field: {key!r}")
        record[key] = value
    return record


def _qrels(
    directory: Path,
    documents: set[str],
    queries: set[str],
) -> dict[str, dict[str, dict[str, int]]]:
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    paths = sorted((directory / "qrels").glob("*.tsv"))
    if not paths:
        raise ValueError("BEIR requires qrels/train.tsv, dev.tsv or test.tsv")
    for path in paths:
        if path.stem not in {"train", "dev", "test"}:
            raise ValueError(f"unsupported BEIR qrels split: {path.stem!r}")
        labels: dict[str, dict[str, int]] = {}
        with path.open(encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream, delimiter="\t")
            if next(rows, None) != ["query-id", "corpus-id", "score"]:
                raise ValueError(f"{path.name} has an invalid qrels header")
            for row in rows:
                if not row:
                    continue
                if len(row) != 3:
                    raise ValueError(f"{path.name} qrels rows require three columns")
                query, document, raw_score = row
                if query not in queries:
                    raise ValueError(f"qrels references unknown query {query!r}")
                if document not in documents:
                    raise ValueError(f"qrels references unknown document {document!r}")
                try:
                    score = Decimal(raw_score)
                except InvalidOperation as error:
                    raise ValueError(
                        "BEIR supports only finite binary qrels (0 or 1)"
                    ) from error
                if not score.is_finite() or score not in {0, 1}:
                    raise ValueError("BEIR supports only finite binary qrels (0 or 1)")
                query_labels = labels.setdefault(query, {})
                if document in query_labels:
                    raise ValueError(f"duplicate qrels pair: {query!r}, {document!r}")
                query_labels[document] = int(score)
        qrels[path.stem] = labels
    return qrels


def load_beir(directory: Path) -> Dataset:
    corpus = _records(directory / "corpus.jsonl")
    queries = _records(directory / "queries.jsonl")
    documents: list[dict[str, Any]] = []
    for document_id, record in corpus.items():
        title = _string(record.get("title", ""), "BEIR document title", empty=True)
        text = _string(record.get("text"), "BEIR document text", empty=True)
        documents.append(
            {
                "id": document_id,
                "source": "beir:corpus.jsonl",
                "title": title or document_id,
                "content": f"{title}\n{text}",
            }
        )
    for record in queries.values():
        _string(record.get("text"), "BEIR query text")
    qrels = _qrels(directory, set(corpus), set(queries))
    source_hash = hashlib.sha256(
        _json(
            {
                "corpus": corpus,
                "queries": queries,
                "qrels": qrels,
            }
        ).encode("utf-8")
    ).hexdigest()
    merged: dict[str, tuple[str, dict[str, int], list[str]]] = {}
    for original_split, labels in sorted(qrels.items()):
        split = "dev" if original_split == "train" else original_split
        for query_id, query_labels in labels.items():
            if query_id in merged:
                previous_split, previous_labels, original_splits = merged[query_id]
                if previous_split != split:
                    raise ValueError(
                        f"query group {query_id!r} leaks between dev and test"
                    )
                if previous_labels != query_labels:
                    raise ValueError(
                        f"conflicting train and dev qrels for query {query_id!r}"
                    )
                original_splits.append(original_split)
            else:
                merged[query_id] = (split, query_labels, [original_split])
    if not merged:
        raise ValueError("BEIR qrels must contain at least one query")
    relevant_documents: set[str] = set()
    cases: list[dict[str, Any]] = []
    for query_id, (split, query_labels, original_splits) in merged.items():
        positive = sorted(
            document for document, score in query_labels.items() if score == 1
        )
        relevant_documents.update(positive)
        query = queries[query_id]
        metadata = _object(query.get("metadata", {}), "BEIR query metadata")
        cases.append(
            {
                "id": query_id,
                "question": query["text"],
                "reference_answer": "",
                "evidence_ids": [f"document:{document}" for document in positive],
                "split": split,
                "tags": [
                    "beir",
                    "retrieval-only",
                    *[f"source-split:{original}" for original in sorted(original_splits)],
                ],
                "language": _string(
                    metadata.get("language", "en"), "BEIR query language"
                ),
                "answerable": bool(positive),
                "group_id": _string(
                    metadata.get("group_id", query_id), "BEIR query group_id"
                ),
            }
        )
    return dataset_from_payload(
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "name": "beir",
            "version": f"sha256:{source_hash}",
            "default_split": "test" if qrels.get("test") else "dev",
            "documents": documents,
            "evidence": [
                {"id": f"document:{document}", "document_id": document}
                for document in sorted(relevant_documents)
            ],
            "cases": cases,
            "provenance": {
                "format": "BEIR corpus.jsonl / queries.jsonl / qrels/*.tsv",
                "annotations": (
                    "Published external relevance labels; "
                    "not authored or reviewed by this project."
                ),
                "evaluation": "document-level retrieval only",
                "references": (
                    "Reference answers are unavailable; "
                    "answer generation and judging are unsupported."
                ),
                "source_sha256": source_hash,
                "qrels": _json(qrels),
                "split_mapping": "train -> dev; dev -> dev; test -> test",
            },
        }
    )
