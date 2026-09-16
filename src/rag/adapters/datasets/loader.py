from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode

from rag.domain.datasets import DATASET_SCHEMA_VERSION, Dataset, EvidenceUnit
from rag.domain.entities import Document, EvalCase

DEFAULT_DATASET = Path(__file__).resolve().parents[4] / "corpus" / "benchmark.yaml"
_SPLITS = {"smoke", "dev", "test"}


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML loading without silently replacing duplicate mapping keys."""


def _yaml_mapping(loader: _StrictLoader, node: MappingNode) -> dict[str, Any]:
    keys: set[str] = set()
    for key, _ in node.value:
        if not isinstance(key, ScalarNode) or key.tag != "tag:yaml.org,2002:str":
            raise ValueError("YAML mapping keys must be strings")
        if key.value in keys:
            raise ValueError(f"duplicate YAML field: {key.value!r}")
        keys.add(key.value)
    return _object(loader.construct_mapping(node, deep=True), "YAML mapping")


_StrictLoader.add_constructor("tag:yaml.org,2002:map", _yaml_mapping)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _fields(record: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = record.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(sorted(unknown))}")


def _string(value: object, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not empty and not value.strip():
        raise ValueError(f"{label} must be nonempty")
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL characters")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _strings(value: object, label: str) -> list[str]:
    values = [_string(item, label) for item in _list(value, label)]
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate value in {label}")
    return sorted(values)


def _integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer or null")
    return value


def _split(value: object) -> str:
    value = _string(value, "split")
    if value not in _SPLITS:
        raise ValueError(f"unsupported split {value!r}; expected smoke, dev or test")
    return value


def _unique(records: list[Any], label: str) -> None:
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise ValueError(f"duplicate {label} id: {record.id!r}")
        seen.add(record.id)


def _document(value: object) -> Document:
    record = _object(value, "document")
    _fields(record, {"id", "source", "title", "content"}, "document")
    return Document(
        id=_string(record.get("id"), "document id"),
        source=_string(record.get("source"), "document source"),
        title=_string(record.get("title"), "document title"),
        content=_string(record.get("content"), "document content"),
    )


def _evidence(value: object, documents: dict[str, Document]) -> EvidenceUnit:
    record = _object(value, "evidence")
    _fields(
        record,
        {"id", "document_id", "word_start", "word_end", "quote"},
        "evidence",
    )
    unit = EvidenceUnit(
        id=_string(record.get("id"), "evidence id"),
        document_id=_string(record.get("document_id"), "evidence document_id"),
        word_start=_integer(record.get("word_start"), "word_start"),
        word_end=_integer(record.get("word_end"), "word_end"),
        quote=_string(record.get("quote", ""), "evidence quote", empty=True),
    )
    if unit.document_id not in documents:
        raise ValueError(f"evidence {unit.id!r} references unknown document")
    start, end = unit.word_start, unit.word_end
    if start is None and end is None:
        if unit.quote:
            raise ValueError("document evidence must not have a quote without a span")
        return unit
    words = documents[unit.document_id].content.split()
    if start is None or end is None or not 0 <= start < end <= len(words):
        raise ValueError(f"evidence {unit.id!r} has an invalid word span")
    if " ".join(unit.quote.split()) != " ".join(words[start:end]):
        raise ValueError(f"evidence {unit.id!r} quote does not match its word span")
    return unit


def _case(value: object, evidence_ids: set[str]) -> EvalCase:
    record = _object(value, "case")
    _fields(
        record,
        {
            "id",
            "question",
            "relevant_snippets",
            "reference_answer",
            "evidence_ids",
            "split",
            "tags",
            "language",
            "answerable",
            "group_id",
        },
        "case",
    )
    case_id = _string(record.get("id"), "case id")
    refs = _strings(record.get("evidence_ids"), "case evidence_ids")
    if set(refs) - evidence_ids:
        raise ValueError(f"case {case_id!r} references unknown evidence")
    answerable = record.get("answerable", True)
    if type(answerable) is not bool:
        raise ValueError("case answerable must be a boolean")
    if answerable != bool(refs):
        raise ValueError("case answerable must agree with its evidence_ids")
    return EvalCase(
        id=case_id,
        question=_string(record.get("question"), "case question"),
        relevant_snippets=_strings(
            record.get("relevant_snippets", []), "case relevant_snippets"
        ),
        reference_answer=_string(
            record.get("reference_answer", ""), "case reference_answer", empty=True
        ),
        evidence_ids=refs,
        split=_split(record.get("split", "smoke")),
        tags=_strings(record.get("tags", []), "case tags"),
        language=_string(record.get("language", "en"), "case language"),
        answerable=answerable,
        group_id=_string(record.get("group_id", ""), "case group_id", empty=True)
        or case_id,
    )


def dataset_from_payload(payload: dict[str, Any]) -> Dataset:
    """Validate a complete persisted snapshot; this function never resolves files."""
    payload = _object(payload, "dataset")
    _fields(
        payload,
        {
            "schema_version",
            "name",
            "version",
            "default_split",
            "provenance",
            "documents",
            "evidence",
            "cases",
        },
        "dataset",
    )
    schema = payload.get("schema_version")
    if type(schema) is not int or schema != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema_version: {schema!r}")
    documents = [_document(item) for item in _list(payload.get("documents"), "documents")]
    _unique(documents, "document")
    documents_by_id = {document.id: document for document in documents}
    evidence = [
        _evidence(item, documents_by_id)
        for item in _list(payload.get("evidence"), "evidence")
    ]
    _unique(evidence, "evidence")
    evidence_ids = {unit.id for unit in evidence}
    cases = [_case(item, evidence_ids) for item in _list(payload.get("cases"), "cases")]
    _unique(cases, "case")
    groups: dict[str, set[str]] = {}
    for case in cases:
        groups.setdefault(case.group_id or case.id, set()).add(case.split)
    for group, splits in groups.items():
        if {"dev", "test"} <= splits:
            raise ValueError(f"group {group!r} leaks between dev and test splits")
    provenance = _object(payload.get("provenance", {}), "provenance")
    dataset = Dataset(
        name=_string(payload.get("name"), "dataset name"),
        version=_string(payload.get("version"), "dataset version"),
        documents=sorted(documents, key=lambda item: item.id),
        evidence=sorted(evidence, key=lambda item: item.id),
        cases=sorted(cases, key=lambda item: item.id),
        default_split=_split(payload.get("default_split", "smoke")),
        provenance={
            key: _string(value, "provenance value") for key, value in provenance.items()
        },
    )
    if cases:
        select_cases(dataset, dataset.default_split)
    return dataset


def select_cases(dataset: Dataset, split: str | None) -> list[EvalCase]:
    selected = dataset.default_split if split is None else _split(split)
    cases = [case for case in dataset.cases if case.split == selected]
    if not cases:
        raise ValueError(f"dataset has no cases in split {selected!r}")
    return cases


def load_dataset(path: Path = DEFAULT_DATASET) -> Dataset:
    """Load versioned YAML, a persisted snapshot, or a local binary BEIR directory."""
    if path.is_dir():
        from rag.adapters.datasets.beir import load_beir

        return load_beir(path)
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid dataset YAML: {error}") from error
    payload = _object(loaded, "dataset")
    documents = []
    for item in _list(payload.get("documents"), "documents"):
        record = _object(item, "document")
        if "path" not in record:
            documents.append(record)
            continue
        _fields(record, {"id", "path"}, "file document")
        relative = Path(_string(record["path"], "document path"))
        if relative.is_absolute() or not (
            path.parent / relative
        ).resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("document path must stay relative to the dataset directory")
        content = (path.parent / relative).read_text(encoding="utf-8")
        document_id = _string(record.get("id"), "document id")
        title = next(
            (line[2:].strip() for line in content.splitlines() if line.startswith("# ")),
            document_id,
        )
        documents.append(
            {
                "id": document_id,
                "source": relative.as_posix(),
                "title": title,
                "content": content,
            }
        )
    payload["documents"] = documents
    return dataset_from_payload(payload)
