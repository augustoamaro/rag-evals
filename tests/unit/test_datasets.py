from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest


def _payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "tiny",
        "version": "1",
        "default_split": "smoke",
        "provenance": {"origin": "test fixture"},
        "documents": [
            {
                "id": "alpha",
                "source": "alpha.md",
                "title": "Alpha",
                "content": "Alpha\n beta  gamma delta.",
            },
            {
                "id": "beta",
                "source": "beta.md",
                "title": "Beta",
                "content": "Another document.",
            },
        ],
        "evidence": [
            {
                "id": "alpha:1-3",
                "document_id": "alpha",
                "word_start": 1,
                "word_end": 3,
                "quote": "beta gamma",
            },
            {"id": "beta:document", "document_id": "beta"},
        ],
        "cases": [
            {
                "id": "first",
                "question": "Which words?",
                "reference_answer": "Beta and gamma.",
                "relevant_snippets": ["beta gamma"],
                "evidence_ids": ["alpha:1-3"],
                "split": "smoke",
                "tags": ["words", "diagnostic"],
                "language": "en",
                "answerable": True,
                "group_id": "alpha-span",
            },
            {
                "id": "second",
                "question": "Which document?",
                "reference_answer": "",
                "evidence_ids": ["beta:document"],
                "split": "dev",
                "tags": [],
                "language": "en",
                "answerable": True,
                "group_id": "beta-document",
            },
        ],
    }


def test_dataset_round_trip_preserves_word_spans_and_document_evidence() -> None:
    from rag.adapters.datasets.loader import dataset_from_payload, select_cases
    from rag.domain.datasets import dataset_payload

    dataset = dataset_from_payload(_payload())
    assert dataset.evidence[0].word_start == 1
    assert dataset.evidence[0].word_end == 3
    assert dataset.evidence[0].quote == "beta gamma"
    assert dataset.evidence[1].word_start is None
    assert dataset.evidence[1].quote == ""
    assert [case.id for case in select_cases(dataset, None)] == ["first"]
    assert [case.id for case in select_cases(dataset, "dev")] == ["second"]
    assert dataset_from_payload(dataset_payload(dataset)) == dataset


def test_dataset_digest_ignores_record_and_set_order_but_tracks_content() -> None:
    from rag.adapters.datasets.loader import dataset_from_payload
    from rag.domain.datasets import dataset_digest

    payload = _payload()
    digest = dataset_digest(dataset_from_payload(payload))
    reordered = copy.deepcopy(payload)
    for key in ("documents", "evidence", "cases"):
        reordered[key].reverse()
    reordered["cases"][1]["tags"].reverse()
    assert dataset_digest(dataset_from_payload(reordered)) == digest

    changed = copy.deepcopy(payload)
    changed["documents"][1]["content"] = "Changed document."
    assert dataset_digest(dataset_from_payload(changed)) != digest


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("documents", "id", 1, "string"),
        ("documents", "id", "", "nonempty"),
        ("documents", "content", "Alpha beta gamma delta.\x00", "NUL"),
        ("documents", "content", "   ", "nonempty"),
        ("evidence", "document_id", "missing", "unknown document"),
        ("evidence", "word_start", True, "integer"),
        ("evidence", "word_start", 1.0, "integer"),
        ("evidence", "word_start", -1, "span"),
        ("evidence", "word_end", 10, "span"),
        ("evidence", "word_end", None, "span"),
        ("evidence", "quote", "beta stale", "quote"),
        ("cases", "question", "", "nonempty"),
        ("cases", "answerable", "false", "boolean"),
        ("cases", "evidence_ids", ["missing"], "unknown evidence"),
        ("cases", "tags", "words", "list"),
        ("cases", "language", 7, "string"),
        ("cases", "reference_answer", None, "string"),
        ("cases", "split", "train", "split"),
    ],
)
def test_dataset_rejects_invalid_labels(
    section: str, field: str, value: Any, error: str
) -> None:
    from rag.adapters.datasets.loader import dataset_from_payload

    payload = _payload()
    payload[section][0][field] = value
    with pytest.raises(ValueError, match=error):
        dataset_from_payload(payload)


@pytest.mark.parametrize("section", ["documents", "evidence", "cases"])
def test_dataset_rejects_duplicate_ids(section: str) -> None:
    from rag.adapters.datasets.loader import dataset_from_payload

    payload = _payload()
    payload[section].append(copy.deepcopy(payload[section][0]))
    with pytest.raises(ValueError, match="duplicate"):
        dataset_from_payload(payload)


def test_dataset_rejects_dev_test_group_leakage() -> None:
    from rag.adapters.datasets.loader import dataset_from_payload

    payload = _payload()
    payload["default_split"] = "dev"
    payload["cases"][0]["split"] = "dev"
    payload["cases"][1]["split"] = "test"
    payload["cases"][1]["group_id"] = "alpha-span"
    with pytest.raises(ValueError, match="group.*dev.*test"):
        dataset_from_payload(payload)


def test_dataset_rejects_absent_split_and_unknown_schema() -> None:
    from rag.adapters.datasets.loader import dataset_from_payload, select_cases

    dataset = dataset_from_payload(_payload())
    with pytest.raises(ValueError, match="split"):
        select_cases(dataset, "test")
    payload = _payload()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        dataset_from_payload(payload)


def test_yaml_resolves_relative_documents_without_leaking_absolute_paths(
    tmp_path: Path,
) -> None:
    import yaml

    from rag.adapters.datasets.loader import load_dataset
    from rag.domain.datasets import dataset_payload

    (tmp_path / "alpha.md").write_text("# Alpha\n\nbeta gamma delta.", encoding="utf-8")
    payload = _payload()
    payload["documents"][0] = {"id": "alpha", "path": "alpha.md"}
    payload["evidence"][0]["word_start"] = 2
    payload["evidence"][0]["word_end"] = 4
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    dataset = load_dataset(dataset_path)
    assert dataset.documents[0].title == "Alpha"
    assert dataset.documents[0].source == "alpha.md"
    assert str(tmp_path) not in str(dataset_payload(dataset))


def _beir_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "original"
    directory.mkdir()
    corpus = [
        {"_id": "article/one", "title": "First", "text": "Alpha evidence."},
        {"_id": "../article-two", "title": "Second", "text": "Beta evidence."},
        {"_id": "unjudged", "title": "", "text": "Background context."},
    ]
    queries = [
        {"_id": "q-train", "text": "An alpha claim."},
        {"_id": "q-dev", "text": "A second alpha claim."},
        {"_id": "q-test", "text": "A beta claim."},
        {"_id": "q-zero", "text": "A claim with no positive labels."},
        {"_id": "q-unjudged", "text": "An unjudged claim."},
    ]
    for name, records in (("corpus", corpus), ("queries", queries)):
        (directory / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    (directory / "qrels").mkdir()
    for split, rows in {
        "train": "q-train\tarticle/one\t1\n",
        "dev": "q-dev\tarticle/one\t1\n",
        "test": "q-test\t../article-two\t1\nq-test\tarticle/one\t0\n"
        "q-zero\tunjudged\t0\n",
    }.items():
        (directory / "qrels" / f"{split}.tsv").write_text(
            "query-id\tcorpus-id\tscore\n" + rows, encoding="utf-8"
        )
    return directory


def test_beir_import_preserves_splits_qrels_and_retrieval_only_labels(
    tmp_path: Path,
) -> None:
    from rag.adapters.datasets.loader import load_dataset, select_cases

    dataset = load_dataset(_beir_directory(tmp_path))
    documents = {document.id: document for document in dataset.documents}
    assert documents["article/one"].content == "First\nAlpha evidence."
    assert documents["../article-two"].content == "Second\nBeta evidence."
    assert documents["unjudged"].content == "\nBackground context."
    assert {case.id for case in select_cases(dataset, "dev")} == {"q-train", "q-dev"}
    assert {case.id for case in select_cases(dataset, "test")} == {"q-test", "q-zero"}
    cases = {case.id: case for case in dataset.cases}
    assert cases["q-test"].group_id == "q-test"
    assert cases["q-test"].evidence_ids == ["document:../article-two"]
    assert cases["q-zero"].evidence_ids == []
    assert not cases["q-zero"].answerable
    assert all(case.reference_answer == "" for case in dataset.cases)
    assert all(
        unit.word_start is None and unit.word_end is None for unit in dataset.evidence
    )
    original_qrels = json.loads(dataset.provenance["qrels"])
    assert original_qrels["train"] == {"q-train": {"article/one": 1}}
    assert original_qrels["test"]["q-test"] == {"../article-two": 1, "article/one": 0}
    assert "published external" in dataset.provenance["annotations"].lower()
    assert dataset.provenance["evaluation"] == "document-level retrieval only"


def test_beir_identity_survives_relocation_and_record_order(tmp_path: Path) -> None:
    from rag.adapters.datasets.loader import load_dataset
    from rag.domain.datasets import dataset_digest, dataset_payload

    source = _beir_directory(tmp_path)
    original = load_dataset(source)
    relocated = tmp_path / "different-name"
    shutil.copytree(source, relocated)
    for filename in ("corpus.jsonl", "queries.jsonl"):
        path = relocated / filename
        path.write_text(
            "\n".join(reversed(path.read_text().splitlines())), encoding="utf-8"
        )
    imported = load_dataset(relocated)
    assert dataset_digest(imported) == dataset_digest(original)
    assert str(tmp_path) not in json.dumps(dataset_payload(imported))
    test_qrels = relocated / "qrels" / "test.tsv"
    test_qrels.write_text(
        test_qrels.read_text().replace(
            "q-test\tarticle/one\t0", "q-test\tarticle/one\t1"
        ),
        encoding="utf-8",
    )
    changed = load_dataset(relocated)
    assert changed.version != original.version
    assert dataset_digest(changed) != dataset_digest(original)


@pytest.mark.parametrize(
    "score",
    ["2", "-1", "0.5", "1.0000000000000001", "1e-10000", "nan", "inf", "-inf", "bad"],
)
def test_beir_rejects_graded_nonfinite_or_invalid_scores(
    tmp_path: Path, score: str
) -> None:
    from rag.adapters.datasets.loader import load_dataset

    directory = _beir_directory(tmp_path)
    (directory / "qrels" / "test.tsv").write_text(
        f"query-id\tcorpus-id\tscore\nq-test\tarticle/one\t{score}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="binary"):
        load_dataset(directory)


@pytest.mark.parametrize(
    ("query", "document", "error"),
    [
        ("missing", "article/one", "unknown query"),
        ("q-test", "missing", "unknown document"),
    ],
)
def test_beir_rejects_qrels_with_unknown_ids(
    tmp_path: Path, query: str, document: str, error: str
) -> None:
    from rag.adapters.datasets.loader import load_dataset

    directory = _beir_directory(tmp_path)
    (directory / "qrels" / "test.tsv").write_text(
        f"query-id\tcorpus-id\tscore\n{query}\t{document}\t1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=error):
        load_dataset(directory)


def test_beir_rejects_conflicting_train_and_dev_qrels(tmp_path: Path) -> None:
    from rag.adapters.datasets.loader import load_dataset

    directory = _beir_directory(tmp_path)
    (directory / "qrels" / "dev.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq-train\t../article-two\t1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="conflicting.*train.*dev"):
        load_dataset(directory)


def test_beir_rejects_shared_query_groups_between_dev_and_test(tmp_path: Path) -> None:
    from rag.adapters.datasets.loader import load_dataset

    directory = _beir_directory(tmp_path)
    queries = directory / "queries.jsonl"
    records = [json.loads(line) for line in queries.read_text().splitlines()]
    for record in records:
        if record["_id"] in {"q-dev", "q-test"}:
            record["metadata"] = {"group_id": "same-claim"}
    queries.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="group.*dev.*test"):
        load_dataset(directory)


def test_bundled_smoke_keeps_original_questions_answers_and_real_spans() -> None:
    from rag.adapters.datasets.loader import load_dataset, select_cases
    from rag.services.corpus_loader import load_golden

    dataset = load_dataset()
    cases = {case.id: case for case in select_cases(dataset, None)}
    original = {case.id: case for case in load_golden()}
    assert cases.keys() == original.keys()
    assert len(cases) == 30
    for case_id, case in cases.items():
        assert case.question == original[case_id].question
        assert case.reference_answer == original[case_id].reference_answer
        assert case.relevant_snippets == original[case_id].relevant_snippets
        assert case.language == "en"
    evidence = {unit.id: unit for unit in dataset.evidence}
    cap_span = evidence[cases["cap-tradeoff"].evidence_ids[0]]
    assert (cap_span.document_id, cap_span.word_start, cap_span.word_end) == (
        "cap-theorem",
        41,
        45,
    )
    assert cap_span.quote == "between consistency and availability:"
    documents = {document.id: document for document in dataset.documents}
    for unit in dataset.evidence:
        assert unit.word_start is not None
        assert unit.word_end is not None
        words = documents[unit.document_id].content.split()
        assert " ".join(words[unit.word_start : unit.word_end]) == unit.quote


def test_bundled_dev_preserves_diagnostic_groups_and_separate_languages() -> None:
    from rag.adapters.datasets.loader import load_dataset, select_cases

    dataset = load_dataset()
    diagnostics = select_cases(dataset, "dev")
    english = [case for case in diagnostics if case.language == "en"]
    portuguese = [case for case in diagnostics if case.language == "pt"]
    assert len(english) == len(portuguese) == 10
    original = {case.id: case for case in select_cases(dataset, "smoke")}
    for case in diagnostics:
        source = original[case.id.removeprefix(f"dev-{case.language}-")]
        assert case.evidence_ids == source.evidence_ids
        assert case.reference_answer == source.reference_answer
        assert case.group_id == source.group_id
        assert "diagnostic" in case.tags
        assert ("paraphrase" if case.language == "en" else "translation") in case.tags
    cap_portuguese = next(case for case in portuguese if case.id == "dev-pt-cap-tradeoff")
    assert cap_portuguese.question == (
        "Quando uma partição de rede ocorre, entre quais propriedades "
        "precisamos escolher segundo o teorema CAP?"
    )
    with pytest.raises(ValueError, match="split"):
        select_cases(dataset, "test")


def test_yaml_rejects_duplicate_fields_instead_of_overwriting(tmp_path: Path) -> None:
    import yaml

    from rag.adapters.datasets.loader import load_dataset

    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(_payload()) + "name: overwritten\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)


def test_yaml_reports_malformed_input_as_validation_error(tmp_path: Path) -> None:
    from rag.adapters.datasets.loader import load_dataset

    path = tmp_path / "benchmark.yaml"
    path.write_text("cases: [unterminated", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        load_dataset(path)


def test_beir_rejects_duplicate_json_fields_instead_of_overwriting(
    tmp_path: Path,
) -> None:
    from rag.adapters.datasets.loader import load_dataset

    directory = _beir_directory(tmp_path)
    path = directory / "corpus.jsonl"
    path.write_text(
        path.read_text().replace(
            '"_id": "article/one"', '"_id": "hidden", "_id": "article/one"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(directory)


@pytest.mark.parametrize("filename", ["corpus.jsonl", "queries.jsonl"])
def test_beir_rejects_duplicate_records(tmp_path: Path, filename: str) -> None:
    from rag.adapters.datasets.loader import load_dataset

    directory = _beir_directory(tmp_path)
    path = directory / filename
    path.write_text(
        path.read_text() + path.read_text().splitlines()[0] + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(directory)


def test_beir_merges_equal_train_dev_labels_without_losing_source_splits(
    tmp_path: Path,
) -> None:
    from rag.adapters.datasets.loader import load_dataset, select_cases

    directory = _beir_directory(tmp_path)
    (directory / "qrels" / "dev.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq-train\tarticle/one\t1\n", encoding="utf-8"
    )
    dataset = load_dataset(directory)
    cases = select_cases(dataset, "dev")
    assert [case.id for case in cases] == ["q-train"]
    assert {"source-split:dev", "source-split:train"} <= set(cases[0].tags)
    assert set(json.loads(dataset.provenance["qrels"])) == {"dev", "train", "test"}


@pytest.mark.parametrize("schema_version", [True, "1", 1.0, None])
def test_dataset_schema_version_requires_an_explicit_integer(schema_version: Any) -> None:
    from rag.adapters.datasets.loader import dataset_from_payload

    payload = _payload()
    payload["schema_version"] = schema_version
    with pytest.raises(ValueError, match="schema_version"):
        dataset_from_payload(payload)


def test_dataset_accepts_zero_evidence_cases_without_inventing_relevance() -> None:
    from rag.adapters.datasets.loader import dataset_from_payload

    payload = _payload()
    payload["cases"][0]["evidence_ids"] = []
    payload["cases"][0]["answerable"] = False
    dataset = dataset_from_payload(payload)
    assert dataset.cases[0].evidence_ids == []
    assert not dataset.cases[0].answerable


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "A changed question?"),
        ("reference_answer", "A changed answer."),
        ("tags", ["new-slice"]),
        ("language", "pt"),
        ("group_id", "new-group"),
        ("evidence_ids", ["beta:document"]),
    ],
)
def test_dataset_identity_includes_evaluation_labels(field: str, value: Any) -> None:
    from rag.adapters.datasets.loader import dataset_from_payload
    from rag.domain.datasets import dataset_digest

    payload = _payload()
    original = dataset_digest(dataset_from_payload(payload))
    payload["cases"][0][field] = value
    assert dataset_digest(dataset_from_payload(payload)) != original


@pytest.mark.parametrize("answerable,evidence", [(True, []), (False, ["alpha:1-3"])])
def test_dataset_rejects_contradictory_answerability(
    answerable: bool, evidence: list[str]
) -> None:
    from rag.adapters.datasets.loader import dataset_from_payload

    payload = _payload()
    payload["cases"][0].update(answerable=answerable, evidence_ids=evidence)
    with pytest.raises(ValueError, match="answerable"):
        dataset_from_payload(payload)


def test_answerability_and_eligibility_change_dataset_identity() -> None:
    from rag.adapters.datasets.loader import dataset_from_payload
    from rag.domain.datasets import dataset_digest

    payload = _payload()
    before = dataset_digest(dataset_from_payload(payload))
    payload["cases"][0].update(answerable=False, evidence_ids=[])
    assert dataset_digest(dataset_from_payload(payload)) != before
