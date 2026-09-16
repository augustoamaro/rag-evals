from pathlib import Path

import pytest

from rag.adapters.eval_artifacts import load_artifact, write_artifact


def test_baseline_artifact_roundtrip_and_failed_export_preserves_previous(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.json"
    write_artifact(path, {"schema_version": 1, "text": "português"})
    assert load_artifact(path) == {"schema_version": 1, "text": "português"}
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_artifact(path, {"score": float("nan")})
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"a": 1, "a": 2}',
        '{"score": NaN}',
        '{"score": Infinity}',
        '{"score": 1e999}',
    ],
)
def test_import_rejects_ambiguous_or_nonfinite_json(tmp_path: Path, text: str) -> None:
    path = tmp_path / "run.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_artifact(path)
