from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rag.adapters.embedding.fastembed_embedder as embedding_module
from rag.adapters.embedding.fastembed_embedder import FastEmbedEmbedder


def _model_files(directory: Path) -> None:
    directory.mkdir(parents=True)
    for name, content in {
        "model.onnx": b"model weights",
        "config.json": b'{"pad_token_id":0}',
        "tokenizer.json": b'{"tokens":["hello"]}',
        "tokenizer_config.json": b'{"max_length":512}',
        "special_tokens_map.json": b'{"pad_token":"[PAD]"}',
        "model.onnx_data": b"external tensor weights",
    }.items():
        (directory / name).write_bytes(content)


def _stub_loaded_model(
    monkeypatch: pytest.MonkeyPatch, directory: Path, *, dim: int = 384
) -> None:
    # Replace only the external ONNX load; fingerprinting still reads real files.
    model = SimpleNamespace(
        _model_dir=directory,
        model_description=SimpleNamespace(
            model_file="model.onnx", additional_files=["model.onnx_data"], dim=dim
        ),
        model=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]),
    )
    monkeypatch.setattr(
        embedding_module, "TextEmbedding", lambda **kwargs: SimpleNamespace(model=model)
    )


def test_identity_hashes_model_configuration_and_tokenizer_without_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_dir = tmp_path / "cache-a"
    second_dir = tmp_path / "cache-b"
    _model_files(first_dir)
    _model_files(second_dir)
    (second_dir / ".env").write_text("PRIVATE_TOKEN=do-not-include")
    _stub_loaded_model(monkeypatch, first_dir)
    first = FastEmbedEmbedder().identity()
    _stub_loaded_model(monkeypatch, second_dir)
    second = FastEmbedEmbedder().identity()

    assert first == second
    assert first["model"] == "BAAI/bge-small-en-v1.5"
    assert first["dimension"] == 384
    artifacts = first["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["model.onnx"] == hashlib.sha256(b"model weights").hexdigest()
    assert (
        artifacts["tokenizer.json"] == hashlib.sha256(b'{"tokens":["hello"]}').hexdigest()
    )
    serialized = json.dumps(first)
    assert str(tmp_path) not in serialized
    assert "PRIVATE_TOKEN" not in serialized
    runtime_versions = first["runtime_versions"]
    assert isinstance(runtime_versions, dict)
    assert runtime_versions["fastembed"]


@pytest.mark.parametrize(
    "changed_file",
    [
        "model.onnx",
        "model.onnx_data",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
)
def test_identity_changes_when_any_loaded_artifact_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_file: str
) -> None:
    directory = tmp_path / "cache"
    _model_files(directory)
    _stub_loaded_model(monkeypatch, directory)
    before = FastEmbedEmbedder().identity()

    (directory / changed_file).write_bytes(b"changed content")
    after = FastEmbedEmbedder().identity()

    assert before["artifact_sha256"] != after["artifact_sha256"]
    assert before != after


def test_identity_is_cached_for_loaded_model_and_defensively_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "cache"
    _model_files(directory)
    _stub_loaded_model(monkeypatch, directory)
    embedder = FastEmbedEmbedder()
    original = embedder.identity()
    returned = embedder.identity()
    artifacts = returned["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["model.onnx"] = "caller mutation"
    (directory / "model.onnx").unlink()

    assert embedder.identity() == original


def test_identity_rejects_missing_artifact_and_wrong_model_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "cache"
    _model_files(directory)
    _stub_loaded_model(monkeypatch, directory, dim=768)
    with pytest.raises(ValueError, match="dimension"):
        FastEmbedEmbedder().identity()

    _stub_loaded_model(monkeypatch, directory)
    (directory / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="artifact"):
        FastEmbedEmbedder().identity()
