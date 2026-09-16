from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from typing import cast

from fastembed import TextEmbedding
from fastembed.text.onnx_embedding import OnnxTextEmbedding

from rag.domain.entities import Vector


class FastEmbedEmbedder:
    """Local, key-free embeddings via fastembed (ONNX). Default: bge-small (384-dim)."""

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._identity: dict[str, object] | None = None

    @property
    def dim(self) -> int:
        return self._dim

    @cached_property
    def _model(self) -> TextEmbedding:
        model = TextEmbedding(model_name=self._model_name)
        # Fingerprint at load time, including when embed() is called first. A later
        # change to the on-disk cache cannot rename the already loaded session.
        self._identity = _model_identity(model, self._model_name, self._dim)
        return model

    def identity(self) -> dict[str, object]:
        """Identify loaded weights/tokenization, without exposing cache paths."""
        _ = self._model
        assert self._identity is not None
        return deepcopy(self._identity)

    def embed(self, texts: list[str]) -> list[Vector]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]


def _model_identity(model: TextEmbedding, model_name: str, dim: int) -> dict[str, object]:
    # FastEmbed's TextEmbedding wraps OnnxTextEmbedding, whose model description
    # names the downloaded ONNX graph and any external tensor files. Its tokenizer
    # loader reads the four named JSON files below from the same model directory.
    loaded = cast(OnnxTextEmbedding, model.model)
    description = loaded.model_description
    if description.dim != dim:
        raise ValueError(
            f"loaded model dimension {description.dim} does not match configured {dim}"
        )
    directory = Path(loaded._model_dir)
    names = {
        description.model_file,
        *description.additional_files,
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    artifacts: dict[str, str] = {}
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model artifact must be relative to its cache directory")
        path = directory / relative
        try:
            with path.open("rb") as artifact:
                digest = hashlib.file_digest(artifact, "sha256").hexdigest()
        except OSError as exc:
            raise ValueError(f"cannot fingerprint model artifact {name!r}") from exc
        artifacts[relative.as_posix()] = digest
    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":"))
    if loaded.model is None:
        raise ValueError("embedding model is not loaded")
    return {
        "provider": "fastembed",
        "model": model_name,
        "dimension": dim,
        "runtime_versions": {
            package: version(package)
            for package in ("fastembed", "onnxruntime", "tokenizers")
        },
        "execution_providers": loaded.model.get_providers(),
        "artifacts": artifacts,
        "artifact_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
