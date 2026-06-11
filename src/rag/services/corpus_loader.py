from __future__ import annotations

from pathlib import Path

import yaml

from rag.domain.entities import Document, EvalCase

CORPUS_DIR = Path(__file__).resolve().parents[3] / "corpus"
DOCS_DIR = CORPUS_DIR / "docs"
GOLDEN_PATH = CORPUS_DIR / "golden.yaml"


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text()
        docs.append(
            Document(
                id=path.stem,
                source=path.name,
                title=_title(text, path.stem),
                content=text,
            )
        )
    return docs


def load_golden(path: Path = GOLDEN_PATH) -> list[EvalCase]:
    data = yaml.safe_load(path.read_text())
    return [
        EvalCase(
            id=str(case["id"]),
            question=str(case["question"]),
            relevant_snippets=[str(s) for s in case["relevant_snippets"]],
            reference_answer=str(case["reference_answer"]),
        )
        for case in data["cases"]
    ]
