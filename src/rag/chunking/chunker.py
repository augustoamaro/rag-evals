from __future__ import annotations

from rag.domain.entities import Chunk, Document


def chunk_document(
    doc: Document, target_words: int = 180, overlap_words: int = 30
) -> list[Chunk]:
    """Split a document into overlapping, deterministic word-windowed chunks."""
    if target_words <= 0 or not 0 <= overlap_words < target_words:
        raise ValueError("chunking requires target_words > overlap_words >= 0")
    words = doc.content.split()
    if not words:
        return []
    step = max(1, target_words - overlap_words)
    chunks: list[Chunk] = []
    for ordinal, start in enumerate(range(0, len(words), step)):
        window = words[start : start + target_words]
        chunks.append(
            Chunk(
                id=f"{doc.id}:{ordinal}",
                document_id=doc.id,
                ordinal=ordinal,
                text=" ".join(window),
                token_count=len(window),
                word_start=start,
                word_end=start + len(window),
            )
        )
        if start + target_words >= len(words):
            break
    return chunks
