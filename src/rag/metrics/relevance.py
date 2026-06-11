from __future__ import annotations

import re
from collections.abc import Iterable

from rag.domain.entities import Chunk

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def relevant_chunk_ids(chunks: Iterable[Chunk], snippets: list[str]) -> set[str]:
    """Passage-anchored relevance.

    A chunk is relevant if its (whitespace-normalised, lower-cased) text contains
    any of the case's relevant snippets. This keeps ground truth stable across
    chunk-boundary changes — relevance is defined by content, not chunk ids.
    """
    norm_snips = [_norm(s) for s in snippets if s.strip()]
    out: set[str] = set()
    for chunk in chunks:
        body = _norm(chunk.text)
        if any(snip in body for snip in norm_snips):
            out.add(chunk.id)
    return out
