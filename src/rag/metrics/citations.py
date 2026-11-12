from __future__ import annotations

import re
from collections.abc import Collection

# A bracketed token without whitespace, e.g. "[cap-theorem:0]".
_CITATION = re.compile(r"\[([^\]\s]+)\]")


def cited_ids(text: str) -> list[str]:
    """Bracketed ids in an answer, unique and in first-seen order."""
    return list(dict.fromkeys(_CITATION.findall(text)))


def split_citations(text: str, retrieved: Collection[str]) -> tuple[list[str], list[str]]:
    """Split cited ids into retrieved passages and ids that match none."""
    cited = cited_ids(text)
    return (
        [cid for cid in cited if cid in retrieved],
        [cid for cid in cited if cid not in retrieved],
    )
