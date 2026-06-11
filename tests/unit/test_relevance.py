from rag.domain.entities import Chunk
from rag.metrics.relevance import relevant_chunk_ids


def _c(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, document_id="d", ordinal=0, text=text, token_count=0)


def test_relevant_by_normalized_substring() -> None:
    chunks = [_c("1", "The  CAP  theorem states..."), _c("2", "unrelated text")]
    assert relevant_chunk_ids(chunks, ["the cap theorem"]) == {"1"}


def test_no_snippet_match_is_empty() -> None:
    chunks = [_c("1", "hello world")]
    assert relevant_chunk_ids(chunks, ["nonexistent"]) == set()


def test_blank_snippets_ignored() -> None:
    chunks = [_c("1", "hello world")]
    assert relevant_chunk_ids(chunks, ["  ", ""]) == set()
