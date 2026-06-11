from rag.chunking.chunker import chunk_document
from rag.domain.entities import Document


def test_chunks_cover_document_with_overlap() -> None:
    words = " ".join(f"w{i}" for i in range(250))
    doc = Document(id="d1", source="s", title="t", content=words)
    chunks = chunk_document(doc, target_words=100, overlap_words=20)
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert all(c.document_id == "d1" for c in chunks)
    assert all(c.id.startswith("d1:") for c in chunks)
    assert chunks[1].text.split()[0] == "w80"  # step = 100 - 20 = 80


def test_short_document_single_chunk() -> None:
    doc = Document(id="d2", source="s", title="t", content="just a few words")
    chunks = chunk_document(doc, target_words=100, overlap_words=20)
    assert len(chunks) == 1
    assert chunks[0].text == "just a few words"


def test_empty_document_no_chunks() -> None:
    doc = Document(id="d3", source="s", title="t", content="   ")
    assert chunk_document(doc) == []
