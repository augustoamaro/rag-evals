from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from rag.adapters.db.chunk_store import PgChunkStore
from rag.adapters.db.pool import Pool
from rag.chunking.chunker import chunk_document
from rag.domain.entities import Chunk, Document, Vector
from rag.services.ingestion_service import IngestionService

pytestmark = pytest.mark.integration


class DeterministicEmbedder:
    @property
    def dim(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> list[Vector]:
        return [[1.0, *([0.0] * 383)] for _ in texts]


def _document(document_id: str, content: str = "indexed text") -> Document:
    return Document(
        id=document_id,
        source=f"source-{document_id}",
        title=f"title-{document_id}",
        content=content,
    )


def _embedded_chunks(doc: Document) -> list[tuple[Chunk, Vector]]:
    return [(chunk, [1.0, *([0.0] * 383)]) for chunk in chunk_document(doc)]


def test_reingesting_an_empty_document_removes_its_old_chunks(pool: Pool) -> None:
    store = PgChunkStore(pool)
    service = IngestionService(DeterministicEmbedder(), store)
    service.ingest([_document("doc")])

    count = service.ingest([_document("doc", content="")])

    assert count == 0
    assert store.all_chunks() == []


def test_synchronizing_an_empty_corpus_removes_every_chunk(pool: Pool) -> None:
    store = PgChunkStore(pool)
    service = IngestionService(DeterministicEmbedder(), store)
    service.ingest([_document("old")])

    service.ingest([])

    assert store.all_chunks() == []


def test_replacement_removes_documents_omitted_from_the_corpus(pool: Pool) -> None:
    store = PgChunkStore(pool)
    service = IngestionService(DeterministicEmbedder(), store)
    service.ingest([_document("kept"), _document("removed")])

    service.ingest([_document("kept", content="new text")])

    assert {chunk.document_id for chunk in store.all_chunks()} == {"kept"}
    with pool.connection() as conn:
        document_ids = {
            row[0] for row in conn.execute("SELECT id FROM documents").fetchall()
        }
    assert document_ids == {"kept"}


def test_replacement_updates_document_metadata_with_content(pool: Pool) -> None:
    store = PgChunkStore(pool)
    original = _document("doc")
    store.replace_corpus([(original, _embedded_chunks(original))])
    changed = Document(
        id="doc",
        source="new-source",
        title="new-title",
        content="new content",
    )

    store.replace_corpus([(changed, _embedded_chunks(changed))])

    with pool.connection() as conn:
        row = conn.execute(
            "SELECT source, title, content FROM documents WHERE id = 'doc'"
        ).fetchone()
    assert row == ("new-source", "new-title", "new content")


def test_failed_embedding_leaves_the_old_corpus_intact(pool: Pool) -> None:
    class FailingEmbedder(DeterministicEmbedder):
        def embed(self, texts: list[str]) -> list[Vector]:
            if texts == ["fails"]:
                raise RuntimeError("embedding failed")
            return super().embed(texts)

    store = PgChunkStore(pool)
    original = _document("old")
    store.replace_corpus([(original, _embedded_chunks(original))])

    with pytest.raises(RuntimeError, match="embedding failed"):
        IngestionService(FailingEmbedder(), store).ingest(
            [_document("new"), _document("broken", content="fails")]
        )

    assert store.all_chunks() == chunk_document(original)


@pytest.mark.parametrize(
    "invalid_chunks",
    [
        [
            (
                Chunk(
                    id="bad:0",
                    document_id="different-document",
                    ordinal=0,
                    text="bad",
                    token_count=1,
                ),
                [1.0, *([0.0] * 383)],
            )
        ],
        [
            (
                Chunk(
                    id="bad:0",
                    document_id="bad",
                    ordinal=0,
                    text="bad",
                    token_count=1,
                ),
                [1.0],
            )
        ],
    ],
)
def test_invalid_replacement_leaves_the_old_corpus_intact(
    pool: Pool, invalid_chunks: list[tuple[Chunk, Vector]]
) -> None:
    store = PgChunkStore(pool)
    original = _document("old")
    store.replace_corpus([(original, _embedded_chunks(original))])

    with pytest.raises(ValueError):
        store.replace_corpus([(_document("bad"), invalid_chunks)])

    assert store.all_chunks() == chunk_document(original)


def test_replacement_rejects_duplicate_document_ids_without_writes(pool: Pool) -> None:
    store = PgChunkStore(pool)
    original = _document("old")
    store.replace_corpus([(original, _embedded_chunks(original))])
    duplicate = _document("duplicate")

    with pytest.raises(ValueError, match="duplicate document id"):
        store.replace_corpus(
            [
                (duplicate, _embedded_chunks(duplicate)),
                (duplicate, _embedded_chunks(duplicate)),
            ]
        )

    assert store.all_chunks() == chunk_document(original)


def test_concurrent_replacements_do_not_merge_distinct_corpora(pool: Pool) -> None:
    # Delay INSERT after each DELETE so the old unlocked implementation overlaps.
    with pool.connection() as conn:
        conn.execute("""
            CREATE FUNCTION slow_document_insert() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN PERFORM pg_sleep(0.3); RETURN NEW; END $$;
            CREATE TRIGGER slow_document_insert BEFORE INSERT ON documents
            FOR EACH ROW EXECUTE FUNCTION slow_document_insert();
        """)
    ready = Barrier(2)

    def replace_document(doc_id: str) -> None:
        doc = _document(doc_id)
        ready.wait(timeout=5)
        PgChunkStore(pool).replace_corpus([(doc, _embedded_chunks(doc))])

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(replace_document, name) for name in ("a", "b")]
            for future in futures:
                future.result(timeout=10)
        assert {c.document_id for c in PgChunkStore(pool).all_chunks()} in ({"a"}, {"b"})
    finally:
        with pool.connection() as conn:
            conn.execute("DROP TRIGGER slow_document_insert ON documents")
            conn.execute("DROP FUNCTION slow_document_insert()")


def test_database_failure_rolls_back_full_replacement(pool: Pool) -> None:
    from psycopg.errors import RaiseException

    store = PgChunkStore(pool)
    original = _document("old")
    store.replace_corpus([(original, _embedded_chunks(original))])
    with pool.connection() as conn:
        conn.execute("""
            CREATE FUNCTION reject_document_insert() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'simulated write failure'; END $$;
            CREATE TRIGGER reject_document_insert BEFORE INSERT ON documents
            FOR EACH ROW EXECUTE FUNCTION reject_document_insert();
        """)
    try:
        new = _document("new")
        with pytest.raises(RaiseException, match="simulated write failure"):
            store.replace_corpus([(new, _embedded_chunks(new))])
        assert store.all_chunks() == chunk_document(original)
    finally:
        with pool.connection() as conn:
            conn.execute("DROP TRIGGER reject_document_insert ON documents")
            conn.execute("DROP FUNCTION reject_document_insert()")
