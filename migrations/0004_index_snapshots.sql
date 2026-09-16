CREATE TABLE dataset_snapshots (
  sha256 text PRIMARY KEY CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE index_snapshots (
  id text PRIMARY KEY CHECK (id ~ '^idx_[0-9a-f]{64}$'),
  dataset_sha256 text NOT NULL REFERENCES dataset_snapshots (sha256),
  embedding jsonb NOT NULL,
  chunking jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX index_snapshots_dataset ON index_snapshots (dataset_sha256);

CREATE TABLE snapshot_chunks (
  index_id text NOT NULL REFERENCES index_snapshots (id)
    DEFERRABLE INITIALLY DEFERRED,
  id text NOT NULL CHECK (length(id) > 0),
  document_id text NOT NULL CHECK (length(document_id) > 0),
  ordinal int NOT NULL CHECK (ordinal >= 0),
  text text NOT NULL CHECK (length(text) > 0),
  embedding vector(384) NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  token_count int NOT NULL CHECK (token_count > 0),
  word_start int NOT NULL CHECK (word_start >= 0),
  word_end int NOT NULL CHECK (word_end > word_start),
  PRIMARY KEY (index_id, id),
  UNIQUE (index_id, document_id, ordinal),
  CHECK (word_end - word_start = token_count)
);

CREATE INDEX snapshot_chunks_tsv_gin ON snapshot_chunks USING gin (tsv);

CREATE FUNCTION reject_snapshot_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'published snapshot rows are immutable';
END
$$;

CREATE TRIGGER dataset_snapshots_immutable BEFORE UPDATE OR DELETE ON dataset_snapshots
  FOR EACH ROW EXECUTE FUNCTION reject_snapshot_mutation();
CREATE TRIGGER index_snapshots_immutable BEFORE UPDATE OR DELETE ON index_snapshots
  FOR EACH ROW EXECUTE FUNCTION reject_snapshot_mutation();
CREATE TRIGGER snapshot_chunks_immutable BEFORE UPDATE OR DELETE ON snapshot_chunks
  FOR EACH ROW EXECUTE FUNCTION reject_snapshot_mutation();

-- A publisher inserts all chunks before their parent index in one transaction.
-- The deferred FK prevents incomplete drafts from committing. Both insert paths
-- share a lock, including SQL callers, so a competing insert cannot slip into an
-- index while another transaction seals it by inserting its parent row.
CREATE FUNCTION lock_snapshot_publication() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  PERFORM pg_advisory_xact_lock(hashtextextended(NEW.id, 0));
  RETURN NEW;
END
$$;

CREATE FUNCTION reject_published_chunk_insert() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  PERFORM pg_advisory_xact_lock(hashtextextended(NEW.index_id, 0));
  IF EXISTS (SELECT 1 FROM index_snapshots WHERE id = NEW.index_id) THEN
    RAISE EXCEPTION 'cannot insert chunks into a published index snapshot';
  END IF;
  RETURN NEW;
END
$$;

CREATE TRIGGER index_snapshots_publication BEFORE INSERT ON index_snapshots
  FOR EACH ROW EXECUTE FUNCTION lock_snapshot_publication();
CREATE TRIGGER snapshot_chunks_publication BEFORE INSERT ON snapshot_chunks
  FOR EACH ROW EXECUTE FUNCTION reject_published_chunk_insert();
