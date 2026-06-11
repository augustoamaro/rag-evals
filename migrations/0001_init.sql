CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id text PRIMARY KEY,
  source text NOT NULL,
  title text NOT NULL,
  content text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id text PRIMARY KEY,
  document_id text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
  ordinal int NOT NULL,
  text text NOT NULL,
  embedding vector(384) NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  token_count int NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS eval_runs (
  id text PRIMARY KEY,
  created_at timestamptz NOT NULL DEFAULT now(),
  config jsonb NOT NULL,
  retrieval_metrics jsonb NOT NULL,
  answer_metrics jsonb,
  cost_usd numeric NOT NULL DEFAULT 0,
  latency_p50_ms int NOT NULL DEFAULT 0,
  latency_p95_ms int NOT NULL DEFAULT 0,
  git_sha text
);

CREATE TABLE IF NOT EXISTS eval_case_results (
  id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES eval_runs (id) ON DELETE CASCADE,
  case_id text NOT NULL,
  strategy text NOT NULL,
  retrieved_ids jsonb NOT NULL,
  recall double precision NOT NULL,
  precision double precision NOT NULL,
  mrr double precision NOT NULL,
  ndcg double precision NOT NULL,
  answer text,
  judge_scores jsonb,
  cost_usd numeric NOT NULL DEFAULT 0,
  latency_ms int NOT NULL DEFAULT 0
);
