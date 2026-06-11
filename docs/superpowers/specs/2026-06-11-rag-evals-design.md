# RAG Service + Eval Harness — Design Spec

**Date:** 2026-06-11
**Status:** Approved (design phase)
**Product name:** rag-evals
**Repo folder:** `rag-evals/`

## 1. Summary

A retrieval-augmented generation (RAG) service paired with a rigorous **evaluation
harness** and a quality dashboard. The service ingests documents, retrieves
relevant passages with **hybrid search** (dense vectors + keyword), and answers
questions with **inline citations** grounded in the retrieved context. The eval
harness is the centerpiece: it measures **retrieval quality** (recall@k, MRR,
nDCG) against a labeled golden set and **answer quality** (faithfulness,
relevance, citation-correctness) via an LLM-as-judge, tracks **cost and latency**,
and **gates regressions in CI**.

This is portfolio project #2. Its job is to be **public proof of AI engineering
with production discipline** — not a weekend RAG demo. The differentiator is that
every retrieval and answer-quality claim is *measured*, reproducibly, with the
results visible on a dashboard and regressions caught automatically.

## 2. Goals & non-goals

### Goals
- Ingest documents → chunk → embed → store in Postgres/pgvector.
- **Hybrid retrieval**: dense (pgvector) + sparse (Postgres full-text search),
  fused with Reciprocal Rank Fusion (RRF); optional cross-encoder rerank.
- Answer questions with **citations** to the specific retrieved chunks used.
- **Eval harness, two tracks:**
  - Retrieval metrics (recall@k, precision@k, MRR, nDCG@k) over a labeled golden
    set — runs with **no LLM and no API key**.
  - Answer quality via **Claude LLM-as-judge** (faithfulness, answer relevance,
    citation-correctness) — gated behind `ANTHROPIC_API_KEY`, degrades gracefully.
- **Cost + latency tracking** per eval run.
- **Regression gate** in CI: fail the build if retrieval metrics drop below
  baselines.
- A **Next.js dashboard**: runs over time, per-run detail, strategy comparison,
  cost/latency, and a "try a query" page.
- Runs from a fresh clone: `docker compose up`, local embeddings (no key) for the
  core; Claude only for generation + judging.
- Production discipline: typed end-to-end (mypy strict + TypeScript strict), tests
  in CI with a badge, secrets scan, polished README with a capture.

### Non-goals (YAGNI)
- Multi-tenant auth / user accounts.
- Real-time document sync or incremental re-indexing UI (ingest is a batch/CLI op).
- A general-purpose connector ecosystem (the bundled corpus + simple loaders suffice).
- Distributed/sharded vector search; horizontal scaling. Single Postgres node is
  the documented scope (the retriever port keeps the door open).
- Fine-tuning or training any model.
- Streaming token-by-token answers in the dashboard (request/response is enough).

## 3. Architecture

Clean architecture: dependencies point **inward** toward a pure domain. The LLM
(Anthropic SDK), the database (pgvector/SQL), and the embedding model each live
behind a port and appear in exactly one adapter.

```
rag-evals/
├── src/rag/
│   ├── domain/        PURE — entities + ports (Protocols). No I/O, no framework.
│   │                  Entities: Document, Chunk, Query, RetrievedChunk, Citation,
│   │                            Answer, EvalCase, CaseResult, RunMetrics.
│   │                  Ports: Embedder, Retriever, Generator, Judge, ChunkStore.
│   ├── metrics/       PURE — recall@k, precision@k, MRR, nDCG@k, RRF fusion.
│   ├── chunking/      PURE — document → chunks (size + overlap, token-aware).
│   ├── adapters/      INFRA — the only modules with I/O / SDK / SQL:
│   │     embedding/fastembed_embedder.py   (bge-small-en-v1.5, 384-dim, local)
│   │     retrieval/pgvector_retriever.py    (dense HNSW + sparse FTS + RRF + rerank)
│   │     llm/claude_generator.py            (RAG answer + citations, Opus 4.8)
│   │     llm/claude_judge.py                (LLM-as-judge, structured output)
│   │     db/                                (psycopg3 pool, migrations runner)
│   ├── services/      APPLICATION — orchestrate ports:
│   │     ingestion_service.py · query_service.py · eval_service.py
│   ├── api/           FastAPI app + routers (thin; delegate to services)
│   ├── cli/           Typer CLI: `rag ingest`, `rag eval`, `rag eval --gate`
│   └── config.py      Settings (pydantic-settings): DB URL, model ids, keys
├── corpus/            Bundled docs + golden set (YAML) — version-controlled
├── migrations/        Versioned .sql files
├── tests/             unit (pure) · integration (real Postgres) · e2e
├── apps/web/          Next.js dashboard (reads the API)
└── infra: docker-compose.yml, Dockerfiles, .github/workflows/ci.yml
```

### Layers

**1. Domain (`src/rag/domain`)** — pure Python, no deps beyond stdlib + pydantic
models for entities. Ports are `typing.Protocol`s:
- `Embedder.embed(texts: list[str]) -> list[Vector]`
- `Retriever.retrieve(query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]`
- `Generator.answer(query: str, context: list[RetrievedChunk]) -> Answer` (Answer
  carries text + `citations: list[Citation]` + token usage)
- `Judge.score(case: EvalCase, answer: Answer, context) -> AnswerScores`
- `ChunkStore.add_documents(...)`, used by ingestion.

**2. Metrics (`src/rag/metrics`)** — pure functions, the senior centerpiece:
- `recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k` over (ranked ids, relevant
  ids). Hand-verified test vectors.
- `rrf_fuse(rankings: list[list[id]], k_const=60) -> list[id]` — Reciprocal Rank
  Fusion. Pure, tested.

**3. Chunking (`src/rag/chunking`)** — pure: split a document into overlapping
chunks with a target size and overlap; deterministic. Tokenization via a simple
word/char heuristic (no heavy tokenizer dependency in the pure layer).

**4. Adapters (`src/rag/adapters`)** — the only place I/O happens:
- `FastEmbedEmbedder` — `fastembed` ONNX model `BAAI/bge-small-en-v1.5` (384-dim).
  Runs CPU-only in-container, **no API key**. Model downloaded at image build.
- `PgVectorRetriever` (reads) + a `PgChunkStore` (writes) — **the persistence
  adapters are the only place SQL lives.** Dense search via pgvector (`<=>` cosine,
  HNSW index); sparse via Postgres FTS (`tsvector` + GIN, `ts_rank_cd`); hybrid via
  RRF over the two ranked lists; optional rerank stage (local cross-encoder
  `bge-reranker-base` via fastembed). Strategy is a parameter:
  `dense | sparse | hybrid | hybrid_rerank`. `PgChunkStore` implements the
  `ChunkStore` port used by ingestion.
- `ClaudeGenerator` — `anthropic` SDK, `claude-opus-4-8`, adaptive thinking. Builds
  a grounded prompt from retrieved chunks (each tagged with an id), instructs the
  model to answer **only** from context and cite chunk ids; returns text +
  citations + `usage`. One of two files importing `anthropic`.
- `ClaudeJudge` — `claude-opus-4-8`, `client.messages.parse()` with a Pydantic
  rubric (`AnswerScores`: faithfulness/relevance/citation_correctness ∈ [0,1] +
  rationale), adaptive thinking, effort high. The other file importing `anthropic`.
- `db/` — psycopg3 connection pool; a tiny migration runner applying ordered
  `.sql` files; pgvector type registration.

**5. Services (`src/rag/services`)** — application orchestration, pure-ish (depend
only on ports + stores, injected):
- `IngestionService` — load corpus docs → chunk → embed → persist.
- `QueryService` — retrieve (strategy) → optional rerank → generate answer + citations.
- `EvalService` — run the golden set: for each case, retrieve and compute retrieval
  metrics; optionally generate an answer and judge it; aggregate into a `RunMetrics`
  (retrieval averages, answer-quality averages, total cost, latency p50/p95); persist
  the run + per-case results.

**6. API (`src/rag/api`)** — FastAPI, thin routers delegating to services:
- `POST /ingest` (or CLI-only), `POST /query` → answer + citations + retrieved
  chunks, `POST /evals/run` → run id, `GET /evals/runs`, `GET /evals/runs/{id}`,
  `GET /healthz`.

**7. CLI (`src/rag/cli`)** — Typer: `rag ingest`, `rag eval [--strategy] [--judge]`,
`rag eval --gate` (regression gate for CI; non-zero exit on metric drop). The CLI
is how the harness runs headless in CI without the web app.

## 4. Data flow

1. **Ingest:** `rag ingest` loads `corpus/*.md` → chunk → embed (local) → insert
   into `chunks` (embedding + tsvector). Golden set loaded for eval reference.
2. **Query:** question → embed → dense top-k + sparse top-k → RRF fuse → (optional
   rerank) → top-n chunks → `ClaudeGenerator` answers citing chunk ids → API
   returns answer + citations + the chunks.
3. **Eval (retrieval track):** for each golden case, retrieve with the configured
   strategy; a chunk is **relevant** iff it contains one of the case's
   `relevant_snippets` (normalized substring match) — robust to chunk-boundary
   changes. Compute recall@k/precision@k/MRR/nDCG@k; aggregate.
4. **Eval (answer track, key-gated):** generate an answer for each case, judge it
   with Claude (faithfulness/relevance/citation-correctness); aggregate; sum token
   cost; record latencies.
5. **Persist + display:** write `eval_runs` + `eval_case_results`; the dashboard
   reads them and renders metrics, trends, per-case detail, and cost/latency.

## 5. Persistence schema (Postgres + pgvector)

- `documents (id, source, title, content, created_at)`
- `chunks (id, document_id, ordinal, text, embedding vector(384), tsv tsvector,
  token_count)` — HNSW index on `embedding`, GIN index on `tsv`.
- `eval_runs (id, created_at, config jsonb, retrieval_metrics jsonb,
  answer_metrics jsonb, cost_usd numeric, latency_p50_ms, latency_p95_ms,
  git_sha text null)`
- `eval_case_results (id, run_id fk, case_id, strategy, retrieved_ids jsonb,
  recall, precision, mrr, ndcg, answer text null, judge_scores jsonb null,
  cost_usd, latency_ms)`

Migrations are ordered `.sql` files applied by a small runner on startup / `rag
migrate`.

## 6. Corpus & golden set (bundled, reproducible)

- `corpus/docs/*.md` — a small, self-contained, curated technical-docs corpus
  (no licensing concerns; authored/curated for this repo).
- `corpus/golden.yaml` — ~30–50 cases, each: `id`, `question`,
  `relevant_snippets: [str]` (passage-anchored ground truth), `reference_answer`.
- Passage-anchored relevance keeps ground truth stable across chunker changes:
  retrieval metrics are computed by snippet-containment, not fixed chunk ids.

## 7. LLM usage (Claude, key-gated)

- Model: `claude-opus-4-8` for both generation and judging (per Anthropic guidance
  to default to the latest capable model).
- **Generator:** adaptive thinking; grounded system prompt ("answer only from the
  provided context; cite chunk ids; if unsupported, say so"); returns text +
  citations + `usage`.
- **Judge:** `client.messages.parse()` with a Pydantic `AnswerScores` schema
  (structured output), adaptive thinking, effort high; scores faithfulness,
  relevance, citation-correctness in [0,1] with a short rationale.
- **Cost:** a pricing table (Opus 4.8 = $5 / $25 per 1M input/output tokens) turns
  `usage` into USD per call; aggregated per run.
- **Graceful degradation:** without `ANTHROPIC_API_KEY`, ingestion + retrieval +
  retrieval-metric evals run fully; generation and the judge track are skipped with
  a clear message (and the CI retrieval gate still passes).

## 8. Testing strategy

- **Unit (pure, no deps):** metrics (recall/precision/MRR/nDCG against
  hand-computed vectors), RRF fusion, chunking determinism, citation mapping,
  cost calculation, snippet-relevance matching.
- **Integration (real Postgres+pgvector):** dense/sparse/hybrid retrieval against a
  tiny fixture corpus return expected orderings; migrations apply cleanly. Postgres
  provided via a CI service container (or testcontainers locally).
- **Adapter/contract:** `FastEmbedEmbedder` yields stable 384-dim vectors;
  `ClaudeGenerator`/`ClaudeJudge` tested against a **fake Anthropic client** (no key
  in CI) verifying prompt construction, citation parsing, and score parsing; one
  opt-in test hits the real API behind a marker.
- **E2E:** ingest fixture corpus → `rag eval` → metrics computed + persisted → API
  returns them. Playwright smoke for the dashboard (runs list + run detail render).
- The golden-set eval doubles as an integration test and the CI regression gate.

## 9. Tooling & conventions

- **uv** for dependency + venv management; **Python pinned to 3.12** (ML wheels;
  3.14 on the host is too new).
- **mypy --strict** end-to-end; **ruff** (lint + format).
- **pydantic v2** for entities/settings; **psycopg3** + explicit SQL (no ORM) so the
  hybrid-retrieval SQL is visible; `pgvector` Python adapter.
- **FastAPI** + **Typer**; **pytest** (+ `pytest` markers for integration/live).
- Next.js (App Router, TS strict) dashboard; **TanStack Query** for data fetching;
  charts via a lightweight lib (e.g. Recharts).
- **Docker Compose:** `db` (`pgvector/pgvector:pg16`) + `api` (FastAPI) + `web`
  (Next.js). The embedding model is baked into the `api` image at build for
  offline/CI friendliness.
- Commits authored as `Leonidas Augusto Amaro <augustoamaro@proton.me>`; English
  only; no AI attribution. Secrets scan before first push.

## 10. Milestones (build order)

1. Repo scaffold: uv project, ruff/mypy strict, pytest, Next app, compose skeleton,
   CI skeleton.
2. `domain` ports + entities; `metrics` (recall/precision/MRR/nDCG + RRF) — TDD, pure.
3. `chunking` + corpus + golden set authored.
4. DB schema + migrations + `PgVectorRetriever` (dense/sparse/hybrid) — integration
   tests against real Postgres.
5. `FastEmbedEmbedder` + `IngestionService`; `rag ingest`.
6. `EvalService` retrieval track + `rag eval` + regression gate (no key path).
7. `ClaudeGenerator` + `QueryService` + `/query` (citations).
8. `ClaudeJudge` + answer-quality track + cost/latency aggregation.
9. FastAPI endpoints + Next.js dashboard (runs, detail, trends, try-a-query).
10. Docker Compose end-to-end + CI (gate green) + README + secrets scan → public.

## 11. Key decisions (rationale)

- **Local embeddings (fastembed) + Claude key-gated** — the core (ingest, hybrid
  retrieval, retrieval-metric evals) runs from a fresh clone with zero secrets and
  in CI; Claude powers the parts that genuinely need a frontier LLM (answers,
  judging). Best "works from a fresh clone" while showcasing Claude.
- **Hybrid = pgvector dense + Postgres FTS sparse + RRF** — real hybrid retrieval
  with **no extra infrastructure** (one Postgres). RRF needs no score calibration.
- **Passage-anchored ground truth** — snippet-containment relevance is stable
  across chunker changes, so retrieval metrics stay meaningful as the pipeline
  evolves.
- **Eval harness is the spine, retrieval-first** — retrieval metrics need no LLM,
  so the headline quality signal is reproducible and CI-gateable for free; the
  LLM-judge answer track layers on top.
- **psycopg3 + explicit SQL over an ORM** — the hybrid-retrieval SQL (pgvector +
  FTS + RRF) is the interesting part; keeping it visible reads better to a senior
  reviewer than hiding it behind an ORM.
- **Strategy comparison as a first-class eval axis** (`dense`/`sparse`/`hybrid`/
  `hybrid_rerank`) — makes the dashboard a genuine quality lab, not a single-number
  readout.
