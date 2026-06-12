# AGENTS.md — rag-evals

Project-specific notes for AI coding agents. Workspace-level rules live in
`../AGENTS.md` (including: never commit `docs/superpowers/`).

## What this is

A RAG service + evaluation harness. Clean architecture, Python 3.12 (uv), mypy
strict; a Next.js dashboard in `apps/web`. Local embeddings keep the core key-free;
Claude (Opus 4.8) powers answer generation + the LLM-as-judge, gated behind
`RAG_ANTHROPIC_API_KEY`.

## Layout

- `src/rag/domain` — pure entities + Protocol ports (`Embedder`, `Retriever`,
  `Generator`, `Judge`, `ChunkStore`). No I/O, no framework.
- `src/rag/metrics` — pure: recall/precision/MRR/nDCG, RRF, snippet relevance, stats.
- `src/rag/chunking` — pure deterministic chunker.
- `src/rag/adapters` — the only modules with I/O/SDK/SQL: `embedding` (fastembed),
  `retrieval` (pgvector hybrid), `db` (psycopg3), `llm` (Claude generator + judge —
  the only Anthropic SDK *usage*; the CLI/API composition roots construct the
  client and inject it).
- `src/rag/services` — ingestion · query · eval.
- `src/rag/api` — FastAPI; `src/rag/cli` — Typer (`rag ingest`, `rag eval`).
- `corpus/` — bundled docs + `golden.yaml`. `migrations/` — ordered `.sql`.

## Commands

```bash
uv sync
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src
uv run pytest                  # unit (fakes, no DB/key)
uv run pytest -m integration   # real Postgres (set RAG_DATABASE_URL)
uv run rag ingest              # migrate + embed + index corpus
uv run rag eval --strategy hybrid --k 10 --gate
uv run uvicorn rag.api.app:build_app --factory --port 8000
cd apps/web && pnpm install && pnpm dev
```

## Conventions

- Keep the LLM inside `adapters/llm`; keep SQL inside `adapters/db` and
  `adapters/retrieval`; keep the domain pure.
- New `Retriever`/eval behaviour is exercised against real Postgres in
  `tests/integration`; pure logic gets hand-verified unit tests.
- Anthropic SDK calls follow the project default: `claude-opus-4-8`, adaptive
  thinking, structured output (`messages.parse`) for the judge.
- Strict typing (mypy strict + TS strict); ruff for lint + format.
- Commits: English, authored as `Leonidas Augusto Amaro <augustoamaro@proton.me>`,
  no AI attribution.
