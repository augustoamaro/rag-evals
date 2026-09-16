**rag-evals — technical and evaluation audit, September 16, 2026**

Audited revision: `f6c2e3659103f7759179d6148c5d9576d2193f6d`.

The project has a working, well-tested engineering foundation for a local RAG evaluation demo. Its current evidence does **not yet establish reliable answer quality, general retrieval quality, or readiness for a public service with billable LLM access**. The highest-value investment is trustworthy evaluation data and failure accounting, followed by representative datasets and a usable experiment workflow.

This audit includes executed tests, a fresh container build and smoke test, browser interaction, 600 measured retrieval queries, fault reproductions, dependency audits, and primary-source research. Findings below distinguish reproduced behavior from design limitations and proposed experiments. Production implementation files were not changed.

**Verification and scope**

| Check | Observed result |
|---|---|
| Ruff lint and formatting | Passed; 66 files formatted |
| Mypy strict, `src tests` | Passed; 66 source files |
| Python unit suite | 47 passed |
| PostgreSQL integration/E2E suite | 13 passed with real PostgreSQL 16 and pgvector 0.8.2 |
| Live Anthropic tests | 2 skipped: no `RAG_ANTHROPIC_API_KEY` in environment or project `.env` |
| Web TypeScript, Vitest, production build | Passed; 2 component tests; Next.js 15.5.19 |
| CLI quality gates | Dense, sparse, and hybrid passed the current CI thresholds at `k=3` |
| Container build and runtime | Both images rebuilt; database, API and dashboard started; query → eval → persistence → detail API passed |
| Browser | Runs list, run detail and playground exercised through browser-harness; mobile clipping and stale result behavior reproduced |
| Credential-pattern scan | No matches in tracked files or reachable commit diffs for the selected token/private-key patterns |
| Dependency audits | Findings present; see the dependency section |

The Docker socket was inaccessible to this session. Docker Compose 5.5.1 was therefore used through Podman's Docker-compatible socket. The existing Dockerfiles and Compose file were exercised with an isolated project, fresh database volume, loopback ports, empty LLM key, and the corresponding frontend API URL build argument. This verifies the container stack under that runtime; native Docker Engine was not independently tested.

LLM generation quality, judge agreement, prompt-injection resistance, live token spending, high-concurrency capacity, large-corpus ANN recall, and production infrastructure were not measured. The credential scan was pattern-based, not an exhaustive secret-detection or credential-validation audit. Browser checks were focused functional checks, not a full accessibility certification. Deprecation warnings from Starlette's test client and Hugging Face were present without failing the suites.

**Retrieval results**

The final benchmark used a fresh database containing exactly the bundled **10 documents / 20 chunks / 30 golden cases**. A preflight compared every indexed chunk ID and text with the chunked corpus. Four strategies were measured at `k = 1, 3, 5, 10`, followed by three paired 10-question diagnostic sets at `k=3`: 480 + 120 measured retrievals.

Models were `BAAI/bge-small-en-v1.5` and `Xenova/ms-marco-MiniLM-L-6-v2`. Each strategy received a warm-up before one sequential pass per configuration. Latencies are local smoke measurements rounded by the existing service, exclude model initialization and LLM calls, and are not throughput or production SLO estimates.

| Strategy | Recall@3 | Precision@3 | MRR over returned top 3 | nDCG@3 | Retrieval p50 | Retrieval p95 |
|---|---:|---:|---:|---:|---:|---:|
| dense | 1.000 | 0.367 | 0.933 | 0.951 | 3 ms | 6 ms |
| sparse | 0.817 | 0.289 | 0.833 | 0.820 | 1 ms | 1 ms |
| hybrid | 1.000 | 0.367 | 0.983 | 0.988 | 4 ms | 5 ms |
| hybrid_rerank | 1.000 | 0.367 | 0.983 | 0.988 | 193 ms | 238 ms |

Recall@1 was 0.817 / 0.800 / 0.917 / 0.917 respectively. At `k=5` and `k=10`, retrieval quality did not improve over `k=3` on the original cases; precision fell as more irrelevant chunks were included. The README's published `k=10` quality numbers were reproduced. Sparse search returned no results for 5/30 original cases.

The small corpus makes interpretation especially important. The default hybrid candidate depth is 50 per arm, larger than the entire corpus. An `EXPLAIN` of the dense candidate query on this database returned `Limit → Sort → Seq Scan`: this run does not demonstrate HNSW performance at scale.

The diagnostic questions were manually reformulated/translated for this audit, reused their originals' relevance labels, and were not tuned after observing results. They are a small exploratory sample rather than an independent benchmark. Portuguese is an exploratory capability probe; the current corpus, lexical configuration and embedding model target English.

| Paired set, 10 cases each | Dense recall@3 | Sparse recall@3 | Hybrid recall@3 | Hybrid + rerank recall@3 |
|---|---:|---:|---:|---:|
| Original English questions | 1.000 | 0.800 | 1.000 | 1.000 |
| English reformulations | 0.800 | 0.100 | 0.800 | 1.000 |
| Portuguese translations | 0.650 | 0.000 | 0.650 | 0.600 |

On reformulations, sparse returned nothing for 9/10 cases; hybrid nDCG@3 fell from 1.000 to 0.718, while reranking reached 0.932. Hybrid missed the labeled passages for the lexical-mismatch and cross-encoder questions. Reranking recovered both. On Portuguese, the English reranker did not improve retrieval. These observations justify measuring reranking and language support by slice instead of declaring one strategy universally superior.

A paired percentile bootstrap on the original 30 cases, 10,000 resamples with seed `20260916`, estimated hybrid-minus-dense mean nDCG@3 at +0.0369, with a 95% interval of [0.0000, 0.0861]: only 3 cases improved and 27 tied. Rerank-minus-hybrid averaged 0.0000, interval [-0.0369, 0.0369], with 1 win, 28 ties, and 1 loss. These conditional intervals cannot correct the small, correlated, unrepresentative dataset. They support restrained claims about these cases, not general superiority.

An initial pilot exposed another issue: running integration tests and then `rag ingest` in the same disposable database left 3 fixture chunks behind, producing 23 total chunks. Those pilot measurements were excluded and retained separately. The complete experiment was rerun on the clean database; the Portuguese reranker result changed from 0.400 in the contaminated pilot to 0.600 in the clean run. This is direct evidence that index state belongs in the experiment manifest.

**Findings that affect trust in evaluation**

| Priority | Evidence | Finding and required behavior |
|---|---|---|
| P1 | `src/rag/services/eval_service.py:71`, `:130`; `src/rag/cli/main.py:78` | **Failed answer cases disappear from score averages.** A fake-driven run with 29 judge failures and 1 successful perfect score reported 1.0 in all three dimensions. Preserve successful-case quality, but expose attempted/scored/failed counts and an explicit coverage policy; incomplete coverage must not silently qualify as a healthy answer eval. |
| P1 | `src/rag/services/eval_service.py:72`; `src/rag/adapters/db/eval_store.py:22` | **Judge failure loses successful generation and its usage.** The reproduction recorded simulated cost $0.0015 although successful generation alone accounted for $0.0300. Record each stage's output and known usage before the next stage; distinguish unknown cost from zero. A fully failed answer track becomes `{}`, then SQL `null`, making it indistinguishable from retrieval-only. |
| P1 | `src/rag/cli/main.py:81` | **All answer failures crash CLI reporting.** Reproduced `KeyError('faithfulness')` after a run with an empty metrics dictionary. Persist explicit failed/partial status and print a useful summary with a deliberate exit policy. |
| P1 | `src/rag/metrics/relevance.py:15`; `src/rag/services/eval_service.py:49` | **Relevance is not stable across chunking.** At 180/30 words, 3 cases label overlap copies as independent positives; at 50/0, 5/30 cases lose every relevance anchor. Define stable document/span or evidence-unit labels, version them, and score evidence coverage independently from redundant chunks. Retain chunk-ranking metrics as a separately defined measure. |
| P1 | `src/rag/services/ingestion_service.py:15`; `src/rag/adapters/db/chunk_store.py:15` | **Index contents retain deleted or empty documents.** Re-ingesting an empty document reported 0 chunks but left its old chunk retrievable. Ingesting an empty document list also retained it. Add corpus-scoped synchronization, removal handling and immutable index snapshots; check expected corpus identity before eval. |
| P1 | `src/rag/api/app.py:96`; `src/rag/cli/main.py:63` | **Run provenance differs across entry points and is incomplete.** Real API run config was only `{strategy, k}`, with `git_sha=null`. Neither path records a complete dataset/index/prompt/model manifest or requested answer track. Centralize manifest construction and persist immutable inputs/results. |
| P1 | `src/rag/services/gate.py:6`; `.github/workflows/ci.yml:57` | **The current gate checks absolute quality floors.** A metric can regress from 0.99 to 0.91 and still pass a 0.90 floor. Add paired baseline deltas, protected critical cases and slice thresholds, while retaining absolute floors. |
| P2 | `src/rag/metrics/retrieval.py:29` | **Duplicate result IDs violate metric invariants.** `ndcg_at_k(['a','a'], {'a'}, 2)` returns 1.63093. Current production retrieval yields unique IDs, so this is a metric/port robustness bug rather than a demonstrated normal-query failure. Reject duplicates or deduplicate by first occurrence; enforce score bounds with meaningful invariant tests. |
| P2 | `src/rag/services/pricing.py:13` | **Unknown model pricing silently becomes $0.** Reproduced with nonzero token usage. Use an explicit unsupported/unknown state and versioned pricing; support separate generator and judge model identities. |

The current 30 cases contain only 19 distinct relevance sets. Twenty-seven cases have one relevant chunk; the other three have two copies of one passage due to overlap. For single-positive cases, recall@k and precision@k convey almost the same rank information, while MRR and nDCG transform the first hit's rank. Four headline numbers therefore provide less independent evidence than they appear to.

The answer judge measures faithfulness, relevance and citation correctness. It does not explicitly measure completeness, reference-answer correctness, correct abstention, contradictory sources, or coverage of claims by citations. Invalid citation IDs are silently removed from the structured citation list while remaining in answer text. Add deterministic citation-validity accounting and a rubric for correctness/completeness and unsupported-question handling. Calibrate the judge on blinded human labels; using the same model for generation and grading is a bias risk to measure, not proof that every grade is wrong.

The corpus lacks substantive tests for hard negatives, ambiguous requests, multiple supporting documents, unanswerable questions, version conflicts, adversarial instructions in retrieved text, metadata/permission filtering, and noisy user queries. `token_count` is actually a whitespace word count. SQL distance/rank ties have no explicit stable tie-break policy. The golden loader lacks schema, duplicate-ID and intended-document validation. These are bounded follow-ups to the evaluation data model, not reasons to replace the clean architecture.

**Operational, security and product findings**

| Priority | Evidence | Finding and recommended change |
|---|---|---|
| P1; deployment blocker for a public paid API | `src/rag/api/app.py:54`, `:67`, `:88`; `docker-compose.yml:28` | Query/eval endpoints have no authentication, rate limit, concurrency limit or application spending budget. A preflight from an untrusted origin receives `Access-Control-Allow-Origin: *`. Bind local demos to loopback; use authentication, bounded execution and exact permitted origins for shared deployments. CORS is not authentication. No external exploitation was attempted. |
| P1 | `src/rag/adapters/db/migrations.py:20` | Migration SQL and ledger writes are separate autocommit operations. A forced ledger failure left the created table committed; retry failed with `DuplicateTable`. Use a transaction for each applicable migration and ledger entry, plus an advisory lock for concurrent runners. Existing migrations happen to be mostly idempotent. |
| P1 | `src/rag/api/app.py:63`; `Dockerfile.api:35`; `docker-compose.yml:38` | `/healthz` is liveness only and always returns OK. Startup re-embeds the corpus; web waits for API container start, not API readiness. Add DB/index/model readiness, health-based dependencies and a one-shot initialization job before multiple replicas. |
| P1 | `apps/web/lib/api.ts:1`; `docker-compose.yml:35` | The browser API URL is baked as localhost by default. Remote visitors would target their own machine unless the image is rebuilt correctly. Prefer a same-origin proxy or validated runtime configuration. |
| P1 | `apps/web/components/TrendChart.tsx:15`; `src/rag/adapters/db/eval_store.py:65` | One trend line mixes strategy, k and dataset configurations; the API and UI load every run. Group comparable experiments, identify baselines/configuration, add filtering and cursor pagination. |
| P2 | `apps/web/app/try/page.tsx:15`, `:64` | After a successful query, submitting an empty question gets API 422 but UI says “Could not reach the API” and retains all 5 old chunks. Clear or explicitly label stale results, display structured validation/server errors, and support cancellation/timeouts. |
| P2 | `apps/web/app/globals.css:159` | On a 390 px emulated device the table measured about 654 px inside a 332 px panel with `overflow: hidden`; rightmost data/actions were clipped. Add an accessible scrolling wrapper or responsive presentation. Review visible keyboard focus and error announcements. |
| P2 | `src/rag/config.py:10`; `migrations/0001_init.sql:16` | Embedding model/dimension configuration can diverge from fixed `vector(384)` storage. Validate the model output dimension and index manifest at startup; require a new index when changing embedding space. |
| P2 | `src/rag/adapters/db/chunk_store.py:20` | Updating a document changes content only. Reproduction retained old `source` and `title` with new content. Update mutable metadata together with the indexed document version. |

The dashboard currently lets users view past runs and submit queries, but starting an eval requires CLI/API knowledge. There is no case drill-down showing question, reference, full retrieved context, answer, judge rationale and errors together; no baseline diff; and no corpus ingestion/status workflow. These are the most consequential product gaps for making the harness useful to someone beyond its author. The existing visual style is coherent, but the experiment workflow is incomplete.

CI proves the native Python and Next.js paths, but does not currently build and smoke-test Compose, audit dependencies, run browser E2E flows, or publish immutable eval artifacts. Containers run as root; demo database ports/default credentials are exposed through host bindings. Tighten deployment defaults and use least-privilege CI/container identities as part of the public-service work.

**Dependency audit, with applicability**

The complete Node audit reported **31 findings across 29 advisory entries** (3 critical, 14 high, 14 moderate). Production dependency filtering reported **21 findings** (2 critical, 11 high, 8 moderate). Python's audit of 68 installed distributions reported **28 records across 3 packages**, containing **15 distinct advisory IDs** after removing repeated IDs: Pillow 12.2.0, pydantic-settings 2.14.1 and Starlette 1.3.0. These counts are scanner output, not counts of demonstrated exploitable application paths.

Next.js 15.5.19 is below the fixes in 15.5.24 for the reviewed Windows-hosting and AVIF image-optimization advisories. The shipped runtime is Linux and this application has no configured image-optimization workflow, Server Actions or rewrites. The critical conditions were not demonstrated reachable in this audit. Update to a compatible patched release and re-audit the final lockfile. [Next.js AVIF advisory](https://github.com/advisories/GHSA-2xp9-vwfh-vxw4), [Windows-hosting advisory](https://github.com/advisories/GHSA-p293-qw3h-jr36).

Vitest 2.1.9 has a critical advisory requiring an exposed UI/API or affected browser-mode conditions; this project runs Vitest in terminal/jsdom mode. Updating only to the critical fix is insufficient for all newer findings: the local audit also reports a later Vitest/mocker fix at 4.1.11. Plan a compatible test-tooling update and rerun the audit. [Vitest critical advisory](https://github.com/advisories/GHSA-5xrq-8626-4rwp).

The Python findings require image/font operations, a particular nested-secrets directory mode, or form parsing. The current service is text/JSON based and does not expose those paths. The scanner lists fixes at Pillow 12.3.0, pydantic-settings 2.14.2 and Starlette 1.3.1; confirm resolver compatibility and rerun relevant tests when refreshing the lock. No dependency upgrade was applied during this assessment.

**What documented frontier-lab practices suggest**

The recommendations below are an application of public engineering guidance to this repository. They do not claim access to a lab's private production stack or imply that adopting a particular framework creates equivalent rigor.

| Primary source | Documented practice | Concrete application here |
|---|---|---|
| [Anthropic: Demystifying evals for AI agents, Jan 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Separate capability and regression suites; combine deterministic, model and human graders; retain trials and execution evidence. | Keep the existing small suite as a fast regression smoke set; add harder capability cases, explicit trial outcomes, calibrated graders and inspectable artifacts. |
| [OpenAI: Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) | Task-specific data reflecting actual usage, continuous evaluation, human calibration and targeted comparisons. | Turn observed failures and user feedback into versioned cases; keep a held-out split; compare candidate and baseline on identical cases and report slices. |
| [Google DeepMind: FACTS suite, Dec 2025](https://arxiv.org/abs/2512.10791) and [FACTS Grounding](https://arxiv.org/abs/2501.03200) | Evaluate factuality and grounding explicitly; validate automated graders and mitigate judge bias. | Score unsupported claims, completeness and grounding separately; validate the judge against human annotations and test wrong, incomplete and abstaining answers. |
| [Anthropic: Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) | Evaluate contextualized chunks, lexical+dense retrieval and reranking together, including cost/latency tradeoffs. | Run controlled ablations for title/section context, lexical query semantics, candidate depth and reranking. Published gains on other datasets are not forecasts for this project. |
| [Anthropic: Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Select relevant context carefully and manage its finite budget. | Measure evidence coverage per input token, duplicate passages, context size and the effect of retrieval noise on answer quality. |
| [BEIR reference implementation](https://github.com/beir-cellar/beir) | Reproducible retrieval comparisons using separate corpus, query and qrels files and saved result artifacts. | Add an optional external corpus profile such as SciFact, preserving its labels/splits and comparing at document level where appropriate. Keep the key-free smoke profile for fast CI. |
| [OpenTelemetry GenAI span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | Structured telemetry for model and retrieval operations. | Trace query embedding, dense/sparse retrieval, fusion, reranking, generation and judging; link trace IDs to eval cases and record errors, usage and durations. Pin the convention version because the schema evolves. |

The configured Opus 4.8 standard token rates ($5 input / $25 output per million tokens) match the official pricing page at the audit date. The gap is incomplete usage/error accounting and single-model configuration, not evidence that this price entry is wrong. Select model changes through quality/cost experiments and a live compatibility check. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).

**Implementation order and acceptance criteria**

These are proposed project milestones; dataset sizes and thresholds below are starting points to refine with the intended users and domain.

1. **Make each reported result trustworthy.** Add stage statuses, attempted/scored/failed counts, known/unknown usage accounting, safe CLI reporting and full run manifests. Scope indexes to immutable corpus snapshots; fix empty/deleted-document synchronization and transactional migrations. Acceptance: the failure reproductions become regression tests; no all-failed run is labeled healthy; CLI/API manifests match; stale-index preflight fails clearly.

2. **Create a benchmark capable of distinguishing improvements.** Add stable evidence labels, schema validation, development and held-out test splits, and tags for difficulty/language/answerability. Start with roughly 200–500 reviewed cases for the intended domain, with meaningful coverage per critical slice; keep the 30-case set for smoke CI. Add an optional external retrieval profile. Acceptance: a held-out, per-case baseline artifact exists; changing chunk boundaries does not silently rewrite evidence coverage; critical missing labels invalidate the run; a simulated meaningful regression is detected even above the absolute floor.

3. **Make the dashboard an experiment workbench.** Add a “Run evaluation” workflow with dataset/strategy/k selection, queued/running/partial/failed/completed status, progress and cancellation. Add run comparison, case drill-down, retriever-stage scores, source inspection, errors and annotation/feedback. Fix mobile tables and stale results. Acceptance: a user can launch an eval, inspect a failed case and compare two compatible runs entirely in the UI; browser tests cover those flows and failure states.

4. **Run controlled retrieval improvements.** Start with lexical query handling because reformulation failures are demonstrated; compare the existing `plainto_tsquery` behavior against a measured alternative and a BM25 baseline if justified. Then compare metadata-aware or contextual chunks, candidate depth, token budgets and reranking. For Portuguese support, evaluate a compatible multilingual embedder and reranker, with a separately versioned vector index. Acceptance: promote a change only when the held-out quality gain is worth its measured latency/cost and no protected slice regresses.

5. **Validate answer quality and judge reliability.** Use a dedicated generation/judge configuration, calibrated rubrics, correctness/completeness/abstention metrics, claim-level citation checks and a human-labeled calibration subset. Repeat a subset to estimate judge variability; consider a second judge for disagreements. Add explicit budgets and versioned prompts. Acceptance: a live run records model/request IDs and stage usage, meets a declared coverage policy, and publishes agreement/error analysis against human labels. This milestone needs configured LLM access and human annotations.

6. **Prepare shared operation.** Add authentication, per-user budgets/limits, a bounded worker queue, request deadlines, readiness checks, telemetry, dataset access controls if private documents are supported, and same-origin API routing. Add Compose smoke tests, dependency/security jobs and update automation. Benchmark exact versus ANN search on a larger corpus and performance under controlled concurrency. Acceptance: duplicate submissions are controlled; eval work survives HTTP disconnects; access cannot cross corpus permissions; startup and failure states are visible; load and cost limits are enforced and tested.

Preserve the existing pure domain, explicit ports, local key-free retrieval baseline and visible SQL. Extend the system with dataset/index manifests, run/trial state, artifact storage and telemetry adapters. Multi-step agentic retrieval can be a later experiment when multi-hop cases show a need; it would require separate tool-use, stopping, outcome and budget evals.

**Reproducibility and retained evidence**

The compact machine-readable result is [evaluation-2026-09-16-results.json](evaluation-2026-09-16-results.json). Detailed local evidence is retained in the ignored `local-data/evaluation-2026-09-16/` directory: `benchmark.py`, `benchmark.json`, `benchmark.log`, `reproduce.py`, `reproductions.json`, package-audit JSON, API/Compose smoke results, query-plan summary and browser screenshots. The `pilot-contaminated-*` files are explicitly excluded from the final benchmark.

Golden YAML SHA-256: `cab9faf43b0c54e99fa44274a0c12e28cd12ac032a75148f144f42c836dcbe95`.

Chunk-text digest in the audit's sorted chunk order: `5b0a93809fb57fe063231d58dd95a5553f95ce86cd92ae4f7f86369e7b40a77a`. This digest is an audit check, not a replacement for the proposed canonical manifest format.

The benchmark script expects `RAG_DATABASE_URL` to point to a fresh, ingested database and rejects extra/mismatched chunks. The reproduction script intentionally exercises migration failures and writes only to its disposable `audit_probes` database; recreate that database before repeating it. Integration tests must always use a disposable `RAG_TEST_DATABASE_URL` because the fixtures truncate tables.
