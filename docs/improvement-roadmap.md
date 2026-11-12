# Improvement roadmap

This roadmap follows the [September 16, 2026 audit](evaluation-2026-09-16.md).
The audit records the original revision; this document tracks subsequent work.
Keep the local, key-free retrieval baseline and the existing ports/adapters.

Phases 1–3 are merged into `main` in
[16af100](https://github.com/augustoamaro/rag-evals/commit/16af100aa8c2004f29ce0c6aaae62ff5135ae201).
The verification sections below retain their measurements from each local checkpoint.

| Phase | Status | Deliverable and acceptance |
|---|---|---|
| 1. Evaluation reliability | Completed and merged, September 16, 2026 | Honest stage outcomes, coverage and cost; atomic corpus synchronization; transactional migrations; common CLI/API manifests; visible failures and retained case evidence. Reproduced failures are regression tests. |
| 2. Representative benchmarks | Completed and merged, September 16, 2026 | Immutable dataset/index snapshots and embedding provenance; stable evidence labels; strict datasets, separate English/Portuguese slices, published SciFact test profile and paired baseline gates. Reviewed project-domain labels remain a follow-up. |
| 3. Experiment workflow | Completed and merged, September 16, 2026 | Dashboard launch/progress/cancel, compatible paired comparisons, run/case notes, real retrieval scores and stale-response handling. Local worker admission, stage checkpoints and restart recovery verified; shared operation remains phase 6. |
| 4. Retrieval experiments | Planned | Measure lexical query handling, context-aware chunks, candidate depth, token budgets and reranking; evaluate multilingual models if Portuguese is a target. Promote only measured held-out gains without critical slice regressions. |
| 5. Answer and judge validation | Started, November 12, 2026 | First live run and a planted-defect check of the judge are [reported](judge-2026-11-12.md); human agreement is not measured yet. Remaining: separate generator/judge configurations, calibrated correctness/completeness/abstention rubrics, deterministic citation checks, human agreement analysis, repeated trials and budgets. Requires configured live access and reviewed labels. |
| 6. Shared operation | Planned | Authentication, quotas, bounded durable jobs, deadlines, readiness/telemetry, access controls, dependency remediation/automation, Compose CI and load/ANN benchmarks. Verify limits and failure recovery before shared billable operation. |

## Phase 1 implementation

- Generation and judging have independent outcomes, durations, usage and costs.
  Judge failure retains the answer and known generation cost. SDK response parsing
  is exercised with an offline HTTP transport, including refusals and truncation.
- Runs expose attempted/scored/failed counts. Means remain conditional on scored
  cases; all-failed runs have no answer aggregate. Incomplete judged CLI runs are
  persisted and exit 1; unavailable answer configuration exits 2.
- Ingestion prepares and validates every embedding before replacing the active
  corpus. Empty/omitted documents disappear, concurrent writers serialize and
  database failures roll back. Migration DDL and ledger entries commit together.
- CLI/API use one manifest schema with allowlisted configuration, runtime/code
  identity and input content hashes. Production eval preflight rejects an index
  that differs from the expected bundled chunks.
- History stores case questions, reference answers and observed retrieved context.
  The dashboard exposes coverage, partial/failed states, stage errors and known
  costs. Historical runs remain explicitly unknown. Tables scroll on narrow views.

At the phase 1 checkpoint, the system recorded a preflight content identity and observed context.
It did not provide an immutable vector snapshot across concurrent retrievals,
verify which embedding weights produced an existing index, or replace snippet
labels with stable evidence units. Those parts of the original first milestone
moved together into phase 2 to avoid introducing a partial dataset versioning model.
The snapshot implementation below now supersedes that limitation for production
CLI/API retrieval; the legacy mutable adapter remains available to existing callers.

## Phase 1 verification checkpoint

| Check | Result |
|---|---|
| Ruff lint/format and strict mypy (`src tests`) | Passed; 73 Python files |
| Python unit tests | 74 passed |
| Real PostgreSQL integration/E2E tests | 30 passed, including migration races, concurrent replacements, rollback and partial/all-failed eval API persistence |
| Web typecheck, component tests, production build | Passed; 8 tests |
| Fresh index + CLI retrieval gates at k=3 | Exactly 20 chunks; dense, sparse, hybrid and hybrid-rerank gates passed |
| Production CLI/API run manifests | Identical for the same hybrid/k=3 configuration |
| Browser via browser-harness | Production build at 1280px and 390px: reliability states, partial case inspection, retained answer/usage, all-failed empty aggregate, legacy detail and scrollable mobile table links passed |
| Independent code review | No outstanding critical or important findings after fixes |

The four retrieval strategies retained the audited recall/nDCG values on the
bundled golden set: dense 1.000/0.951, sparse 0.817/0.820, hybrid 1.000/0.988 and
hybrid-rerank 1.000/0.988. This verifies behavior preservation on the smoke set,
not a new claim of general retrieval quality.

Local verification scripts, compact results and browser screenshots are under the
ignored `local-data/phase1-verification/` directory. Browser answer-failure fixtures
use simulated LLMs with real persistence; no provider was charged. A pre-existing
Starlette/httpx test-client deprecation warning remains. Dependency versions and
the dependency-audit findings were not changed in this reliability release.

Live Anthropic quality and billable spending require a configured project key;
offline failure-path checks do not establish live answer quality or judge agreement.

## Phase 2 implementation

- `corpus/benchmark.yaml` declares documents, qualified word-span/document evidence,
  split, language, tags, answerability and evidence groups. Strict YAML/JSON/BEIR
  import rejects ambiguous fields, duplicate IDs, invalid/stale spans, unknown
  references, unsupported qrels and development/test group leakage. Missing group
  IDs normalize to case IDs, and equivalent label ordering keeps the same identity.
- English remains the default: smoke 30 and development 10. Portuguese development
  has ten separate cases over the English corpus. Their assistant-authored labels
  and translations are explicitly unreviewed; no local held-out claim is made.
- Snapshot publication validates complete dataset/chunk/vector coverage, seals
  rows against later mutation and retains old datasets and indexes. Actual model,
  tokenizer/configuration artifacts, runtime identity and float32 vectors determine
  the index identity. An old index can be replayed after local files change or
  disappear; incompatible query embeddings fail preflight.
- Evidence recall unions intervals so overlap copies count once. Chunk ranking
  metrics stay explicit; document qrels use deduplicated document rankings from
  50 candidate chunks. Cases without positive evidence are excluded from retrieval
  quality, with explicit eligible counts and N/A evidence scores.
- CLI supports dataset/index/split/language selection, atomic per-case exports and
  paired baselines. Overall/slice tolerances and protected cases supplement the
  existing floors. Incompatible baselines fail before retrieval or provider calls;
  CI retains candidate artifacts for review.
- API uses the same snapshot composition and exposes split/language selection.
  Persistence retains evidence metadata and slices. The dashboard selects compatible
  histories, prefers English, separates legacy records, and exposes coverage and
  word offsets. Evaluations through the legacy adapter retain their chunk metrics
  and leave versioned evidence metadata unknown. Phase 3 below now adds execution
  controls and richer comparisons.
- The complete published BEIR/SciFact corpus was ingested: 5,183 documents,
  8,778 chunks and 300 held-out retrieval test cases. Fixed hybrid@10 reached
  recall 0.841 and nDCG 0.721; no test-set tuning or model-provider calls occurred.
  [Measurements, source checksums and limitations](benchmarks-2026-09-16.md) accompany
  seven per-case baseline artifacts.

## Phase 2 verification

| Check | Result |
|---|---|
| Ruff lint/format, strict mypy (`src tests`), diff whitespace | Passed; 91 Python files |
| Python unit tests | 223 passed |
| Real PostgreSQL integration/E2E | 69 passed, including immutable publication, SQL races, historical replay, CLI/API agreement and preflight before model calls |
| Web TypeScript, component/scope tests, production build | Passed; 39 tests |
| CI commands reproduced locally | Dense, sparse and hybrid passed absolute floors and paired smoke baselines at k=3 |
| External profile | Full SciFact corpus/test split evaluated; archive checksum verified; per-case artifact validated |
| Browser via browser-harness | Production build at 1280px and 390px: English default despite newer PT run, language/benchmark switching, separate legacy history, scrollable mobile tables, PT evidence/offsets and 300-case external detail passed |
| Independent reviews | Backend/data/gates and dashboard scope review: no outstanding critical or important findings |

The external archive, API verification and screenshots are in ignored
`local-data/phase2-verification/`. Public baseline JSON contains identifiers,
metrics and provenance, not external corpus text. Disposable PostgreSQL databases
were used for tests; no user database was truncated. No commits or pushes had been
made at that verification checkpoint. A pre-existing Starlette/httpx test-client
deprecation warning remains;
dependency remediation and full Compose deployment/load verification remain
phase 6. Live answer quality still requires configured access and reviewed labels.

## Phase 3 implementation and verification

The dashboard now launches evaluations, polls durable stage progress, cancels
without discarding observed results, compares compatible runs and saves run/case
notes with revision conflict handling. True retrieval scores and component
provenance replace synthetic reciprocal-rank values. English and Portuguese stay
separate, incomplete prefixes stay out of comparisons, and stale requests cannot
replace current results.

Local execution admits one job under a database worker lock and an active-job
constraint. Checkpoints are atomic, retries are idempotent across runtime changes,
and restart fails interrupted work without replaying paid calls. Distributed
workers, auth, deadlines and quotas remain phase 6.

Verification: **267 unit tests, 107 real PostgreSQL integration/E2E tests and 68 web
tests passed**. Ruff, strict Python/TypeScript checks and the production build
passed. All four English smoke strategies retained their paired baselines at k=3.
Browser checks covered desktop/mobile launch, reload/cancel, comparison, notes and
query scores. Independent reviews have no important findings remaining.

See the [phase 3 verification report](experiments-2026-09-16.md) for evidence and
limits. No model-provider charges were incurred. The evidence describes the local
checkpoint before publication. Phase 4 next measures retrieval changes against
these retained baselines.
