# Experiment workflow verification — September 16, 2026

Phase 3 completes the local experiment workflow on top of the versioned indexes
and benchmarks from phase 2. English remains the default; Portuguese is evaluated
separately. This release changes experiment execution and inspection, with no
retrieval tuning or new answer-quality claim.

## Implemented behavior

- **Launch and progress:** `/experiments` selects split/language, strategy, k and
  optional answer evaluation. Availability accounts for both provider configuration
  and reference answers. Jobs retain their configuration and saved progress across
  page reloads. Request IDs prevent duplicate execution, including retries after
  the active index, default split, cases or provider availability changes.
- **Bounded local execution:** one worker owns a dedicated PostgreSQL advisory-lock
  connection; a database constraint admits one active job. Run, case and job
  checkpoints commit atomically. The synchronous endpoint uses the same limit.
  Shutdown retains ownership until the worker exits. Restart fails interrupted
  work without replaying model calls and marks potentially missing cost unknown.
- **Cancellation:** no new retrieval/model stage starts after cancellation is
  observed. A call in flight may finish; its observed answer, judge result, usage
  and known cost are retained. Running, cancelled or failed retrieval prefixes are
  excluded from benchmark comparisons and trend groups.
- **Comparison:** `/compare` validates scope, labels and full case coverage, then
  displays paired aggregate/slice changes, descriptive 95% bootstrap intervals and
  per-case evidence links. It reuses CLI baseline semantics: 0.02 overall/slice
  maximum drop and no evidence-recall regression on critical cases.
- **Review notes:** run and case notes remain separate from labels/results. Atomic
  revision checks reject concurrent overwrites; the interface retains the draft
  and shows the current saved note before an explicit replacement.
- **Actual scores:** stored context and the playground expose cosine similarity,
  PostgreSQL `ts_rank_cd`, RRF sums and cross-encoder output with component scores.
  Scales are named; legacy provenance remains unknown. Custom rank-only rerankers
  expose a null numeric score. Rank order and candidate depth are preserved.
- **Client state:** old requests cannot replace a newer query/run/comparison.
  Ambiguous launch failures retain the request token, including when browser
  storage is unavailable. Query results identify the submitted question/strategy.

## Verification

| Check | Result |
|---|---|
| Ruff lint and formatting | Passed |
| Strict mypy (`src tests`) | Passed; 107 Python files |
| Python unit tests | 267 passed |
| PostgreSQL integration/E2E | 107 passed on a fresh disposable database |
| Web tests | 68 passed across 11 files |
| TypeScript and production build | Passed; `/experiments`, `/compare`, query and run routes built |
| Retrieval baselines | All four strategy baselines and absolute gates passed at k=3 |
| Independent reviews | Backend worker/control, API/annotations/comparison and frontend; no important findings remain |

Database checks cover concurrent admission, idempotent retries, lock ownership,
startup/shutdown races, atomic checkpoint rollback, restart without replay,
cancellation during paid stages, known-cost retention after a failed cancellation
lookup, revision conflicts and actual retrieval score values. Provider behavior
is simulated; these tests incur no model-provider charge.

The four English smoke baselines retain their metrics:

| Strategy | Recall | nDCG | MRR | Evidence recall |
|---|---:|---:|---:|---:|
| Dense | 1.000 | 0.951 | 0.933 | 1.000 |
| Sparse | 0.817 | 0.820 | 0.833 | 0.833 |
| Hybrid | 1.000 | 0.988 | 0.983 | 1.000 |
| Hybrid + reranking | 1.000 | 0.988 | 0.983 | 1.000 |

Each strategy was compared with its own retained phase 2 artifact. Equal aggregate
metrics across different strategies do not imply identical cases: the browser's
hybrid-versus-reranked comparison correctly surfaced a consistency-model slice
drop despite equal overall means. This is an inspection example, not a newly
introduced ranking regression or a promoted retrieval change.

## Browser acceptance

The production Next.js build was served locally and exercised through
browser-harness at 1280px and 390px:

- English default, disabled answer track without a key, successful launch and
  navigation to the saved run; separate Portuguese development launch completed.
- Active job recovered after reload; cancellation retained four completed cases
  in the controlled timing fixture and excluded that prefix from comparison.
- Compatible aggregate/slice/case comparison, explicit incompatible direct-URL
  rejection and case links opening the corresponding stored evidence.
- Run note, case note, concurrent-edit conflict and explicit draft reconciliation
  persisted successfully.
- Query submission identity and real score components were visible. Tables scroll
  within panels on mobile, with no page overflow; newer Portuguese history did
  not displace the default English scope.

Production API composition was checked separately with the actual local index.
For deterministic browser cancellation, an ignored acceptance fixture wrapped
the real retriever with a 600ms delay per call. It changed timing only; it did not
change rankings, labels, persistence or scores. No LLM was enabled. Those timings
are not a performance benchmark.

Compact machine-readable evidence is in
[the verification results](experiments-2026-09-16-results.json). Local screenshots,
API/browser observations, gate artifacts and logs remain under the ignored
`local-data/phase3-verification/` directory.

## Remaining limits

This is a local/demo workflow with one active evaluation and no distributed queue.
Cancellation waits for the current call; deadlines, authentication, quotas and
shared-operation controls remain phase 6. Answer quality and judge calibration
remain phase 5 and require reviewed labels plus configured live access. No paid
calls were made during verification. The evidence records the local checkpoint
before publication. The pre-existing Starlette/httpx test-client deprecation
warning remains.
