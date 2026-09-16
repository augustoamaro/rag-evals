# Dashboard screenshots

Captured on September 16, 2026 from the production Next.js build and the real
FastAPI/PostgreSQL service. Application source:
[16af100](https://github.com/augustoamaro/rag-evals/commit/16af100aa8c2004f29ce0c6aaae62ff5135ae201).
The images are browser captures of the application, with no substituted metrics
or generated interface elements.

| Image | View and data |
|---|---|
| [dashboard.png](dashboard.png) | Four complete English smoke runs at k=3: dense, sparse, hybrid and hybrid plus reranking. History is filtered to their compatible scope. |
| [experiments.png](experiments.png) | English smoke launch controls and completed, separate English/Portuguese development jobs. The answer track is unavailable without a key. |
| [comparison.png](comparison.png) | Hybrid baseline versus hybrid plus reranking, over the same 30 English smoke cases at k=3. Aggregate scores match; the consistency-models slice fails the comparison gate. |
| [evidence.png](evidence.png) | The hybrid run's `cap-ap` case, with its reference, covered evidence, retrieved passages and RRF/dense scores. Generation and judging were skipped. |

The dataset is `rag-engineering`, version `2026-09-16.1`: ten bundled English
documents, 20 chunks, 30 English smoke cases, and ten development probes per
language. A fresh disposable database was ingested and six evaluations were
launched through the production jobs API. All completed with `with_answers=false`;
no model provider was called. The smoke aggregates match the
[retained baselines](../../benchmarks/).

The comparison illustrates an existing trade-off between strategies, not a newly
promoted retrieval change. Displayed latencies come from these local runs and are
not a load-test result. These images do not include the separate SciFact profile
or establish answer quality.

To reproduce the views, follow [Run it](../../README.md#run-it), launch each
strategy with the English smoke split at k=3, then launch separate development
runs for English and Portuguese. Select the smoke benchmark on **Runs**, compare
hybrid with hybrid plus reranking, and open **Baseline evidence** for `cap-ap`.
Run IDs, timestamps and latency will differ.

Captures use a 1440px-wide desktop viewport at device scale 1. The dashboard shows
its complete history panel; the other images show the relevant viewport of a
longer page. All four files have descriptive alt text in the project README.
