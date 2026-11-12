# Retrieval baselines

`baselines/*.json` contain per-case measurements and reproducibility manifests.
They contain no provider calls, model-generated answers or external corpus text.
Artifacts use schema 1 and evaluation manifests use schema 2. Evaluation exports
do not replace a baseline automatically; review a changed artifact as a code
change, including its dataset/model/source identities and individual regressions.

The `smoke-en-{strategy}-k3.json` files are the CI baselines for the versioned
bundled dataset. English and Portuguese development artifacts are separate
diagnostics; their labels have not been independently reviewed. The
`scifact-test-en-hybrid-k10.json` artifact evaluates the entire published BEIR
SciFact corpus and test split using the unchanged default hybrid configuration.
It is an external retrieval baseline, not an answer-quality benchmark or a claim
of a tuned SciFact model. The [measurement report](../docs/benchmarks-2026-09-16.md)
records results and source checksums.

To compare a candidate without replacing the reference:

```bash
uv run rag ingest
uv run rag eval --strategy hybrid --k 3 \
  --baseline benchmarks/baselines/smoke-en-hybrid-k3.json \
  --output local-data/candidate.json --gate --min-recall 0.9 --min-ndcg 0.9
```

Exit 0 means compatible/passing; exit 1 means measured regression, a failed
quality floor or an incomplete judged run; exit 2 means invalid inputs/preflight.
Missing cases and mismatched dataset, split, language, k or metric semantics are
configuration errors, not a passing comparison. `--baseline` enforces its gate
even without `--gate`; the latter adds absolute recall/nDCG floors. Baseline
compatibility is checked before retrieval and paid calls.

The default overall and per-slice tolerance is a 0.02 absolute drop. A `critical`
tag or `--protect-case ID` additionally forbids any evidence-recall loss on that
case. Bootstrap intervals are descriptive paired case resamples; a broad interval
does not waive a regression. Each artifact reports chunk/document metric scope.
Chunk ranking denominators can change with chunking; inspect stable evidence
recall when comparing chunk configurations. Cases without judged positive
evidence have no evidence recall and are excluded from quality aggregates.

Exact snapshot cosine search deliberately avoids filtered global-ANN recall loss
as historical snapshots accumulate. Latencies include model warm-up, depend on
the machine and are not production load-test results. English embeddings and FTS
remain the defaults; do not combine Portuguese results into the English baseline.

## Judge calibration set

`judge-calibration/2026-11-12.json` is a different kind of artifact. It **does**
contain model-generated answers and passages of the bundled corpus: 30 answers from
a live run and 12 copies with one planted defect each. Every item records its
question, retrieved passages, answer, the dimension a defect was built to lower,
the edit that produced it, and the judge's scores and rationale. Human labels are
empty; `rag calibrate-judge --labels` fills them from a JSON file keyed by item id.

```bash
uv run rag calibrate-judge --set benchmarks/judge-calibration/2026-11-12.json
```

That prints how many planted defects the judge scored down and how many clean
answers it flagged, without calling a model. `--score` re-judges every item with
the configured model (paid calls) and overwrites the stored scores, so review the
diff as you would a baseline change. The [judge report](../docs/judge-2026-11-12.md)
records the results and their limits.
