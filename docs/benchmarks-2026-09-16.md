# Versioned benchmark measurements — September 16, 2026

Phase 2 adds immutable dataset/index snapshots, stable evidence labels and paired
regression gates. The default hybrid configuration reached **0.841 document
recall@10 and 0.721 nDCG@10** on the complete published BEIR/SciFact test profile.
This is an untuned retrieval baseline. It does not establish generated-answer
quality, judge agreement, or quality on an unseen user's corpus.

Machine-readable [results](benchmarks-2026-09-16-results.json), complete
[per-case baselines](../benchmarks/baselines/) and [dataset documentation](../corpus/README.md)
accompany this report. Runs were measured locally from an uncommitted working tree;
each artifact retains the observed source digest, Git state, dependency versions,
model artifact hashes, dataset identity and immutable index ID.

## Local smoke and development profiles

The original ten documents yield 20 chunks at a 180-word target and 30-word
overlap. The versioned smoke set preserves the original 30 questions, reference
answers and anchors. Word-span conversion, ten existing English paraphrases and
their ten Portuguese counterparts have not been independently reviewed. The
development variants share source evidence groups with smoke cases; they are
diagnostics, not held-out examples. English and Portuguese are separate runs.

All local runs below use **k=3**. Recall, precision, MRR and nDCG rank chunks whose
word intervals intersect qualified labels. Evidence recall instead checks complete
coverage of each labeled span by the union of retrieved intervals. It counts a
fact once even if overlapping chunks contain multiple copies.

| Split / language | Strategy | Cases | Recall | nDCG | MRR | Evidence recall |
|---|---|---:|---:|---:|---:|---:|
| Smoke / English | Dense | 30 | 1.000 | 0.951 | 0.933 | 1.000 |
| Smoke / English | Sparse | 30 | 0.817 | 0.820 | 0.833 | 0.833 |
| Smoke / English | Hybrid | 30 | 1.000 | 0.988 | 0.983 | 1.000 |
| Smoke / English | Hybrid + rerank | 30 | 1.000 | 0.988 | 0.983 | 1.000 |
| Dev / English | Hybrid | 10 | 0.800 | 0.718 | 0.700 | 0.800 |
| Dev / Portuguese | Hybrid | 10 | 0.650 | 0.544 | 0.517 | 0.700 |

The smoke ranking values match the earlier audited behavior. The diagnostic
paraphrases are harder, and the current English embedding/FTS configuration is
less effective on this small Portuguese sample. These ten-case differences are
descriptive; they do not establish a general multilingual effect or justify
changing retrieval configuration without reviewed development data.

## Published external retrieval profile

The [BEIR catalog](https://github.com/beir-cellar/beir) distributes a retrieval
adaptation of [SciFact](https://github.com/allenai/scifact), introduced by Wadden
and colleagues in *Fact or Fiction: Verifying Scientific Claims*. The original
SciFact task also includes evidence/stance annotation; this run evaluates only
the BEIR binary document-relevance labels.

- Full corpus: **5,183 documents**, producing **8,778 chunks**. Each imported
  document is its published title, a newline, and text. Unjudged documents remain
  in the candidate corpus.
- Published train qrels: **809 development cases**, imported but not tuned here.
  Published test qrels: **300 test cases**, all with judged positives. Original
  qrels and source split tags remain in the self-contained dataset snapshot.
- Fixed configuration: `BAAI/bge-small-en-v1.5`, 384 dimensions, 180/30 word
  chunking, exact cosine within the bound snapshot, English `plainto_tsquery`,
  50 candidates per hybrid arm, RRF constant 60, no reranker.
- Document ranking: take the first chunk hit for each unique document from the
  50 fused candidate chunks, retain up to ten unique documents, and evaluate
  binary qrels. This is this project's passage-to-document aggregation pipeline,
  not a reproduction of a paper's full-document encoder setup.

| Split / language | Strategy | k | Document recall | Precision | nDCG | MRR |
|---|---|---:|---:|---:|---:|---:|
| Published test / English | Hybrid | 10 | **0.841** | 0.095 | **0.721** | 0.689 |

Document evidence recall is also 0.841 because the evidence units are documents.
That does not mean a representative chunk contains the complete supporting
passage. No configuration was tuned on these test results. No generator or judge
was called; reference answers are absent, and the answer track rejects this
profile before paid calls. Observed retrieval p95 was 27ms in this local run;
this includes the experiment's runtime conditions and is not a load-test result.

The archive was downloaded from the
[official BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip)
and matched the catalog's MD5. [Recorded source checksums](../benchmarks/scifact-source.json):

```text
Archive bytes: 2816079
MD5:    5f7d1de60b170fc8027bb7898e2efca1
SHA256: 536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165
Dataset SHA256: 6ae00eae5ba7b99750ff4db14cb5f12bb2abcf036e4ec803ae68e2582ba63d74
```

SciFact annotations/claims are CC BY 4.0 and corpus abstracts are ODC-By 1.0;
its code is Apache 2.0. See the [upstream license](https://github.com/allenai/scifact/blob/master/LICENSE.md).
The external corpus is kept under ignored local data, not redistributed in this
repository. Baseline exports retain IDs, metrics and provenance without abstracts.

## Regression policy and verification

The three CI smoke strategies passed both their original recall/nDCG floors and
paired comparisons against the new artifacts. Baselines require the same complete
case set, dataset, split, language, k and metric definitions. Regressions are
checked overall, by tag/language slice, and on protected cases. Confidence
intervals are paired case bootstrap summaries; small slices and related variants
limit inference. A wide interval never waives a configured maximum drop.

Tests exercise schema ambiguity, stale spans, duplicate ranks, non-finite
artifacts, incompatible baselines and preflight before provider calls. Real
PostgreSQL checks cover immutable publication, concurrency/rollback, index
isolation, retained historical replay, evidence persistence and CLI/API agreement.
The dashboard separates incompatible histories and preserves legacy runs with
unknown scope. English is selected by default even if a newer Portuguese run
exists. See [verification totals and remaining phases](improvement-roadmap.md).

Next experiments should select changes on development data, reserve an independent
test set for promotion, and add reviewed domain questions. Sparse query semantics,
multilingual embeddings, candidate budgets, context quality and calibrated answer
grading remain separate measured work; none was silently changed to raise these
test scores.
