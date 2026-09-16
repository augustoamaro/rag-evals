# Versioned evaluation datasets

[`benchmark.yaml`](benchmark.yaml) is the local dataset, `rag-engineering` version
`2026-09-16.1`. It uses the ten bundled English documents. The original
[`golden.yaml`](golden.yaml) remains a legacy fixture.

| Split | Language | Cases | Purpose |
| --- | --- | ---: | --- |
| smoke | English | 30 | Original questions, reference answers and snippet anchors |
| dev | English | 10 | Existing audit paraphrases |
| dev | Portuguese | 10 | Existing audit translations over the English corpus |

English is the base evaluation language. Portuguese is a separate diagnostic
slice; do not combine its scores with the English headline results. The
development probes and word-span conversion were assistant-authored and have not
been independently reviewed. These small, local cases make no held-out or domain
generalization claim. There is no local `test` split. Published external retrieval
labels are evaluated separately through the BEIR import described below.

## YAML schema

The file declares integer `schema_version: 1`, string `name` and `version`,
`default_split`, string-valued `provenance`, and three lists:

- `documents`: unique string `id` plus a relative `path`; alternatively, an inline
  record with `id`, `source`, `title` and `content`. Paths must stay inside the
  dataset directory. Snapshots always contain inline documents, so loading a
  historical snapshot does not depend on the original files.
- `evidence`: unique `id`, `document_id`, optional `word_start`/`word_end`, and
  `quote`. Word offsets use Python `content.split()`, including Markdown headings
  and punctuation. The start is inclusive and the end is exclusive. The quote
  must match the span after whitespace normalization, preserving case and
  punctuation. Both offsets absent, with an empty quote, means document-level
  evidence.
- `cases`: unique `id`, `question`, `reference_answer`, `evidence_ids`, `split`,
  `tags`, `language`, `answerable` and `group_id`. `relevant_snippets` preserves
  legacy anchors; versioned evaluation uses the qualified evidence labels. Empty
  reference answers are allowed for retrieval-only datasets. Empty evidence lists
  require `answerable: false` and have no retrieval-quality denominator. Omitted or
  empty group IDs default to the case ID; variants must explicitly share a group.

Allowed splits are `smoke`, `dev` and `test`. Missing selected splits, invalid
types, duplicate IDs, unknown references, stale quotes, invalid offsets and group
overlap between development and test cases are rejected. A semantic variant keeps
the same `group_id` as its source question; the local English/Portuguese probes
therefore keep their smoke case's evidence group.

`dataset_payload()` produces a self-contained snapshot. `dataset_digest()` hashes
the full content and metadata, including labels, group IDs, language and split.
Changing those values changes its identity; rearranging documents, cases, evidence,
tags or evidence references does not. Update the declared version when publishing
an intentional change to an authored dataset.

## Published BEIR / SciFact profile

The loader accepts an extracted directory containing `corpus.jsonl`,
`queries.jsonl`, and `qrels/{train,dev,test}.tsv`. It reads every available split,
maps `train` to development, and leaves `test` as test. If both train and dev
define the same query, their labels must agree. Original split labels, including
zero scores, remain in the snapshot's `provenance.qrels`; source-split tags also
identify each imported case. Query IDs are retained. Group IDs come from
`metadata.group_id` when present and otherwise from the query ID. Query language
comes from `metadata.language` and defaults to English for this profile.

Only binary relevance scores, 0 and 1, are supported. Graded, negative and
non-finite scores are rejected. Positive qrels become document evidence units;
the full corpus remains available for retrieval, including unjudged documents.
Document content is exactly `title + "\n" + text`. Corpus IDs are identifiers and
are never interpreted as paths. Queries with only zero labels receive no
retrieval-quality denominator. Imported cases have empty reference answers and
support document-level retrieval evaluation only.

The imported dataset is named `beir`; its version is a SHA-256 of canonical source
records and original split qrels. Relocating or renaming the extracted directory
does not change its identity. Record the source archive URL and file checksums in
the experiment report, alongside the dataset digest. No download or framework
dependency is required by the loader.

The external profile uses **SciFact**, introduced by David Wadden and colleagues
in *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020). The
[original project](https://github.com/allenai/scifact) supplies the data and task
documentation. The [BEIR dataset catalog](https://github.com/beir-cellar/beir)
publishes a retrieval adaptation with train/test splits and 300 test queries.
Use BEIR's published split labels when reporting this profile; these are distinct
from the original claim-verification task's split names and evaluation.

SciFact's claims and evidence annotations use **CC BY 4.0**, and its corpus
abstracts use **ODC-By 1.0**. Its upstream code uses **Apache 2.0**. See the
[SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md).
Those external terms remain attached to the data; importing them does not make
them locally authored or reviewed annotations. Keep the published test evaluation
separate from local diagnostics and do not tune retrieval choices on its results.
