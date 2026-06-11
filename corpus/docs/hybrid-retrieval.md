# Hybrid Retrieval and Reciprocal Rank Fusion

Hybrid retrieval combines a dense semantic retriever with a sparse keyword retriever
so the strengths of each cover the other's blind spots. The dense arm finds passages
that share meaning with the query, and the sparse arm finds passages that share exact
terms, and a fusion step merges their two ranked lists into one.

Reciprocal Rank Fusion, or RRF, is a simple and robust way to merge ranked lists. For
each document it sums one divided by a constant plus the rank, across every list the
document appears in. The constant, often set to sixty, dampens the influence of very
high ranks so that broad agreement across lists matters more than a single top
placement. Documents that rank highly in multiple lists rise to the top of the fused
ranking.

The great advantage of RRF is that it needs no score calibration: it uses only the
rank positions, not the raw similarity scores, so it does not matter that cosine
distances and BM25 scores live on completely different scales. This makes hybrid
retrieval easy to assemble from heterogeneous retrievers.

An optional reranking stage can follow fusion. A cross-encoder reranker reads the
query and each candidate passage together and scores their relevance directly, which
is more accurate than comparing independent embeddings but too expensive to run over
the whole collection, so it is applied only to the fused shortlist.
