# Retrieval Evaluation Metrics

Retrieval quality is measured against a labeled set of queries for which the relevant
documents are known. The metrics below all operate on the ranked list of results a
retriever returns for each query.

Recall at k is the fraction of all relevant documents that appear in the top k
results. It answers "of everything I should have found, how much did I find near the
top." Precision at k is the fraction of the top k results that are actually relevant.
Recall rewards finding relevant items; precision penalizes returning irrelevant ones.

Mean reciprocal rank, or MRR, focuses on the position of the first relevant result. For
each query it takes the reciprocal of the rank of the first relevant hit — one if it is
first, one half if it is second — and averages this over all queries. MRR is ideal when
the user mostly cares about the single best answer being near the top.

Normalized discounted cumulative gain, nDCG, accounts for the entire ranking with a
position discount: a relevant result contributes less the lower it appears, weighted by
one over the logarithm of its rank. The raw discounted gain is divided by the ideal
gain — the score of the best possible ordering — so that nDCG always falls between zero
and one, making it comparable across queries with different numbers of relevant
documents.
