# Vector Search and Approximate Nearest Neighbors

Vector search finds the stored vectors closest to a query vector under a distance
metric such as cosine or Euclidean distance. A brute-force scan compares the query
against every stored vector, which is exact but grows linearly with the size of the
collection and becomes too slow for millions of vectors.

Approximate nearest neighbor, or ANN, search trades a small amount of accuracy for a
large gain in speed. Instead of scanning everything, an ANN index narrows the search
to a promising subset of candidates. The most common index for high-recall search is
HNSW, Hierarchical Navigable Small World graphs, which connect vectors into a layered
graph that can be traversed in logarithmic time.

In Postgres, the pgvector extension adds a vector column type and supports both
exact search and HNSW indexes. A cosine-distance query is written with the distance
operator, ordering rows by how close their embedding is to the query vector and
limiting to the top results.

Recall is the key quality measure for an ANN index: it is the fraction of the true
nearest neighbors that the index actually returns. Index parameters trade recall
against build time and query latency, so tuning an ANN index means choosing an
acceptable point on the recall-versus-speed curve for the workload.
