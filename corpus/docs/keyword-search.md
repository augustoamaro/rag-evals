# Keyword Search and BM25

Keyword search, also called lexical search, ranks documents by the query terms they
contain. It is fast, interpretable, and excellent at matching exact terms such as
names, error codes, and rare jargon that an embedding model may not represent well.
Its weakness is the vocabulary mismatch problem: a document that means the same thing
but uses different words will not match.

BM25 is the standard ranking function for keyword search. It scores a document by
summing, over each query term, a weight that rewards term frequency but with
diminishing returns, multiplied by an inverse document frequency factor that
down-weights common terms. BM25 also normalizes for document length so that long
documents are not unfairly favored just for containing more words.

Postgres provides built-in full-text search. Text is converted to a tsvector of
normalized lexemes, and a query is converted to a tsquery; the match operator tests
whether the document satisfies the query. The ts_rank_cd function then ranks matches
by term proximity and frequency. A GIN index on the tsvector column makes these
lookups fast.

Because keyword search and semantic search fail in different ways — lexical search
misses paraphrases, semantic search misses exact rare terms — combining them tends to
outperform either one alone.
