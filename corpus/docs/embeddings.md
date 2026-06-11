# Text Embeddings

A text embedding is a dense vector of floating-point numbers that represents the
meaning of a piece of text in a high-dimensional space. Texts with similar meaning
map to vectors that are close together, so semantic similarity becomes geometric
proximity that a computer can measure directly.

Embeddings are produced by neural models trained so that related inputs land near
each other. Modern sentence-embedding models output vectors with a few hundred
dimensions; the bge-small model, for example, produces 384-dimensional vectors.
Because the dimensionality is fixed, every text — short or long — becomes a vector
of the same length.

Similarity between two embeddings is usually measured with cosine similarity, which
compares the angle between the vectors and ignores their magnitude. Cosine
similarity ranges from -1 to 1, where 1 means the vectors point in the same
direction and are maximally similar. Many systems store normalized vectors so that
cosine similarity reduces to a simple dot product.

Embeddings power semantic search: a query is embedded with the same model as the
documents, and the nearest vectors are returned as the most relevant results. Unlike
keyword matching, embedding search can retrieve passages that share no words with
the query but express the same idea, which is its key advantage over lexical search.
