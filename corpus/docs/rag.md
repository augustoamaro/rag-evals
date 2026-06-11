# Retrieval-Augmented Generation

Retrieval-augmented generation, or RAG, grounds a language model's answer in
documents retrieved at query time instead of relying only on the model's parametric
memory. The pipeline has two stages: a retriever fetches the passages most relevant
to the question, and a generator conditions on those passages to produce the answer.

RAG addresses two weaknesses of using a language model alone. It reduces
hallucination by giving the model concrete source text to draw from, and it lets the
system answer questions about private or up-to-date information the model was never
trained on, simply by adding documents to the index rather than retraining the model.

Good RAG answers include citations that point back to the specific chunks the answer
relied on. Citations let a reader verify each claim against its source and make it
possible to measure whether the answer is actually grounded in the retrieved context
rather than invented.

The quality of a RAG system is bounded by its retrieval: the generator cannot cite
what the retriever failed to find. This is why retrieval quality is measured
separately from answer quality. If a relevant passage is never retrieved, no amount of
prompting will let the model ground its answer in it, so improving retrieval is
usually the highest-leverage way to improve a RAG system end to end.
