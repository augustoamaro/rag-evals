# LLM-as-a-Judge Evaluation

Answer quality is hard to measure with string matching because two correct answers
can be worded completely differently. LLM-as-a-judge evaluation uses a strong language
model to score an answer against a rubric, approximating human judgment at a fraction
of the cost and turnaround time.

For a RAG system, three rubric dimensions are especially useful. Faithfulness measures
whether every claim in the answer is supported by the retrieved context — an
unsupported claim is a hallucination even if it happens to be true. Answer relevance
measures whether the answer actually addresses the question that was asked. Citation
correctness measures whether the cited chunks genuinely support the statements that
cite them.

To make scores usable, the judge is asked to return structured output: a number in
a fixed range for each dimension plus a short rationale. Structured output turns a
free-form opinion into machine-readable data that can be aggregated across an
evaluation run and tracked over time.

LLM judges have known limitations. They can be inconsistent between runs, and they may
prefer longer or more confident answers regardless of correctness, a bias worth
controlling for. Because of this, the judge track complements rather than replaces the
retrieval metrics, which are deterministic and need no model at all.
