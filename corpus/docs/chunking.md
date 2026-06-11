# Chunking Documents for Retrieval

Retrieval systems do not embed whole documents; they split them into smaller chunks
and embed each chunk independently. Chunking matters because a chunk is the unit of
retrieval: if a chunk is too large, its embedding blurs several topics together and
retrieval precision drops; if it is too small, it loses the surrounding context
needed to answer a question.

A common strategy is fixed-size chunking with overlap. The document is divided into
windows of a target length, and consecutive windows overlap by a fixed amount so that
a sentence spanning a boundary is not lost. The overlap means a fact that falls at the
edge of one chunk also appears whole in the next, improving the odds that retrieval
surfaces it.

The target size is usually expressed in tokens or words. Smaller chunks of a hundred
to two hundred words favor precision and pinpoint citations, while larger chunks favor
recall and richer context. The right size depends on the documents and the questions,
which is exactly the kind of decision an evaluation harness exists to settle.

More advanced strategies chunk on semantic or structural boundaries — paragraphs,
headings, or sentence groups — so that each chunk is internally coherent. Whatever the
strategy, chunking should be deterministic so that re-ingesting the same document
yields the same chunks and evaluation results remain reproducible.
