from __future__ import annotations

import re
from typing import Any

from rag.adapters.llm.errors import check_stop_reason
from rag.domain.entities import Answer, Citation, RetrievedChunk, Usage

_CITATION = re.compile(r"\[([^\]\s]+)\]")

# No sample id here on purpose: a concrete example such as "[doc:0]" was copied
# into live answers as a citation of a passage that had never been retrieved.
_SYSTEM = (
    "You answer questions strictly from the provided context passages. Each "
    "passage starts with its id in square brackets on a line of its own. Cite "
    "the passages you used inline by copying their ids exactly as given, "
    "brackets included; never invent, shorten or add a prefix to an id. If the "
    "context does not contain the answer, say so plainly. Do not use outside "
    "knowledge."
)


class ClaudeGenerator:
    """RAG answer generation with citations via Claude. One of two LLM adapters."""

    def __init__(self, client: Any, model: str = "claude-opus-4-8") -> None:
        self._client = client
        self._model = model

    def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
        prompt = _build_prompt(question, context)
        response = self._client.messages.create(
            model=self._model,
            # Generous cap: adaptive thinking draws from the same budget, and a
            # too-small value truncates (or empties) the visible answer.
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = Usage(
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )
        check_stop_reason(
            getattr(response, "stop_reason", None), "answer generation", usage=usage
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        citations, invalid = _parse_citations(text, context)
        return Answer(
            text=text, citations=citations, usage=usage, invalid_citations=invalid
        )


def _build_prompt(question: str, context: list[RetrievedChunk]) -> str:
    passages = "\n\n".join(f"[{rc.chunk.id}]\n{rc.chunk.text}" for rc in context)
    return f"Context passages:\n\n{passages}\n\nQuestion: {question}"


def _parse_citations(
    text: str, context: list[RetrievedChunk]
) -> tuple[list[Citation], list[str]]:
    """Split the cited ids into retrieved passages and ids that match none."""
    valid = {rc.chunk.id for rc in context}
    seen: set[str] = set()
    cited: list[Citation] = []
    invalid: list[str] = []
    for match in _CITATION.finditer(text):
        cid = match.group(1)
        if cid in seen:
            continue
        seen.add(cid)
        if cid in valid:
            cited.append(Citation(chunk_id=cid))
        else:
            invalid.append(cid)
    return cited, invalid
