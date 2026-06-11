from __future__ import annotations

import re
from typing import Any

from rag.adapters.llm.errors import check_stop_reason
from rag.domain.entities import Answer, Citation, RetrievedChunk, Usage

_CITATION = re.compile(r"\[([^\]\s]+)\]")

_SYSTEM = (
    "You answer questions strictly from the provided context passages. "
    "Each passage is labelled with an id like [doc:0]. Cite the ids of the "
    "passages you used inline, e.g. [doc:0]. If the context does not contain "
    "the answer, say so plainly. Do not use outside knowledge."
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
        check_stop_reason(getattr(response, "stop_reason", None), "answer generation")
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = Usage(
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )
        return Answer(text=text, citations=_parse_citations(text, context), usage=usage)


def _build_prompt(question: str, context: list[RetrievedChunk]) -> str:
    passages = "\n\n".join(f"[{rc.chunk.id}]\n{rc.chunk.text}" for rc in context)
    return f"Context passages:\n\n{passages}\n\nQuestion: {question}"


def _parse_citations(text: str, context: list[RetrievedChunk]) -> list[Citation]:
    valid = {rc.chunk.id for rc in context}
    seen: set[str] = set()
    cited: list[Citation] = []
    for match in _CITATION.finditer(text):
        cid = match.group(1)
        if cid in valid and cid not in seen:
            seen.add(cid)
            cited.append(Citation(chunk_id=cid))
    return cited
