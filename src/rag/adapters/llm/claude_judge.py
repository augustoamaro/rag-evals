from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag.adapters.llm.errors import check_stop_reason
from rag.domain.entities import Answer, AnswerScores, RetrievedChunk, Usage

_SYSTEM = (
    "You are a strict evaluator of retrieval-augmented answers. Score the answer "
    "against the provided context and reference. Faithfulness: are all claims "
    "supported by the context? Relevance: does the answer address the question? "
    "Citation correctness: do the cited passages support the cited claims? Each "
    "score is a number from 0 to 1."
)


class AnswerScoresModel(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    citation_correctness: float = Field(ge=0.0, le=1.0)
    rationale: str


class ClaudeJudge:
    """LLM-as-judge scoring via Claude structured output. The second LLM adapter."""

    def __init__(self, client: Any, model: str = "claude-opus-4-8") -> None:
        self._client = client
        self._model = model

    def score(
        self,
        question: str,
        answer: Answer,
        context: list[RetrievedChunk],
        reference_answer: str,
    ) -> tuple[AnswerScores, Usage]:
        prompt = _build_prompt(question, answer, context, reference_answer)
        response = self._client.messages.parse(
            model=self._model,
            # Generous cap: adaptive thinking draws from the same budget, and a
            # truncated structured response fails to parse mid-run.
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            output_format=AnswerScoresModel,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        check_stop_reason(getattr(response, "stop_reason", None), "judge scoring")
        parsed = response.parsed_output
        scores = AnswerScores(
            faithfulness=float(parsed.faithfulness),
            relevance=float(parsed.relevance),
            citation_correctness=float(parsed.citation_correctness),
            rationale=str(parsed.rationale),
        )
        usage = Usage(
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )
        return scores, usage


def _build_prompt(
    question: str, answer: Answer, context: list[RetrievedChunk], reference_answer: str
) -> str:
    passages = "\n\n".join(f"[{rc.chunk.id}]\n{rc.chunk.text}" for rc in context)
    cited = ", ".join(c.chunk_id for c in answer.citations) or "(none)"
    return (
        f"Question: {question}\n\n"
        f"Context passages:\n\n{passages}\n\n"
        f"Answer to evaluate:\n{answer.text}\n\n"
        f"Citations in the answer: {cited}\n\n"
        f"Reference answer (for relevance only): {reference_answer}"
    )
