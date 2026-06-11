from __future__ import annotations

from rag.adapters.llm.claude_judge import AnswerScoresModel, ClaudeJudge
from rag.adapters.llm.fake import FakeAnthropic
from rag.domain.entities import Answer, Chunk, Citation, RetrievedChunk


def _context() -> list[RetrievedChunk]:
    chunk = Chunk(
        id="rag:0", document_id="rag", ordinal=0, text="RAG grounds...", token_count=2
    )
    return [RetrievedChunk(chunk=chunk, score=1.0, rank=1)]


def test_judge_maps_structured_output_and_usage() -> None:
    parsed = AnswerScoresModel(
        faithfulness=0.95, relevance=0.9, citation_correctness=0.8, rationale="solid"
    )
    client = FakeAnthropic(parsed=parsed, usage=(200, 40))
    judge = ClaudeJudge(client, model="claude-opus-4-8")

    answer = Answer(text="grounded", citations=[Citation("rag:0")])
    scores, usage = judge.score("q", answer, _context(), reference_answer="ref")

    assert scores.faithfulness == 0.95
    assert scores.relevance == 0.9
    assert scores.citation_correctness == 0.8
    assert scores.rationale == "solid"
    assert usage.input_tokens == 200
    # the judge sent the answer, the reference, and the context to the model
    sent = client.messages.parse_calls[0]
    assert sent["output_format"] is AnswerScoresModel
    assert "grounded" in sent["messages"][0]["content"]
    assert "ref" in sent["messages"][0]["content"]


def test_scores_model_rejects_out_of_range() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnswerScoresModel(
            faithfulness=1.5, relevance=0.5, citation_correctness=0.5, rationale="x"
        )
