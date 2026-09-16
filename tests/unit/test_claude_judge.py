from __future__ import annotations

import anthropic
import httpx
import pytest

from rag.adapters.llm.claude_judge import AnswerScoresModel, ClaudeJudge
from rag.adapters.llm.fake import FakeAnthropic
from rag.domain.entities import Answer, Chunk, Citation, RetrievedChunk, Usage


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


def test_judge_rejects_truncated_responses() -> None:
    import pytest

    from rag.adapters.llm.errors import LlmError

    parsed = AnswerScoresModel(
        faithfulness=0.5, relevance=0.5, citation_correctness=0.5, rationale="x"
    )
    client = FakeAnthropic(parsed=parsed, stop_reason="max_tokens", usage=(200, 40))
    judge = ClaudeJudge(client)
    with pytest.raises(LlmError, match="max_tokens") as exc:
        judge.score("q", Answer(text="a"), _context(), reference_answer="ref")
    assert exc.value.usage == Usage(input_tokens=200, output_tokens=40)


def test_judge_missing_output_retains_billable_usage() -> None:
    import pytest

    from rag.adapters.llm.errors import LlmError

    client = FakeAnthropic(parsed=None, usage=(200, 40))
    with pytest.raises(LlmError, match="structured output") as exc:
        ClaudeJudge(client).score("q", Answer(text="a"), _context(), "ref")
    assert exc.value.usage == Usage(input_tokens=200, output_tokens=40)


def test_scores_model_rejects_out_of_range() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnswerScoresModel(
            faithfulness=1.5, relevance=0.5, citation_correctness=0.5, rationale="x"
        )


@pytest.mark.parametrize(
    ("stop_reason", "text"),
    [
        ("max_tokens", '{"faithfulness":'),
        ("refusal", "I cannot provide a score."),
        (
            "end_turn",
            '{"faithfulness":2,"relevance":1,"citation_correctness":1,"rationale":"invalid"}',
        ),
    ],
)
def test_real_sdk_parse_failures_preserve_response_usage(
    stop_reason: str, text: str
) -> None:
    from rag.adapters.llm.errors import LlmError

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 200, "output_tokens": 40},
            },
        )

    with (
        anthropic.Anthropic(
            api_key="offline-test",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client,
        pytest.raises(LlmError) as exc,
    ):
        ClaudeJudge(client).score("q", Answer(text="a"), _context(), "ref")
    assert exc.value.usage == Usage(200, 40)
