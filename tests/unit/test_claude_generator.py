from __future__ import annotations

from rag.adapters.llm.claude_generator import ClaudeGenerator
from rag.adapters.llm.fake import FakeAnthropic
from rag.domain.entities import Answer, Chunk, RetrievedChunk, Strategy, Usage
from rag.services.query_service import QueryService


def _context() -> list[RetrievedChunk]:
    chunks = [
        Chunk(
            id="cap-theorem:0",
            document_id="cap",
            ordinal=0,
            text="CAP says...",
            token_count=2,
        ),
        Chunk(
            id="rag:0", document_id="rag", ordinal=0, text="RAG grounds...", token_count=2
        ),
    ]
    return [RetrievedChunk(chunk=c, score=1.0, rank=i + 1) for i, c in enumerate(chunks)]


def test_generator_parses_citations_and_usage() -> None:
    client = FakeAnthropic(
        text="During a partition you choose [cap-theorem:0]. Unknown [nope] ignored.",
        usage=(120, 30),
    )
    gen = ClaudeGenerator(client, model="claude-opus-4-8")
    answer = gen.answer("What does CAP say?", _context())

    assert "During a partition" in answer.text
    assert [c.chunk_id for c in answer.citations] == [
        "cap-theorem:0"
    ]  # invalid id dropped
    assert answer.usage.input_tokens == 120
    assert answer.usage.output_tokens == 30
    # the prompt embeds each passage id and the question
    sent = client.messages.create_calls[0]
    assert "cap-theorem:0" in sent["messages"][0]["content"]
    assert "What does CAP say?" in sent["messages"][0]["content"]


def test_system_prompt_has_no_example_id_a_model_could_copy() -> None:
    # A concrete sample such as "[doc:0]" leaked into live answers as a citation
    # of a passage that was never retrieved.
    from rag.metrics.citations import cited_ids

    client = FakeAnthropic(text="ok [rag:0]", usage=(1, 1))
    ClaudeGenerator(client).answer("q", _context())
    system = client.messages.create_calls[0]["system"]
    assert cited_ids(system) == []


def test_generator_reports_cited_ids_that_were_not_retrieved() -> None:
    client = FakeAnthropic(
        text="CAP [cap-theorem:0] and more [doc:0][doc:cap-theorem:0], again [doc:0].",
        usage=(1, 1),
    )
    answer = ClaudeGenerator(client).answer("q", _context())
    assert [c.chunk_id for c in answer.citations] == ["cap-theorem:0"]
    assert answer.invalid_citations == ["doc:0", "doc:cap-theorem:0"]


def test_generator_reports_no_invalid_citations_for_a_clean_answer() -> None:
    client = FakeAnthropic(text="CAP [cap-theorem:0], RAG [rag:0].", usage=(1, 1))
    assert ClaudeGenerator(client).answer("q", _context()).invalid_citations == []


def test_generator_rejects_truncated_or_refused_responses() -> None:
    import pytest

    from rag.adapters.llm.errors import LlmError

    for stop in ("max_tokens", "refusal"):
        client = FakeAnthropic(text="partial...", usage=(10, 5), stop_reason=stop)
        gen = ClaudeGenerator(client)
        with pytest.raises(LlmError, match=stop) as exc:
            gen.answer("q", _context())
        assert exc.value.usage == Usage(input_tokens=10, output_tokens=5)


class StubGenerator:
    def answer(self, question: str, context: list[RetrievedChunk]) -> Answer:
        return Answer(text=f"answer to {question}", citations=[])


class StubRetriever:
    def retrieve(self, query: str, k: int, strategy: Strategy) -> list[RetrievedChunk]:
        return _context()


def test_query_service_retrieves_then_generates() -> None:
    service = QueryService(StubRetriever(), StubGenerator())
    answer, context = service.answer("hello", 2, Strategy.HYBRID)
    assert answer.text == "answer to hello"
    assert len(context) == 2
