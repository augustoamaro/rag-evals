from __future__ import annotations

from rag.adapters.llm.claude_generator import ClaudeGenerator
from rag.adapters.llm.fake import FakeAnthropic
from rag.domain.entities import Answer, Chunk, RetrievedChunk, Strategy
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
