"""Opt-in smoke tests against the real Anthropic API.

Run with: RAG_ANTHROPIC_API_KEY=... uv run pytest -m live
Never runs in CI; skipped without a key.
"""

from __future__ import annotations

import os

import pytest

from rag.adapters.llm.claude_generator import ClaudeGenerator
from rag.adapters.llm.claude_judge import ClaudeJudge
from rag.domain.entities import Chunk, RetrievedChunk

pytestmark = pytest.mark.live


@pytest.fixture
def client() -> object:
    key = os.environ.get("RAG_ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("RAG_ANTHROPIC_API_KEY not set — live tests are opt-in")
    import anthropic

    return anthropic.Anthropic(api_key=key)


def _context() -> list[RetrievedChunk]:
    chunk = Chunk(
        id="cap:0",
        document_id="cap",
        ordinal=0,
        text=(
            "The CAP theorem states that during a network partition a distributed "
            "system must choose between consistency and availability."
        ),
        token_count=24,
    )
    return [RetrievedChunk(chunk=chunk, score=1.0, rank=1)]


def test_generator_answers_with_citation(client: object) -> None:
    answer = ClaudeGenerator(client).answer(
        "What does the CAP theorem force you to choose between during a partition?",
        _context(),
    )
    assert answer.text
    assert any(c.chunk_id == "cap:0" for c in answer.citations)
    assert answer.usage.input_tokens > 0


def test_judge_scores_a_grounded_answer(client: object) -> None:
    generator = ClaudeGenerator(client)
    answer = generator.answer("What does CAP force you to choose between?", _context())
    scores, usage = ClaudeJudge(client).score(
        "What does CAP force you to choose between?",
        answer,
        _context(),
        reference_answer="Between consistency and availability.",
    )
    assert 0.0 <= scores.faithfulness <= 1.0
    assert usage.output_tokens > 0
