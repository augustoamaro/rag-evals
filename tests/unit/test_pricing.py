from rag.domain.entities import Usage
from rag.services.pricing import cost_usd


def test_opus_cost() -> None:
    # Opus 4.8: $5 / 1M input, $25 / 1M output
    u = Usage(input_tokens=1_000_000, output_tokens=200_000)
    assert cost_usd("claude-opus-4-8", u) == 5.0 + 5.0  # 5 + 0.2 * 25


def test_unknown_model_zero() -> None:
    assert cost_usd("mystery", Usage(1000, 1000)) == 0.0
