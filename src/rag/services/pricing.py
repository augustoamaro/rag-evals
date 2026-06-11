from __future__ import annotations

from rag.domain.entities import Usage

# USD per 1M tokens (input, output). Source: Anthropic pricing — Claude Opus 4.8.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
}


def cost_usd(model: str, usage: Usage) -> float:
    if model not in _PRICES:
        return 0.0
    price_in, price_out = _PRICES[model]
    return usage.input_tokens / 1e6 * price_in + usage.output_tokens / 1e6 * price_out
