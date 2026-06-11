from __future__ import annotations


class LlmError(RuntimeError):
    """An LLM call finished in a state whose content must not be trusted."""


def check_stop_reason(stop_reason: object, context: str) -> None:
    """Reject responses that ended for the wrong reason.

    ``max_tokens`` means the visible output was truncated (with adaptive
    thinking, the budget can be consumed before the answer is complete);
    ``refusal`` means the content is empty or partial. Trusting either would
    silently corrupt eval results.
    """
    if stop_reason in ("max_tokens", "refusal"):
        raise LlmError(f"{context} stopped with stop_reason={stop_reason!r}")
