from __future__ import annotations

from rag.domain.entities import Usage


class ModelCallError(RuntimeError):
    """An unusable model result, optionally carrying known provider usage."""

    def __init__(self, message: str, *, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class EvalPreflightError(ValueError):
    """Evaluation inputs or the indexed corpus are inconsistent."""
