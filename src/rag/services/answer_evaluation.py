from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

from rag.domain.entities import (
    Answer,
    AnswerScores,
    EvalCase,
    RetrievedChunk,
    StageResult,
    Usage,
)
from rag.domain.errors import ModelCallError
from rag.domain.ports import Generator, Judge
from rag.services.pricing import cost_usd

logger = logging.getLogger(__name__)


def evaluate_answer(
    case: EvalCase,
    context: list[RetrievedChunk],
    generator: Generator,
    judge: Judge,
    model: str,
    *,
    is_cancelled: Callable[[], bool] | None = None,
    on_generation: Callable[[Answer | None, StageResult], None] | None = None,
) -> tuple[Answer | None, AnswerScores | None, StageResult, StageResult]:
    def generate() -> tuple[Answer, Usage]:
        answer = generator.answer(case.question, context)
        return answer, answer.usage

    if is_cancelled is not None and is_cancelled():
        return None, None, StageResult(), StageResult()
    answer, generation = _measure_stage(generate, model)
    if on_generation is not None:
        on_generation(answer, generation)
    if answer is None or (is_cancelled is not None and is_cancelled()):
        return answer, None, generation, StageResult()
    scores, judging = _measure_stage(
        lambda: judge.score(case.question, answer, context, case.reference_answer), model
    )
    return answer, scores, generation, judging


def _measure_stage[T](
    call: Callable[[], tuple[T, Usage]], model: str
) -> tuple[T | None, StageResult]:
    started = time.perf_counter()
    result: T | None = None
    usage: Usage | None = None
    error_type: str | None = None
    try:
        result, usage = call()
    except Exception as exc:
        if isinstance(exc, ModelCallError):
            usage = exc.usage
        error_type = type(exc).__name__
        # Provider messages may contain prompts, keys or other private data.
        logger.warning("model evaluation stage failed (%s)", error_type)
    return result, StageResult(
        status="failed" if error_type else "succeeded",
        usage=usage,
        cost_usd=cost_usd(model, usage) if usage is not None else None,
        latency_ms=max(1, math.ceil((time.perf_counter() - started) * 1000)),
        error_type=error_type,
    )
