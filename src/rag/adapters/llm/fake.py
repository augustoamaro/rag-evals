"""An in-memory stand-in for the Anthropic client.

Lets the Claude generator and judge be exercised in tests and offline demos
without an API key, by mimicking the small slice of the SDK they consume:
``client.messages.create(...)`` and ``client.messages.parse(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _CreateResponse:
    content: list[_Block]
    usage: _Usage
    stop_reason: str = "end_turn"


@dataclass
class _ParseResponse:
    parsed_output: Any
    usage: _Usage
    stop_reason: str = "end_turn"


@dataclass
class _FakeMessages:
    text: str = ""
    usage: tuple[int, int] = (0, 0)
    parsed: Any = None
    stop_reason: str = "end_turn"
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    parse_calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> _CreateResponse:
        self.create_calls.append(kwargs)
        return _CreateResponse(
            [_Block("text", self.text)], _Usage(*self.usage), self.stop_reason
        )

    def parse(self, **kwargs: Any) -> _ParseResponse:
        self.parse_calls.append(kwargs)
        return _ParseResponse(self.parsed, _Usage(*self.usage), self.stop_reason)


class FakeAnthropic:
    def __init__(
        self,
        *,
        text: str = "",
        usage: tuple[int, int] = (0, 0),
        parsed: Any = None,
        stop_reason: str = "end_turn",
    ) -> None:
        self.messages = _FakeMessages(
            text=text, usage=usage, parsed=parsed, stop_reason=stop_reason
        )
