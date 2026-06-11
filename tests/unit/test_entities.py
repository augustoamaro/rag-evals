from dataclasses import FrozenInstanceError

import pytest

from rag.domain.entities import Chunk, Strategy


def test_chunk_is_frozen_and_value_equal() -> None:
    a = Chunk(id="c1", document_id="d1", ordinal=0, text="hi", token_count=1)
    b = Chunk(id="c1", document_id="d1", ordinal=0, text="hi", token_count=1)
    assert a == b
    with pytest.raises(FrozenInstanceError):
        a.text = "x"  # type: ignore[misc]


def test_strategy_values() -> None:
    assert Strategy.HYBRID.value == "hybrid"
    assert Strategy("dense") is Strategy.DENSE
