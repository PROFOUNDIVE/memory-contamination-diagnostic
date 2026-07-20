from __future__ import annotations

import importlib
import importlib.util

import pytest


def test_memory_card_envelope_requires_declared_support_to_be_a_parent() -> None:
    assert importlib.util.find_spec("memcontam.memory.cards"), (
        "BASELINE-FIDELITY-V1 requires typed memory cards"
    )
    cards = importlib.import_module("memcontam.memory.cards")

    assert getattr(cards, "MemoryCard", None) is not None
    assert getattr(cards, "MemoryCardEnvelope", None) is not None

    card = cards.MemoryCard(card_id="card-1", content="memory", card_type="reflection")
    with pytest.raises(ValueError, match="declared support"):
        cards.MemoryCardEnvelope(card=card, parent_card_ids=("card-0",))

    envelope = cards.MemoryCardEnvelope(
        card=card,
        parent_card_ids=("card-0",),
        declared_support_ids=("card-0",),
    )
    assert envelope.parent_card_ids == ("card-0",)
