from __future__ import annotations

import importlib
import importlib.util

import pytest


def test_memory_card_envelope_requires_supports_to_be_declared_parents() -> None:
    assert importlib.util.find_spec("memcontam.memory.cards"), (
        "BASELINE-FIDELITY-V1 requires typed memory cards"
    )
    cards = importlib.import_module("memcontam.memory.cards")

    assert getattr(cards, "MemoryCard", None) is not None
    assert getattr(cards, "MemoryCardEnvelope", None) is not None

    with pytest.raises(ValueError, match="declared parent"):
        cards.MemoryCardEnvelope(
            entry_id="card-1",
            semantic_kind="reflection",
            writer_id="writer",
            writer_event_id="event",
            trial_log_support_ids=("trial-1",),
            memory_support_ids=("card-0",),
            declared_parent_ids=(),
            source_trial_id="trial-1",
            source_outcome=False,
            order_key=1,
        )

    envelope = cards.MemoryCardEnvelope(
        entry_id="card-1",
        semantic_kind="reflection",
        writer_id="writer",
        writer_event_id="event",
        trial_log_support_ids=("trial-1",),
        memory_support_ids=("card-0",),
        declared_parent_ids=("card-0", "other-parent"),
        source_trial_id="trial-1",
        source_outcome=False,
        order_key=1,
    )
    assert envelope.memory_support_ids == ("card-0",)
