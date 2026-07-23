from __future__ import annotations

import importlib
import importlib.util


def _cards():
    assert importlib.util.find_spec("memcontam.memory.cards_v3") is not None
    return importlib.import_module("memcontam.memory.cards_v3")


def _registry():
    assert importlib.util.find_spec("memcontam.memory.writer_registry") is not None
    return importlib.import_module("memcontam.memory.writer_registry")


def _envelope(cards, *, writer_stage: str):
    content = "A reusable template."
    return cards.MemoryCardEnvelopeV3(
        entry_id="bot-template-1",
        baseline="bot_style",
        semantic_kind="thought_template",
        schema_version="memory_card_v3",
        writer_id="bot_buffer_manager",
        writer_event_id="event-bot-template-1",
        writer_stage=writer_stage,
        created_trial_id="trial-bot-1",
        source_trial_ids=("trial-bot-1",),
        source_outcome=True,
        trial_support_ids=("trial-bot-1",),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=1,
        native_component="buffer",
        content=content,
        content_hash=cards.canonical_content_hash(content),
    )


def test_writer_permissions_bind_baseline_kind_writer_and_stage() -> None:
    cards = _cards()
    registry = _registry().WriterRegistry.native()

    assert registry.permits(_envelope(cards, writer_stage="bot_thought_distill"))
    assert not registry.permits(_envelope(cards, writer_stage="bot_instantiate_solve"))
