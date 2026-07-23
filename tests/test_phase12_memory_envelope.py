from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util

import pytest


def _cards():
    assert importlib.util.find_spec("memcontam.memory.cards_v3") is not None
    return importlib.import_module("memcontam.memory.cards_v3")


def _registry():
    assert importlib.util.find_spec("memcontam.memory.writer_registry") is not None
    return importlib.import_module("memcontam.memory.writer_registry")


def _envelope(cards, entry_id: str, order_key: int, **overrides):
    payload = {
        "entry_id": entry_id,
        "baseline": "full_history",
        "semantic_kind": "full_history_transcript",
        "schema_version": "memory_card_v3",
        "writer_id": "fh_appender",
        "writer_event_id": f"event-{entry_id}",
        "writer_stage": "full_history_generate",
        "created_trial_id": f"trial-{entry_id}",
        "source_trial_ids": (f"trial-{entry_id}",),
        "source_outcome": None,
        "trial_support_ids": (f"trial-{entry_id}",),
        "memory_support_ids": (),
        "direct_parent_ids": (),
        "version_predecessor_id": None,
        "order_key": order_key,
        "native_component": "history",
        "content": f"content for {entry_id}",
    }
    payload.update(overrides)
    payload["content_hash"] = cards.canonical_content_hash(payload["content"])
    return cards.MemoryCardEnvelopeV3(**payload)


def _assert_code(cards, registry, envelope, code: str, prior_entries=()) -> None:
    with pytest.raises(cards.MemoryEnvelopeError) as error:
        cards.validate_memory_envelope(envelope, registry, prior_entries)
    assert error.value.code == code


def test_validates_all_registered_native_envelopes() -> None:
    cards = _cards()
    registry = _registry().WriterRegistry.native()
    envelopes = (
        _envelope(cards, "fh-1", 1),
        _envelope(
            cards,
            "bot-1",
            2,
            baseline="bot_style",
            semantic_kind="thought_template",
            writer_id="bot_buffer_manager",
            writer_stage="bot_thought_distill",
            native_component="buffer",
        ),
        _envelope(
            cards,
            "reflection-1",
            3,
            baseline="reflexion_style",
            semantic_kind="verbal_reflection",
            writer_id="reflexion_reflector",
            writer_stage="reflexion_reflect",
            native_component="reflections",
        ),
        _envelope(
            cards,
            "rag-1",
            4,
            baseline="retrieval_rag",
            semantic_kind="rag_document",
            writer_id="rag_corpus_loader",
            writer_stage="rag_corpus_load",
            created_trial_id=None,
            source_trial_ids=(),
            trial_support_ids=(),
            native_component="corpus",
        ),
        _envelope(
            cards,
            "dc-archive-1",
            5,
            baseline="dynamic_cheatsheet_rs_optional",
            semantic_kind="dc_rs_io_pair",
            writer_id="dc_archive_writer",
            writer_stage="dc_rs_generate",
            native_component="archive",
        ),
        _envelope(
            cards,
            "dc-strategy-1",
            6,
            baseline="dynamic_cheatsheet_rs_optional",
            semantic_kind="dynamic_cheatsheet",
            writer_id="dc_strategy_writer",
            writer_stage="dc_rs_synthesize",
            native_component="strategy",
        ),
    )

    prior_entries = ()
    for envelope in envelopes:
        validated = cards.validate_memory_envelope(envelope, registry, prior_entries)
        assert validated.envelope == envelope
        assert envelope.content_hash == cards.canonical_content_hash(envelope.content)
        prior_entries = (*prior_entries, envelope)


def test_rejects_invalid_writer_support_and_hash_cases() -> None:
    cards = _cards()
    registry = _registry().WriterRegistry.native()
    valid = _envelope(cards, "entry-1", 1)

    _assert_code(
        cards,
        registry,
        replace(valid, writer_stage="bot_thought_distill"),
        "UNREGISTERED_WRITER_EVENT",
    )
    _assert_code(
        cards,
        registry,
        replace(valid, source_trial_ids=(), trial_support_ids=()),
        "MISSING_SOURCE_TRIAL",
    )
    _assert_code(
        cards,
        registry,
        replace(valid, memory_support_ids=("parent-1",)),
        "SUPPORT_OUTSIDE_PARENTS",
    )
    _assert_code(cards, registry, replace(valid, content_hash="0" * 64), "CONTENT_HASH_MISMATCH")


def test_rejects_conflated_predecessors_future_references_and_cycles() -> None:
    cards = _cards()
    registry = _registry().WriterRegistry.native()
    predecessor = _envelope(cards, "predecessor", 1)
    conflated = _envelope(
        cards,
        "conflated",
        2,
        direct_parent_ids=("predecessor",),
        version_predecessor_id="predecessor",
    )
    future = _envelope(
        cards,
        "future",
        1,
        direct_parent_ids=("predecessor",),
        memory_support_ids=("predecessor",),
    )
    first = _envelope(cards, "first", 1, direct_parent_ids=("second",))
    second = _envelope(cards, "second", 2, direct_parent_ids=("first",))

    _assert_code(cards, registry, conflated, "VERSION_PREDECESSOR_CONFLATED", (predecessor,))
    _assert_code(cards, registry, future, "FUTURE_REFERENCE", (predecessor,))
    _assert_code(cards, registry, first, "CYCLE", (second,))
