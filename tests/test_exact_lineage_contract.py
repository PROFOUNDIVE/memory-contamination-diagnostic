from __future__ import annotations

import importlib

from memcontam.memory.stores import MemoryEntry


def test_exact_lineage_never_promotes_visible_context_or_sources_to_parents() -> None:
    provenance = importlib.import_module("memcontam.logging.provenance")
    event = provenance.normalize_memory_event(
        baseline="bot_style",
        source_trial_id="trial-1",
        memory_before=[],
        memory_after=[
            MemoryEntry(
                entry_id="derived-1",
                content="derived",
                memory_type="thought_template",
                clean_or_contaminated="clean",
                source_trial_id="trial-1",
                metadata={"source_entry_ids": ["visible-source"]},
            )
        ],
        memory_write_event={"type": "insert", "source_entry_ids": ["visible-source"]},
    )

    assert event is not None
    assert event.parent_entry_ids == []
    assert event.source_entry_ids == ["visible-source"]


def test_bot_write_uses_only_explicitly_used_visible_ids_as_parents_and_supports() -> None:
    bot_write = importlib.import_module("memcontam.baselines.bot_write")
    payload = bot_write.BoTTemplatePayload(
        description="A reusable arithmetic procedure.",
        template="Build factor pairs before combining them.",
        category="procedure-based",
        explicitly_used_memory_ids=("used-template",),
    )

    entry = bot_write.build_template_entry(
        payload=payload,
        source_trial_id="trial-1",
        visible_entry_ids=["visible-but-unused", "used-template"],
    )

    assert entry.metadata["declared_updater_context_ids"] == [
        "visible-but-unused",
        "used-template",
    ]
    assert entry.metadata["direct_parent_ids"] == ["used-template"]
    assert entry.metadata["memory_support_ids"] == ["used-template"]
    assert entry.metadata["source_entry_ids"] == ["used-template"]
