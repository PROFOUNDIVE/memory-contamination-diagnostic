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
