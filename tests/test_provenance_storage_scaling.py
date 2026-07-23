from __future__ import annotations

import json

from memcontam.logging.provenance import normalize_memory_event
from memcontam.logging.schema import MemoryEvent, MemoryItemLog
from memcontam.memory.stores import MemoryEntry


def _chain(length: int) -> tuple[list[MemoryEntry], list[MemoryEvent]]:
    root = MemoryEntry(
        entry_id="root",
        content="Injected root.",
        memory_type="strategy",
        clean_or_contaminated="contaminated",
        metadata={
            "contamination_class": "injected",
            "lineage_status": "exact",
            "lineage_basis": "seed",
            "injected_root_ids": ["root"],
        },
    )
    entries = [root]
    events = []
    for index in range(1, length):
        child = MemoryEntry(
            entry_id=f"derived-{index}",
            content=f"Derived entry {index}.",
            memory_type="verbal_reflection",
            clean_or_contaminated="contaminated",
            source_trial_id=f"trial-{index}",
            metadata={
                "direct_parent_ids": [entries[-1].entry_id],
                "lineage_status": "exact",
                "lineage_basis": "recorded_parent",
                "ancestor_ids": [entry.entry_id for entry in entries],
            },
        )
        event = normalize_memory_event(
            "reflexion_style",
            f"trial-{index}",
            entries,
            [*entries, child],
            {"type": "reflexion_append", "status": "accepted", "new_entry_id": child.entry_id},
        )
        assert event is not None
        entries.append(child)
        events.append(event)
    return entries, events


def _canonical_lineage_bytes(length: int) -> tuple[int, int]:
    entries, events = _chain(length)
    payload = {
        "entries": [
            MemoryItemLog.from_memory_entry(entry, entries).model_dump() for entry in entries
        ],
        "edges": [edge.model_dump() for event in events for edge in event.lineage_edges],
    }
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":"))), len(payload["edges"])


def test_direct_edge_storage_has_bounded_near_linear_growth() -> None:
    bytes_at_n, edges_at_n = _canonical_lineage_bytes(32)
    bytes_at_2n, edges_at_2n = _canonical_lineage_bytes(64)

    assert edges_at_n == 31
    assert edges_at_2n == 63
    assert bytes_at_2n < bytes_at_n * 3


def test_exact_derived_chain_preserves_only_its_direct_parent_and_root() -> None:
    entries, events = _chain(3)
    derived = MemoryItemLog.from_memory_entry(entries[-1], entries)

    assert derived.contamination_class == "derived"
    assert derived.direct_parent_ids == ["derived-1"]
    assert derived.injected_root_ids == ["root"]
    assert events[-1].lineage_edges[0].child_entry_id == "derived-2"
    assert events[-1].lineage_edges[0].parent_entry_id == "derived-1"
    assert events[-1].lineage_edges[0].injected_root_ids == ["root"]


def test_accepted_update_reconciles_its_direct_parent_edge() -> None:
    root = MemoryEntry(
        entry_id="root",
        content="Injected root.",
        memory_type="strategy",
        clean_or_contaminated="contaminated",
        metadata={
            "contamination_class": "injected",
            "lineage_status": "exact",
            "lineage_basis": "seed",
            "injected_root_ids": ["root"],
        },
    )
    old = MemoryEntry(entry_id="rewrite", content="old", memory_type="dynamic_cheatsheet")
    updated = MemoryEntry(
        entry_id="rewrite",
        content="new",
        memory_type="dynamic_cheatsheet",
        clean_or_contaminated="contaminated",
        metadata={"direct_parent_ids": ["root"], "lineage_status": "exact"},
    )

    event = normalize_memory_event(
        "dynamic_cheatsheet_optional",
        "trial-rewrite",
        [root, old],
        [root, updated],
        {"type": "dynamic_cheatsheet_update", "status": "accepted"},
    )

    assert event is not None
    assert event.new_entry_ids == []
    assert event.updated_entry_ids == ["rewrite"]
    assert [(edge.child_entry_id, edge.parent_entry_id) for edge in event.lineage_edges] == [
        ("rewrite", "root")
    ]
