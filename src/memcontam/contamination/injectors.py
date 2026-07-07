from __future__ import annotations

from memcontam.memory.stores import MemoryEntry, MemoryState


def inject_entry(memory: MemoryState, catalog_entry: dict) -> MemoryState:
    memory.entries.append(
        MemoryEntry(
            entry_id=catalog_entry["entry_id"],
            content=catalog_entry["content"],
            memory_type=catalog_entry["type"],
            clean_or_contaminated="contaminated",
            metadata=catalog_entry,
        )
    )
    return memory
