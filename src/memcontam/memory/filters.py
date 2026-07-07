from __future__ import annotations

from memcontam.memory.stores import MemoryEntry


def drop_known_contaminated(entries: list[MemoryEntry]) -> tuple[list[MemoryEntry], dict]:
    kept = [entry for entry in entries if entry.clean_or_contaminated != "contaminated"]
    return kept, {"filter": "drop_known_contaminated", "dropped": len(entries) - len(kept)}
