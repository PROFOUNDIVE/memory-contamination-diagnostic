from __future__ import annotations

from memcontam.memory.stores import MemoryEntry


def lexical_retrieve(query: str, entries: list[MemoryEntry], k: int = 3) -> list[tuple[MemoryEntry, float]]:
    query_terms = set(query.lower().split())
    scored = []
    for entry in entries:
        entry_terms = set(entry.content.lower().split())
        score = len(query_terms & entry_terms) / max(len(query_terms | entry_terms), 1)
        scored.append((entry, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
