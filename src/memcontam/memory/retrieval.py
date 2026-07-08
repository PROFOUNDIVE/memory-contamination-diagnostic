from __future__ import annotations

import re

from memcontam.memory.stores import MemoryEntry


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def lexical_retrieve(query: str, entries: list[MemoryEntry], k: int = 3) -> list[tuple[MemoryEntry, float]]:
    query_terms = _terms(query)
    scored = []
    for entry in entries:
        entry_terms = _terms(entry.content)
        score = len(query_terms & entry_terms) / max(len(query_terms | entry_terms), 1)
        scored.append((entry, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
