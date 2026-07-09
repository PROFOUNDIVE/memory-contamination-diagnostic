from __future__ import annotations

import hashlib
import re
from math import sqrt
from typing import TypedDict

from memcontam.memory.stores import MemoryEntry


_EMBEDDING_DIMENSION = 16


class RetrievedRecord(TypedDict):
    entry_id: str
    content: str
    score: float
    rank: int
    memory_type: str
    clean_or_contaminated: str
    source_trial_id: str | None
    metadata: dict
    memory_entry: MemoryEntry


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _embed(text: str) -> list[float]:
    vector = [0.0] * _EMBEDDING_DIMENSION
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(_EMBEDDING_DIMENSION):
            byte = digest[index % len(digest)]
            vector[index] += (byte / 255.0) * 2.0 - 1.0
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _record(entry: MemoryEntry, score: float, rank: int) -> RetrievedRecord:
    return {
        "entry_id": entry.entry_id,
        "content": entry.content,
        "score": score,
        "rank": rank,
        "memory_type": entry.memory_type,
        "clean_or_contaminated": entry.clean_or_contaminated,
        "source_trial_id": entry.source_trial_id,
        "metadata": entry.metadata,
        "memory_entry": entry,
    }


def render_retrieved_record(record: RetrievedRecord) -> str:
    provenance = ", ".join(
        [
            f"memory_type={record['memory_type']}",
            f"clean_or_contaminated={record['clean_or_contaminated']}",
            f"source_trial_id={record['source_trial_id']}",
            f"metadata={record['metadata']}",
        ]
    )
    return (
        f"#{record['rank']} entry_id={record['entry_id']} score={record['score']:.6f} "
        f"{provenance}\n{record['content']}"
    )


def retrieve_records(query: str, entries: list[MemoryEntry], k: int = 3) -> list[RetrievedRecord]:
    if k <= 0 or not entries:
        return []

    query_embedding = _embed(query)
    scored = [
        (entry, _cosine(query_embedding, _embed(entry.content)))
        for entry in entries
    ]
    ordered = sorted(scored, key=lambda item: (-item[1], item[0].entry_id))[:k]
    return [_record(entry, score, rank) for rank, (entry, score) in enumerate(ordered, start=1)]


def lexical_retrieve(query: str, entries: list[MemoryEntry], k: int = 3) -> list[tuple[MemoryEntry, float]]:
    # ponytail: compatibility wrapper for older tuple callers; new code should use retrieve_records().
    return [(record["memory_entry"], record["score"]) for record in retrieve_records(query, entries, k=k)]
