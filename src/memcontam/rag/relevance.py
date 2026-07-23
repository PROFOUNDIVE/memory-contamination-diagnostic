from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentTriplet:
    false_document_id: str
    correct_document_id: str
    irrelevant_document_id: str


@dataclass(frozen=True)
class RelevanceRelation:
    query_id: str
    relevant_document_ids: frozenset[str]

    def includes(self, document_id: str) -> bool:
        return document_id in self.relevant_document_ids


def recall_at_k(
    retrieved_document_ids: tuple[str, ...], relevant_document_ids: set[str], k: int
) -> float:
    if k <= 0 or not relevant_document_ids:
        return 0.0
    return len(set(retrieved_document_ids[:k]) & relevant_document_ids) / len(relevant_document_ids)
