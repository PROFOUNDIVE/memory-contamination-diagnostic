from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone

from memcontam.baselines.bot_write import BoTTemplatePayload
from memcontam.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider, normalized_dot_top_k
from memcontam.memory.stores import MemoryEntry


@dataclass(frozen=True)
class BotBufferIdentity:
    run_id: str
    task_name: str
    baseline: str
    arm: str
    backbone: str


@dataclass
class ThoughtTemplate:
    entry_id: str
    content: str
    source_trial_id: str
    source_entry_ids: list[str] = field(default_factory=list)
    accepted_at: datetime | None = None
    metadata: dict = field(default_factory=dict)


class BotBufferRegistry:
    def __init__(self):
        self._buffers: dict[BotBufferIdentity, list[ThoughtTemplate]] = {}

    def insert(self, identity: BotBufferIdentity, entry: ThoughtTemplate) -> ThoughtTemplate:
        stored = copy.deepcopy(entry)
        if stored.accepted_at is None:
            stored.accepted_at = datetime.now(timezone.utc)
        self._buffers.setdefault(identity, []).append(stored)
        return stored

    def snapshot(self, identity: BotBufferIdentity) -> tuple[ThoughtTemplate, ...]:
        return tuple(self._buffers.get(identity, []))

    def clone(self, identity: BotBufferIdentity) -> list[ThoughtTemplate]:
        return copy.deepcopy(self._buffers.get(identity, []))


@dataclass(frozen=True)
class NativeNoveltyDecision:
    admitted: bool
    compared_entry_id: str | None
    top_similarity: float | None


def evaluate_native_novelty(
    candidate: BoTTemplatePayload,
    existing: list[MemoryEntry],
    provider: EmbeddingProvider | None = None,
) -> NativeNoveltyDecision:
    if not existing:
        return NativeNoveltyDecision(True, None, None)
    embedding_provider = provider or FakeEmbeddingProvider()
    descriptions = [
        entry.metadata["description"]
        if isinstance(entry.metadata.get("description"), str)
        else entry.content
        for entry in existing
    ]
    entry_id, score = normalized_dot_top_k(
        embedding_provider.encode_query(candidate.description),
        [embedding_provider.encode_document(description) for description in descriptions],
        [entry.entry_id for entry in existing],
        k=1,
    )[0]
    return NativeNoveltyDecision(score < 0.7, entry_id, score)
