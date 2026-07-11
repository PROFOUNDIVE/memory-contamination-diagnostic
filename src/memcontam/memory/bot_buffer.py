from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone


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
