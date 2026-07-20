from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memcontam.baselines.contracts import NonEmptyStr
from memcontam.memory.provenance import require_declared_parent_support


@dataclass(frozen=True)
class MemoryCard:
    card_id: NonEmptyStr
    content: NonEmptyStr
    card_type: NonEmptyStr
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryCardEnvelope:
    entry_id: str
    semantic_kind: str
    writer_id: str
    writer_event_id: str
    trial_log_support_ids: tuple[str, ...]
    memory_support_ids: tuple[str, ...]
    declared_parent_ids: tuple[str, ...]
    source_trial_id: str | None
    source_outcome: bool | None
    order_key: int | str

    def __post_init__(self) -> None:
        require_declared_parent_support(self.memory_support_ids, self.declared_parent_ids)
