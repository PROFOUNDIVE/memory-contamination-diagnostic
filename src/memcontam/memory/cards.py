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
    card: MemoryCard
    parent_card_ids: tuple[NonEmptyStr, ...] = ()
    declared_support_ids: tuple[NonEmptyStr, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_declared_parent_support(self.parent_card_ids, self.declared_support_ids)
