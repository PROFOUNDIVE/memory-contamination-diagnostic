from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Sequence

from memcontam.memory.cards import MemoryCard, MemoryCardEnvelope
from memcontam.memory.checkpoints import NativeCheckpoint


@dataclass(frozen=True)
class ContaminationRoot:
    card: MemoryCard
    envelope: MemoryCardEnvelope

    def __post_init__(self) -> None:
        if self.card.card_id != self.envelope.entry_id:
            raise ValueError("contamination root card and envelope IDs must match")


class NativeContaminationRenderer:
    def render(
        self, checkpoint: NativeCheckpoint, roots: Sequence[ContaminationRoot]
    ) -> NativeCheckpoint:
        if checkpoint.identity.arm == "clean":
            raise ValueError("cannot render contamination into a clean checkpoint")
        if len(roots) != 1:
            raise ValueError("native contamination rendering requires exactly one root")
        root = roots[0]
        if root.card.card_id in {card.card_id for card in checkpoint.cards}:
            raise ValueError("contamination root ID already exists in checkpoint")
        return replace(
            checkpoint,
            cards=deepcopy(checkpoint.cards) + (deepcopy(root.card),),
            envelopes=deepcopy(checkpoint.envelopes) + (deepcopy(root.envelope),),
            parameters=deepcopy(checkpoint.parameters),
        )
