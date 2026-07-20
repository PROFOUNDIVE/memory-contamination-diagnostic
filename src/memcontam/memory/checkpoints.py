from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from memcontam.baselines.contracts import StreamIdentity, StreamPairKey, stream_pair_key
from memcontam.memory.cards import MemoryCard, MemoryCardEnvelope


@dataclass(frozen=True)
class NativeCheckpoint:
    identity: StreamIdentity
    cards: tuple[MemoryCard, ...]
    envelopes: tuple[MemoryCardEnvelope, ...]
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        card_ids = tuple(card.card_id for card in self.cards)
        envelope_ids = tuple(envelope.entry_id for envelope in self.envelopes)
        if card_ids != envelope_ids:
            raise ValueError("checkpoint cards and envelopes must have matching IDs and order")
        if len(set(card_ids)) != len(card_ids):
            raise ValueError("checkpoint card IDs must be unique")

    @property
    def checkpoint_hash(self) -> str:
        payload = {
            "identity": asdict(self.identity),
            "cards": [asdict(card) for card in self.cards],
            "envelopes": [asdict(envelope) for envelope in self.envelopes],
            "parameters": self.parameters,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def clone_checkpoint_to_arm(
    checkpoint: NativeCheckpoint,
    arm: Literal["clean", "contaminated", "contaminated_filter"],
) -> NativeCheckpoint:
    if checkpoint.identity.arm == arm:
        raise ValueError("checkpoint clone requires a different arm")
    return replace(
        checkpoint,
        identity=replace(checkpoint.identity, arm=arm),
        cards=deepcopy(checkpoint.cards),
        envelopes=deepcopy(checkpoint.envelopes),
        parameters=deepcopy(checkpoint.parameters),
    )


def validate_stream_pair(
    first: NativeCheckpoint, second: NativeCheckpoint
) -> StreamPairKey:
    if first.identity.arm == second.identity.arm:
        raise ValueError("matched checkpoints must have different arms")
    first_key = stream_pair_key(first.identity)
    second_key = stream_pair_key(second.identity)
    if first_key != second_key:
        raise ValueError("matched checkpoints differ in a non-arm identity field")
    return first_key
