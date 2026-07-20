from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Literal, Sequence

from memcontam.memory.admission import AdmissionDecision
from memcontam.memory.checkpoints import NativeCheckpoint


@dataclass(frozen=True)
class PartitionDecision:
    entry_id: str
    state: Literal["active", "quarantine"]
    reason: str


@dataclass(frozen=True)
class FilteredNativeState:
    active: NativeCheckpoint
    quarantine: NativeCheckpoint
    decisions: tuple[PartitionDecision, ...]


def partition_native_state_for_fixture(
    checkpoint: NativeCheckpoint,
    admission_decisions: Sequence[AdmissionDecision],
) -> FilteredNativeState:
    decisions_by_id = _decision_map(checkpoint, admission_decisions)
    filtered_identity = replace(checkpoint.identity, arm="contaminated_filter")
    active_cards = []
    active_envelopes = []
    quarantine_cards = []
    quarantine_envelopes = []
    partition_decisions = []

    for card, envelope in zip(checkpoint.cards, checkpoint.envelopes, strict=True):
        decision = decisions_by_id[card.card_id]
        if decision.admitted:
            active_cards.append(deepcopy(card))
            active_envelopes.append(deepcopy(envelope))
            state: Literal["active", "quarantine"] = "active"
        else:
            quarantine_cards.append(deepcopy(card))
            quarantine_envelopes.append(deepcopy(envelope))
            state = "quarantine"
        partition_decisions.append(PartitionDecision(card.card_id, state, decision.reason))

    partition = FilteredNativeState(
        active=NativeCheckpoint(
            identity=filtered_identity,
            cards=tuple(active_cards),
            envelopes=tuple(active_envelopes),
            parameters=deepcopy(checkpoint.parameters),
        ),
        quarantine=NativeCheckpoint(
            identity=filtered_identity,
            cards=tuple(quarantine_cards),
            envelopes=tuple(quarantine_envelopes),
            parameters=deepcopy(checkpoint.parameters),
        ),
        decisions=tuple(partition_decisions),
    )
    validate_partition_preserves_native_contract(checkpoint, partition)
    return partition


def serialize_filtered_state(state: FilteredNativeState) -> str:
    return json.dumps(
        {
            "active": _serialize_checkpoint(state.active),
            "decisions": [asdict(decision) for decision in state.decisions],
            "quarantine": _serialize_checkpoint(state.quarantine),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def validate_partition_preserves_native_contract(
    source: NativeCheckpoint, partition: FilteredNativeState
) -> None:
    filtered_identity = replace(source.identity, arm="contaminated_filter")
    if partition.active.identity != filtered_identity or partition.quarantine.identity != filtered_identity:
        raise ValueError("filtered checkpoints differ in a non-arm identity field")
    if partition.active.parameters != source.parameters or partition.quarantine.parameters != source.parameters:
        raise ValueError("filtered checkpoint parameters changed")

    source_cards = {card.card_id: card for card in source.cards}
    source_envelopes = {envelope.entry_id: envelope for envelope in source.envelopes}
    active_ids = tuple(card.card_id for card in partition.active.cards)
    quarantine_ids = tuple(card.card_id for card in partition.quarantine.cards)
    if set(active_ids) & set(quarantine_ids):
        raise ValueError("active and quarantine states must be disjoint")
    if set(active_ids) | set(quarantine_ids) != set(source_cards):
        raise ValueError("active and quarantine states must be total over source entries")

    for checkpoint, entry_ids in (
        (partition.active, active_ids),
        (partition.quarantine, quarantine_ids),
    ):
        expected_ids = tuple(card.card_id for card in source.cards if card.card_id in entry_ids)
        if entry_ids != expected_ids:
            raise ValueError("filtered checkpoint changes native ordering")
        for card, envelope in zip(checkpoint.cards, checkpoint.envelopes, strict=True):
            if card != source_cards.get(card.card_id) or envelope != source_envelopes.get(envelope.entry_id):
                raise ValueError("filtered checkpoint changes native entries")

    if tuple(decision.entry_id for decision in partition.decisions) != tuple(source_cards):
        raise ValueError("partition decisions must be total and source ordered")
    for decision in partition.decisions:
        if decision.state not in {"active", "quarantine"}:
            raise ValueError("partition decision has an unknown state")
        if (decision.entry_id in active_ids) != (decision.state == "active"):
            raise ValueError("partition decisions do not match active membership")


def _decision_map(
    checkpoint: NativeCheckpoint, admission_decisions: Sequence[AdmissionDecision]
) -> dict[str, AdmissionDecision]:
    decisions = tuple(admission_decisions)
    if not all(isinstance(decision, AdmissionDecision) for decision in decisions):
        raise ValueError("invalid admission decision")
    decision_ids = tuple(decision.entry_id for decision in decisions)
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("duplicate admission decision IDs")
    source_ids = {card.card_id for card in checkpoint.cards}
    unknown_ids = set(decision_ids) - source_ids
    if unknown_ids:
        raise ValueError("unknown admission decision ID")
    if set(decision_ids) != source_ids:
        raise ValueError("admission decisions must form a total partition")
    return {decision.entry_id: decision for decision in decisions}


def _serialize_checkpoint(checkpoint: NativeCheckpoint) -> dict[str, object]:
    return {
        "cards": [asdict(card) for card in checkpoint.cards],
        "envelopes": [asdict(envelope) for envelope in checkpoint.envelopes],
        "identity": asdict(checkpoint.identity),
        "parameters": checkpoint.parameters,
    }
