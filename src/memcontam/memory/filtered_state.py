from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Literal, Sequence

from memcontam.memory.admission import (
    AdmissionContext,
    AdmissionDecision,
    AdmissionError,
    evaluate_admission,
)
from memcontam.memory.checkpoints import NativeCheckpoint
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3
from memcontam.memory.checkpoint_v3 import (
    CheckpointError,
    NativeEntry,
    NativeState,
    Phase12Checkpoint,
    append_native_entry,
    deserialize_checkpoint,
    serialize_checkpoint,
)


@dataclass(frozen=True)
class PartitionDecision:
    entry_id: str
    state: Literal["active", "quarantine"]
    reason: str | None


@dataclass(frozen=True)
class FilteredNativeState:
    active: NativeCheckpoint
    quarantine: NativeCheckpoint
    decisions: tuple[PartitionDecision, ...]


@dataclass(frozen=True)
class CandidateWrite:
    entry: NativeEntry
    envelope: MemoryCardEnvelopeV3


@dataclass(frozen=True)
class FilteredCheckpoint:
    source_checkpoint: Phase12Checkpoint
    active: Phase12Checkpoint
    quarantine: Phase12Checkpoint
    active_envelopes: tuple[MemoryCardEnvelopeV3, ...]
    quarantined_envelopes: tuple[MemoryCardEnvelopeV3, ...]
    decisions: tuple[PartitionDecision, ...]

    @property
    def reader_entries(self) -> tuple[str | NativeEntry, ...]:
        return self.active.state.entries

    @property
    def updater_entries(self) -> tuple[str | NativeEntry, ...]:
        return self.active.state.entries


@dataclass(frozen=True)
class FilterTransition:
    state: FilteredCheckpoint
    decision: AdmissionDecision

    @property
    def reader_entries(self) -> tuple[str | NativeEntry, ...]:
        return self.state.reader_entries

    @property
    def updater_entries(self) -> tuple[str | NativeEntry, ...]:
        return self.state.updater_entries


def partition_native_checkpoint(
    checkpoint: Phase12Checkpoint, context: AdmissionContext
) -> FilteredCheckpoint:
    try:
        state = deserialize_checkpoint(checkpoint)
    except CheckpointError as error:
        raise AdmissionError(error.code) from error

    envelopes_by_id = _envelopes_by_id(context.evidence_envelopes)
    active_entries: list[str | NativeEntry] = []
    quarantine_entries: list[str | NativeEntry] = []
    active_envelopes: list[MemoryCardEnvelopeV3] = list(context.active_envelopes)
    quarantined_envelopes: list[MemoryCardEnvelopeV3] = list(context.quarantined_envelopes)
    decisions: list[PartitionDecision] = []

    for entry in state.entries:
        entry_id = _entry_id(entry)
        envelope = envelopes_by_id.get(entry_id)
        if envelope is None:
            decision = AdmissionDecision(entry_id, False, "MISSING_SUPPORT_EVIDENCE")
        else:
            decision = evaluate_admission(
                envelope,
                replace(
                    context,
                    active_envelopes=tuple(active_envelopes),
                    quarantined_envelopes=tuple(quarantined_envelopes),
                ),
            )
        if decision.admitted:
            if envelope is None:
                raise AdmissionError("MISSING_SUPPORT_EVIDENCE")
            active_entries.append(entry)
            active_envelopes.append(envelope)
            decisions.append(PartitionDecision(entry_id, "active", None))
        else:
            quarantine_entries.append(entry)
            if envelope is not None:
                quarantined_envelopes.append(envelope)
            decisions.append(PartitionDecision(entry_id, "quarantine", decision.reason))

    return FilteredCheckpoint(
        source_checkpoint=checkpoint,
        active=_checkpoint_with_entries(state, active_entries),
        quarantine=_checkpoint_with_entries(state, quarantine_entries),
        active_envelopes=tuple(active_envelopes),
        quarantined_envelopes=tuple(quarantined_envelopes),
        decisions=tuple(decisions),
    )


def route_candidate_write(
    state: FilteredCheckpoint, candidate: CandidateWrite, context: AdmissionContext
) -> FilterTransition:
    _validate_candidate(candidate, state)
    evidence = _merge_envelopes(
        context.evidence_envelopes,
        state.active_envelopes,
        state.quarantined_envelopes,
        (candidate.envelope,),
    )
    decision = evaluate_admission(
        candidate.envelope,
        replace(
            context,
            evidence_envelopes=evidence,
            active_envelopes=state.active_envelopes,
            quarantined_envelopes=state.quarantined_envelopes,
        ),
    )
    try:
        active = state.active
        quarantine = state.quarantine
        active_envelopes = state.active_envelopes
        quarantined_envelopes = state.quarantined_envelopes
        if decision.admitted:
            active = append_native_entry(active, candidate.entry)
            active_envelopes = (*active_envelopes, candidate.envelope)
            partition = PartitionDecision(candidate.envelope.entry_id, "active", None)
        else:
            quarantine = append_native_entry(quarantine, candidate.entry)
            quarantined_envelopes = (*quarantined_envelopes, candidate.envelope)
            partition = PartitionDecision(
                candidate.envelope.entry_id, "quarantine", decision.reason
            )
    except CheckpointError as error:
        raise AdmissionError(error.code) from error

    return FilterTransition(
        state=FilteredCheckpoint(
            source_checkpoint=state.source_checkpoint,
            active=active,
            quarantine=quarantine,
            active_envelopes=active_envelopes,
            quarantined_envelopes=quarantined_envelopes,
            decisions=(*state.decisions, partition),
        ),
        decision=decision,
    )


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
    if (
        partition.active.identity != filtered_identity
        or partition.quarantine.identity != filtered_identity
    ):
        raise ValueError("filtered checkpoints differ in a non-arm identity field")
    if (
        partition.active.parameters != source.parameters
        or partition.quarantine.parameters != source.parameters
    ):
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
            if card != source_cards.get(card.card_id) or envelope != source_envelopes.get(
                envelope.entry_id
            ):
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


def _envelopes_by_id(
    envelopes: Sequence[MemoryCardEnvelopeV3],
) -> dict[str, MemoryCardEnvelopeV3]:
    by_id: dict[str, MemoryCardEnvelopeV3] = {}
    for envelope in envelopes:
        if not isinstance(envelope, MemoryCardEnvelopeV3) or envelope.entry_id in by_id:
            raise AdmissionError("INVALID_ENVELOPE_EVIDENCE")
        by_id[envelope.entry_id] = envelope
    return by_id


def _entry_id(entry: str | NativeEntry) -> str:
    if isinstance(entry, NativeEntry):
        return entry.entry_id
    if isinstance(entry, str) and entry:
        return entry
    raise AdmissionError("INVALID_NATIVE_ENTRY")


def _checkpoint_with_entries(
    source: NativeState, entries: Sequence[str | NativeEntry]
) -> Phase12Checkpoint:
    try:
        return serialize_checkpoint(
            NativeState(
                baseline=source.baseline,
                entries=tuple(entries),
                native_state=source.native_state,
                schema_version=source.schema_version,
            )
        )
    except CheckpointError as error:
        raise AdmissionError(error.code) from error


def _validate_candidate(candidate: CandidateWrite, state: FilteredCheckpoint) -> None:
    if not isinstance(candidate, CandidateWrite) or not isinstance(candidate.entry, NativeEntry):
        raise AdmissionError("INVALID_NATIVE_ENTRY")
    envelope = candidate.envelope
    if not isinstance(envelope, MemoryCardEnvelopeV3) or (
        candidate.entry.entry_id,
        candidate.entry.semantic_kind,
        candidate.entry.native_component,
        candidate.entry.content,
        candidate.entry.content_hash,
    ) != (
        envelope.entry_id,
        envelope.semantic_kind,
        envelope.native_component,
        envelope.content,
        envelope.content_hash,
    ):
        raise AdmissionError("ENTRY_ENVELOPE_MISMATCH")
    existing_ids = {
        _entry_id(entry) for entry in (*state.active.state.entries, *state.quarantine.state.entries)
    }
    if candidate.entry.entry_id in existing_ids:
        raise AdmissionError("DUPLICATE_ENTRY")


def _merge_envelopes(
    *groups: Sequence[MemoryCardEnvelopeV3],
) -> tuple[MemoryCardEnvelopeV3, ...]:
    merged: dict[str, MemoryCardEnvelopeV3] = {}
    for group in groups:
        for envelope in group:
            if isinstance(envelope, MemoryCardEnvelopeV3):
                merged.setdefault(envelope.entry_id, envelope)
    return tuple(merged.values())
