from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from memcontam.memory.cards import MemoryCard, MemoryCardEnvelope
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3, MemoryEnvelopeError, validate_memory_envelope
from memcontam.memory.writer_registry import WriterRegistry


class AdmissionGraphError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class AdmissionError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class AuthorizedWriterRegistry:
    writer_ids: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "writer_ids", frozenset(self.writer_ids))

    def permits(self, writer_id: str) -> bool:
        return writer_id in self.writer_ids


@dataclass(frozen=True)
class AdmissionContext:
    authorized_writers: AuthorizedWriterRegistry | None = None
    trial_log_support_ids: frozenset[str] = frozenset()
    admitted_envelopes: tuple[MemoryCardEnvelope, ...] = ()
    writer_registry: WriterRegistry = field(default_factory=WriterRegistry.native)
    writer_event_ids: frozenset[str] = frozenset()
    trial_record_ids: frozenset[str] = frozenset()
    evidence_envelopes: tuple[MemoryCardEnvelopeV3, ...] = ()
    active_envelopes: tuple[MemoryCardEnvelopeV3, ...] = ()
    quarantined_envelopes: tuple[MemoryCardEnvelopeV3, ...] = ()
    semantic_kind_support: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_log_support_ids", frozenset(self.trial_log_support_ids))
        object.__setattr__(self, "admitted_envelopes", tuple(self.admitted_envelopes))
        object.__setattr__(self, "writer_event_ids", frozenset(self.writer_event_ids))
        object.__setattr__(self, "trial_record_ids", frozenset(self.trial_record_ids))
        object.__setattr__(self, "evidence_envelopes", tuple(self.evidence_envelopes))
        object.__setattr__(self, "active_envelopes", tuple(self.active_envelopes))
        object.__setattr__(self, "quarantined_envelopes", tuple(self.quarantined_envelopes))
        object.__setattr__(
            self,
            "semantic_kind_support",
            {kind: frozenset(supported) for kind, supported in self.semantic_kind_support.items()},
        )


@dataclass(frozen=True)
class AdmissionDecision:
    entry_id: str
    admitted: bool
    reason: str


def validate_support_reference(
    envelope: MemoryCardEnvelope,
    *,
    available_entry_ids: frozenset[str],
    trial_log_support_ids: frozenset[str],
) -> None:
    supports = envelope.memory_support_ids
    parents = envelope.declared_parent_ids
    if not set(supports).issubset(parents):
        raise AdmissionGraphError("invalid_support")
    if not set(envelope.trial_log_support_ids).issubset(trial_log_support_ids):
        raise AdmissionGraphError("missing_reference")
    if not set(supports).issubset(available_entry_ids):
        raise AdmissionGraphError("missing_reference")


def validate_parent_graph(
    envelopes: Sequence[MemoryCardEnvelope],
    *,
    admitted_envelopes: Sequence[MemoryCardEnvelope] = (),
) -> None:
    current_envelopes = tuple(envelopes)
    if (
        len(current_envelopes) != sum(
            isinstance(envelope, MemoryCardEnvelope) for envelope in current_envelopes
        )
        or len({envelope.entry_id for envelope in current_envelopes}) != len(current_envelopes)
    ):
        raise AdmissionGraphError("invalid_schema")
    issues = _parent_issues(current_envelopes, admitted_envelopes)
    if issues:
        raise AdmissionGraphError(next(iter(issues.values())))
    if _cycle_entry_ids(current_envelopes):
        raise AdmissionGraphError("cycle")


def evaluate_entry_admission(
    card: MemoryCard, envelope: MemoryCardEnvelope, context: AdmissionContext
) -> AdmissionDecision:
    return evaluate_admission_graph(((card, envelope),), context)[0]


def evaluate_admission(
    envelope: MemoryCardEnvelopeV3, context: AdmissionContext
) -> AdmissionDecision:
    """Evaluate a Phase-12 write from operational evidence only."""
    if not isinstance(envelope, MemoryCardEnvelopeV3):
        return AdmissionDecision("", False, "INVALID_ENVELOPE")

    prior_entries = _unique_v3_envelopes(
        (
            *context.evidence_envelopes,
            *context.active_envelopes,
            *context.quarantined_envelopes,
        ),
        excluding=envelope.entry_id,
    )
    try:
        validate_memory_envelope(envelope, context.writer_registry, prior_entries)
    except MemoryEnvelopeError as error:
        return AdmissionDecision(envelope.entry_id, False, _quarantine_reason(error.code))

    if envelope.writer_event_id not in context.writer_event_ids:
        return AdmissionDecision(envelope.entry_id, False, "UNREGISTERED_WRITER_EVENT")
    if not set(envelope.trial_support_ids).issubset(context.trial_record_ids):
        return AdmissionDecision(envelope.entry_id, False, "MISSING_SUPPORT_EVIDENCE")

    active = _v3_envelopes_by_id(context.active_envelopes)
    quarantined = _v3_envelopes_by_id(context.quarantined_envelopes)
    for parent_id in envelope.direct_parent_ids:
        if parent_id in quarantined:
            return AdmissionDecision(envelope.entry_id, False, "PARENT_QUARANTINED")
        if parent_id not in active:
            return AdmissionDecision(envelope.entry_id, False, "MISSING_PARENT_EVIDENCE")

    allowed_support_kinds = context.semantic_kind_support.get(envelope.semantic_kind)
    if allowed_support_kinds is not None:
        if any(
            active[parent_id].semantic_kind not in allowed_support_kinds
            for parent_id in envelope.memory_support_ids
        ):
            return AdmissionDecision(envelope.entry_id, False, "INVALID_SUPPORT_EVIDENCE")

    predecessor_id = envelope.version_predecessor_id
    if predecessor_id is not None:
        predecessor = active.get(predecessor_id)
        if predecessor is None or predecessor_id in quarantined:
            return AdmissionDecision(envelope.entry_id, False, "INVALID_VERSION_EVIDENCE")
        if (
            predecessor.baseline,
            predecessor.semantic_kind,
            predecessor.native_component,
        ) != (
            envelope.baseline,
            envelope.semantic_kind,
            envelope.native_component,
        ):
            return AdmissionDecision(envelope.entry_id, False, "INVALID_VERSION_EVIDENCE")

    return AdmissionDecision(envelope.entry_id, True, "ADMITTED")


def evaluate_admission_graph(
    entries: Sequence[object], context: AdmissionContext
) -> tuple[AdmissionDecision, ...]:
    pairs = tuple(_entry_pair(entry) for entry in entries)
    envelopes = tuple(envelope for _, envelope in pairs)
    schema_issues = _schema_issues(pairs)
    candidate_ids = frozenset(
        envelope.entry_id for envelope in envelopes if isinstance(envelope, MemoryCardEnvelope)
    )
    available_ids = candidate_ids | frozenset(
        envelope.entry_id
        for envelope in context.admitted_envelopes
        if isinstance(envelope, MemoryCardEnvelope)
    )
    issues = dict(schema_issues)
    admitted_entries = tuple(
        envelope
        for envelope in context.admitted_envelopes
        if isinstance(envelope, MemoryCardEnvelope)
    )
    if (
        len(admitted_entries) != len(context.admitted_envelopes)
        or len({envelope.entry_id for envelope in admitted_entries}) != len(admitted_entries)
    ):
        issues.update(
            {
                envelope.entry_id: "invalid_schema"
                for envelope in envelopes
                if isinstance(envelope, MemoryCardEnvelope)
            }
        )
    admitted_ids = {envelope.entry_id for envelope in admitted_entries}
    for envelope in envelopes:
        if isinstance(envelope, MemoryCardEnvelope) and envelope.entry_id in admitted_ids:
            issues[envelope.entry_id] = "invalid_schema"

    for envelope in envelopes:
        if not isinstance(envelope, MemoryCardEnvelope) or envelope.entry_id in issues:
            continue
        try:
            validate_support_reference(
                envelope,
                available_entry_ids=available_ids,
                trial_log_support_ids=context.trial_log_support_ids,
            )
        except AdmissionGraphError as error:
            issues[envelope.entry_id] = error.reason

    for entry_id, reason in _parent_issues(envelopes, context.admitted_envelopes).items():
        issues.setdefault(entry_id, reason)
    for entry_id in _cycle_entry_ids(envelopes):
        issues[entry_id] = "cycle"

    by_id = {envelope.entry_id: envelope for envelope in envelopes if isinstance(envelope, MemoryCardEnvelope)}
    decisions: dict[str, AdmissionDecision] = {}

    def evaluate(entry_id: str) -> AdmissionDecision:
        if entry_id in decisions:
            return decisions[entry_id]
        envelope = by_id[entry_id]
        if entry_id in issues:
            decision = AdmissionDecision(entry_id, False, issues[entry_id])
        elif context.authorized_writers is None or not context.authorized_writers.permits(envelope.writer_id):
            decision = AdmissionDecision(entry_id, False, "unauthorized_writer")
        elif any(
            not evaluate(parent_id).admitted
            for parent_id in envelope.declared_parent_ids
            if parent_id not in admitted_ids
        ):
            decision = AdmissionDecision(entry_id, False, "rejected_parent")
        else:
            decision = AdmissionDecision(entry_id, True, "admitted")
        decisions[entry_id] = decision
        return decision

    return tuple(
        evaluate(envelope.entry_id)
        if isinstance(envelope, MemoryCardEnvelope) and envelope.entry_id in by_id
        else AdmissionDecision(_entry_id(card, envelope), False, "invalid_schema")
        for card, envelope in pairs
    )


def _schema_issues(
    entries: Sequence[tuple[object, object]],
) -> dict[str, str]:
    issues: dict[str, str] = {}
    seen_ids: set[str] = set()
    for card, envelope in entries:
        entry_id = _entry_id(card, envelope)
        if not _valid_entry_schema(card, envelope) or entry_id in seen_ids:
            issues[entry_id] = "invalid_schema"
        seen_ids.add(entry_id)
    return issues


def _parent_issues(
    envelopes: Sequence[object], admitted_envelopes: Sequence[MemoryCardEnvelope]
) -> dict[str, str]:
    all_envelopes = tuple(
        envelope
        for envelope in (*admitted_envelopes, *envelopes)
        if isinstance(envelope, MemoryCardEnvelope)
    )
    by_id = {envelope.entry_id: envelope for envelope in all_envelopes}
    issues: dict[str, str] = {}
    for envelope in envelopes:
        if not isinstance(envelope, MemoryCardEnvelope):
            continue
        for parent_id in envelope.declared_parent_ids:
            parent = by_id.get(parent_id)
            if parent is None:
                issues[envelope.entry_id] = "missing_reference"
                break
            if not _precedes(parent.order_key, envelope.order_key):
                issues[envelope.entry_id] = "future_reference"
                break
    return issues


def _cycle_entry_ids(envelopes: Sequence[object]) -> frozenset[str]:
    candidate_ids = {envelope.entry_id for envelope in envelopes if isinstance(envelope, MemoryCardEnvelope)}
    parents = {
        envelope.entry_id: tuple(parent for parent in envelope.declared_parent_ids if parent in candidate_ids)
        for envelope in envelopes
        if isinstance(envelope, MemoryCardEnvelope)
    }
    visiting: list[str] = []
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(entry_id: str) -> None:
        if entry_id in visiting:
            cycles.update(visiting[visiting.index(entry_id) :])
            return
        if entry_id in visited:
            return
        visiting.append(entry_id)
        for parent_id in parents[entry_id]:
            visit(parent_id)
        visiting.pop()
        visited.add(entry_id)

    for entry_id in parents:
        visit(entry_id)
    return frozenset(cycles)


def _valid_entry_schema(card: object, envelope: object) -> bool:
    if not isinstance(card, MemoryCard) or not isinstance(envelope, MemoryCardEnvelope):
        return False
    if card.card_id != envelope.entry_id or card.card_type != envelope.semantic_kind:
        return False
    if not all(
        _nonempty_string(value)
        for value in (
            card.card_id,
            card.content,
            card.card_type,
            envelope.entry_id,
            envelope.semantic_kind,
            envelope.writer_id,
            envelope.writer_event_id,
        )
    ):
        return False
    if not all(
        _identifier_tuple(value)
        for value in (
            envelope.trial_log_support_ids,
            envelope.memory_support_ids,
            envelope.declared_parent_ids,
        )
    ):
        return False
    if envelope.source_trial_id is not None and not _nonempty_string(envelope.source_trial_id):
        return False
    if envelope.source_outcome is not None and not isinstance(envelope.source_outcome, bool):
        return False
    return _valid_order_key(envelope.order_key)


def _entry_id(card: object, envelope: object) -> str:
    if isinstance(envelope, MemoryCardEnvelope) and isinstance(envelope.entry_id, str):
        return envelope.entry_id
    if isinstance(card, MemoryCard) and isinstance(card.card_id, str):
        return card.card_id
    return ""


def _entry_pair(entry: object) -> tuple[object, object]:
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry
    return None, None


def _identifier_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(_nonempty_string(item) for item in value)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_order_key(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or _nonempty_string(value)


def _precedes(parent_order: int | str, child_order: int | str) -> bool:
    if isinstance(parent_order, int) and isinstance(child_order, int):
        return parent_order < child_order
    if isinstance(parent_order, str) and isinstance(child_order, str):
        return parent_order < child_order
    return False


def _quarantine_reason(code: str) -> str:
    if code == "UNREGISTERED_WRITER_EVENT":
        return code
    if code in {"MISSING_SOURCE_TRIAL", "MISSING_TRIAL_SUPPORT", "SUPPORT_OUTSIDE_PARENTS"}:
        return "MISSING_SUPPORT_EVIDENCE"
    if code == "MISSING_REFERENCE":
        return "MISSING_PARENT_EVIDENCE"
    if code in {"VERSION_PREDECESSOR_CONFLATED", "VERSION_PREDECESSOR_MISMATCH"}:
        return "INVALID_VERSION_EVIDENCE"
    if code == "FUTURE_REFERENCE":
        return "INVALID_PARENT_EVIDENCE"
    return "INVALID_ENVELOPE"


def _v3_envelopes_by_id(
    envelopes: Sequence[MemoryCardEnvelopeV3],
) -> dict[str, MemoryCardEnvelopeV3]:
    return {
        envelope.entry_id: envelope
        for envelope in envelopes
        if isinstance(envelope, MemoryCardEnvelopeV3)
    }


def _unique_v3_envelopes(
    envelopes: Sequence[MemoryCardEnvelopeV3], *, excluding: str
) -> tuple[MemoryCardEnvelopeV3, ...]:
    by_id: dict[str, MemoryCardEnvelopeV3] = {}
    for known in envelopes:
        if isinstance(known, MemoryCardEnvelopeV3) and known.entry_id != excluding:
            by_id.setdefault(known.entry_id, known)
    return tuple(by_id.values())
