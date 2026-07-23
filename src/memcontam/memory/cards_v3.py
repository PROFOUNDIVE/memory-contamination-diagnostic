from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from memcontam.memory.writer_registry import WriterRegistry


MEMORY_CARD_V3 = "memory_card_v3"


class MemoryEnvelopeError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class MemoryCardEnvelopeV3:
    entry_id: str
    baseline: str
    semantic_kind: str
    schema_version: str
    writer_id: str
    writer_event_id: str
    writer_stage: str
    created_trial_id: str | None
    source_trial_ids: tuple[str, ...]
    source_outcome: bool | None
    trial_support_ids: tuple[str, ...]
    memory_support_ids: tuple[str, ...]
    direct_parent_ids: tuple[str, ...]
    version_predecessor_id: str | None
    order_key: int | str
    native_component: str
    content: str
    content_hash: str


@dataclass(frozen=True)
class ValidatedEnvelope:
    envelope: MemoryCardEnvelopeV3


def canonical_content_hash(content: str) -> str:
    payload = json.dumps({"content": content}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_memory_envelope(
    envelope: MemoryCardEnvelopeV3,
    registry: WriterRegistry,
    prior_entries: Iterable[MemoryCardEnvelopeV3 | ValidatedEnvelope] = (),
) -> ValidatedEnvelope:
    if not isinstance(envelope, MemoryCardEnvelopeV3):
        raise MemoryEnvelopeError("INVALID_SCHEMA")
    _validate_schema(envelope)
    if not registry.permits(envelope):
        raise MemoryEnvelopeError("UNREGISTERED_WRITER_EVENT")
    if envelope.content_hash != canonical_content_hash(envelope.content):
        raise MemoryEnvelopeError("CONTENT_HASH_MISMATCH")
    if not set(envelope.memory_support_ids).issubset(envelope.direct_parent_ids):
        raise MemoryEnvelopeError("SUPPORT_OUTSIDE_PARENTS")
    if envelope.version_predecessor_id in envelope.direct_parent_ids:
        raise MemoryEnvelopeError("VERSION_PREDECESSOR_CONFLATED")

    permission = registry.permission_for(envelope)
    if permission is None:
        raise MemoryEnvelopeError("UNREGISTERED_WRITER_EVENT")
    if permission.requires_source_trial and (
        envelope.created_trial_id is None
        or not envelope.source_trial_ids
        or envelope.created_trial_id not in envelope.source_trial_ids
    ):
        raise MemoryEnvelopeError("MISSING_SOURCE_TRIAL")
    if not set(envelope.source_trial_ids).issubset(envelope.trial_support_ids):
        raise MemoryEnvelopeError("MISSING_TRIAL_SUPPORT")

    prior = tuple(_unwrap(entry) for entry in prior_entries)
    if any(entry is None for entry in prior):
        raise MemoryEnvelopeError("INVALID_SCHEMA")
    entries = tuple(entry for entry in prior if entry is not None)
    if len({entry.entry_id for entry in entries}) != len(entries) or any(
        entry.entry_id == envelope.entry_id for entry in entries
    ):
        raise MemoryEnvelopeError("DUPLICATE_ENTRY")

    by_id = {entry.entry_id: entry for entry in (*entries, envelope)}
    if _has_cycle(by_id):
        raise MemoryEnvelopeError("CYCLE")
    _validate_references(envelope, by_id)
    return ValidatedEnvelope(envelope)


def _unwrap(entry: MemoryCardEnvelopeV3 | ValidatedEnvelope) -> MemoryCardEnvelopeV3 | None:
    if isinstance(entry, ValidatedEnvelope):
        return entry.envelope
    return entry if isinstance(entry, MemoryCardEnvelopeV3) else None


def _validate_schema(envelope: MemoryCardEnvelopeV3) -> None:
    if envelope.schema_version != MEMORY_CARD_V3:
        raise MemoryEnvelopeError("INVALID_SCHEMA")
    if not all(
        _nonempty(value)
        for value in (
            envelope.entry_id,
            envelope.baseline,
            envelope.semantic_kind,
            envelope.writer_id,
            envelope.writer_event_id,
            envelope.writer_stage,
            envelope.native_component,
            envelope.content,
            envelope.content_hash,
        )
    ):
        raise MemoryEnvelopeError("INVALID_SCHEMA")
    if envelope.created_trial_id is not None and not _nonempty(envelope.created_trial_id):
        raise MemoryEnvelopeError("INVALID_SCHEMA")
    if envelope.version_predecessor_id is not None and not _nonempty(envelope.version_predecessor_id):
        raise MemoryEnvelopeError("INVALID_SCHEMA")
    if envelope.source_outcome is not None and not isinstance(envelope.source_outcome, bool):
        raise MemoryEnvelopeError("INVALID_SCHEMA")
    if not _valid_order_key(envelope.order_key) or not all(
        _identifier_tuple(value)
        for value in (
            envelope.source_trial_ids,
            envelope.trial_support_ids,
            envelope.memory_support_ids,
            envelope.direct_parent_ids,
        )
    ):
        raise MemoryEnvelopeError("INVALID_SCHEMA")


def _validate_references(
    envelope: MemoryCardEnvelopeV3, by_id: dict[str, MemoryCardEnvelopeV3]
) -> None:
    for reference_id in (*envelope.direct_parent_ids, envelope.version_predecessor_id):
        if reference_id is None:
            continue
        reference = by_id.get(reference_id)
        if reference is None:
            raise MemoryEnvelopeError("MISSING_REFERENCE")
        if not _precedes(reference.order_key, envelope.order_key):
            raise MemoryEnvelopeError("FUTURE_REFERENCE")
        if reference_id == envelope.version_predecessor_id and (
            reference.baseline,
            reference.semantic_kind,
            reference.native_component,
        ) != (
            envelope.baseline,
            envelope.semantic_kind,
            envelope.native_component,
        ):
            raise MemoryEnvelopeError("VERSION_PREDECESSOR_MISMATCH")


def _has_cycle(entries: dict[str, MemoryCardEnvelopeV3]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(entry_id: str) -> bool:
        if entry_id in visiting:
            return True
        if entry_id in visited:
            return False
        visiting.add(entry_id)
        envelope = entries[entry_id]
        references = (*envelope.direct_parent_ids, envelope.version_predecessor_id)
        for reference_id in references:
            if reference_id in entries and visit(reference_id):
                return True
        visiting.remove(entry_id)
        visited.add(entry_id)
        return False

    return any(visit(entry_id) for entry_id in entries)


def _identifier_tuple(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == len(set(value)) and all(_nonempty(item) for item in value)


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_order_key(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or _nonempty(value)


def _precedes(previous: int | str, current: int | str) -> bool:
    if isinstance(previous, int) and isinstance(current, int):
        return previous < current
    if isinstance(previous, str) and isinstance(current, str):
        return previous < current
    return False
