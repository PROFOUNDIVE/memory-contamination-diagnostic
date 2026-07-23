from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


CHECKPOINT_V3 = "checkpoint_v3"
NATIVE_ENTRY_V1 = "phase12_native_entry_v1"


class CheckpointError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class NativeEntry:
    entry_id: str
    semantic_kind: str
    schema_version: str
    native_component: str
    content: str
    content_hash: str
    direct_parent_ids: tuple[str, ...] = ()
    render_id: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "content_hash": self.content_hash,
            "direct_parent_ids": list(self.direct_parent_ids),
            "entry_id": self.entry_id,
            "native_component": self.native_component,
            "render_id": self.render_id,
            "schema_version": self.schema_version,
            "semantic_kind": self.semantic_kind,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NativeEntry:
        return cls(
            entry_id=value["entry_id"],
            semantic_kind=value["semantic_kind"],
            schema_version=value["schema_version"],
            native_component=value["native_component"],
            content=value["content"],
            content_hash=value["content_hash"],
            direct_parent_ids=tuple(value.get("direct_parent_ids", ())),
            render_id=value.get("render_id"),
        )


@dataclass(frozen=True)
class NativeState:
    baseline: str
    entries: tuple[str | NativeEntry, ...]
    native_state: Mapping[str, Any]
    schema_version: str = CHECKPOINT_V3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NativeState:
        entries = tuple(
            NativeEntry.from_mapping(entry) if isinstance(entry, dict) else entry
            for entry in value["entries"]
        )
        return cls(
            baseline=value["baseline"],
            entries=entries,
            native_state=value["native_state"],
            schema_version=value.get("schema_version", CHECKPOINT_V3),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "entries": [
                entry.to_mapping() if isinstance(entry, NativeEntry) else entry
                for entry in self.entries
            ],
            "native_state": self.native_state,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class Phase12CheckpointIdentity:
    checkpoint_id: str
    baseline: str
    sha256: str


FROZEN_CHECKPOINT_IDENTITIES = {
    "b4f050455d0cce7d9799c82e3420e7c7ede12b14db1563625a01235a2ddbf198": (
        Phase12CheckpointIdentity(
            checkpoint_id="ckpt-bot-s1-t2",
            baseline="bot_style",
            sha256="59fc6b792d1f9c960b4c6f0384582521886e104186dac2493e1da90da51d40ba",
        )
    ),
    "4a7fc17400a86760882331ebafea71f5859a9453804672ac0ea6e16daa7d6a78": (
        Phase12CheckpointIdentity(
            checkpoint_id="ckpt-fh-s1-t2",
            baseline="fh_bounded",
            sha256="b8e26b85c7ba1c95c2b194385f089109c34b3573931d4f65ac0594af8343a77b",
        )
    ),
    "21a39cd29a6e6d58ab408e108cfa37c430354fed33d75ff41d5db722e90a98a3": (
        Phase12CheckpointIdentity(
            checkpoint_id="ckpt-rag-s1-t2",
            baseline="rag_frozen",
            sha256="136af677e8bbd631d2dea943775d14931beb7bfa44f35ff6fc0e18e5878abdfb",
        )
    ),
    "49d23888eb7e2bbbd45ec9e8495e94c7060f37b9f4b672503727b5380992fb40": (
        Phase12CheckpointIdentity(
            checkpoint_id="ckpt-ref-s1-t2",
            baseline="reflexion_style",
            sha256="75dc818b4db257f6deae6f517b967d666c631844a35ccc35f7eed6c17bae9620",
        )
    ),
}


@dataclass(frozen=True)
class Phase12Checkpoint:
    identity: Phase12CheckpointIdentity
    state: NativeState
    canonical_bytes: bytes
    canonical_sha256: str


def serialize_checkpoint(state: NativeState, *, registry=None) -> Phase12Checkpoint:
    registry = _registry_or_native(registry)
    registry.validate(state)
    canonical_bytes = _canonical_bytes(state)
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    identity = FROZEN_CHECKPOINT_IDENTITIES.get(canonical_sha256)
    if identity is None:
        identity = Phase12CheckpointIdentity(
            checkpoint_id=f"checkpoint-{canonical_sha256[:16]}",
            baseline=state.baseline,
            sha256=canonical_sha256,
        )
    return Phase12Checkpoint(
        identity=identity,
        state=state,
        canonical_bytes=canonical_bytes,
        canonical_sha256=canonical_sha256,
    )


def deserialize_checkpoint(checkpoint: Phase12Checkpoint, *, registry=None) -> NativeState:
    if not isinstance(checkpoint, Phase12Checkpoint):
        raise CheckpointError("INVALID_CHECKPOINT")
    canonical_bytes = _canonical_bytes(checkpoint.state)
    if (
        canonical_bytes != checkpoint.canonical_bytes
        or hashlib.sha256(canonical_bytes).hexdigest() != checkpoint.canonical_sha256
        or checkpoint.identity.baseline != checkpoint.state.baseline
    ):
        raise CheckpointError("CHECKPOINT_HASH_MISMATCH")
    expected_identity = FROZEN_CHECKPOINT_IDENTITIES.get(checkpoint.canonical_sha256)
    if expected_identity is not None and checkpoint.identity != expected_identity:
        raise CheckpointError("CHECKPOINT_IDENTITY_MISMATCH")
    if expected_identity is None and checkpoint.identity != Phase12CheckpointIdentity(
        checkpoint_id=f"checkpoint-{checkpoint.canonical_sha256[:16]}",
        baseline=checkpoint.state.baseline,
        sha256=checkpoint.canonical_sha256,
    ):
        raise CheckpointError("CHECKPOINT_IDENTITY_MISMATCH")
    registry = _registry_or_native(registry)
    registry.validate(checkpoint.state)
    return checkpoint.state


def append_native_entry(
    checkpoint: Phase12Checkpoint, entry: NativeEntry, *, registry=None
) -> Phase12Checkpoint:
    state = deserialize_checkpoint(checkpoint, registry=registry)
    if entry.entry_id in _entry_ids(state.entries):
        raise CheckpointError("DUPLICATE_ROOT")
    return serialize_checkpoint(
        NativeState(
            baseline=state.baseline,
            entries=(*state.entries, entry),
            native_state=state.native_state,
            schema_version=state.schema_version,
        ),
        registry=registry,
    )


def _canonical_bytes(state: NativeState) -> bytes:
    return json.dumps(state.to_mapping(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _entry_ids(entries: tuple[str | NativeEntry, ...]) -> set[str]:
    return {entry.entry_id if isinstance(entry, NativeEntry) else entry for entry in entries}


def _registry_or_native(registry):
    if registry is not None:
        return registry
    from memcontam.memory.serializer_registry import SerializerRegistry

    return SerializerRegistry.native()
