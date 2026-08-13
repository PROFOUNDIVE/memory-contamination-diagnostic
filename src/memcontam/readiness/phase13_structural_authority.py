from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_execution_semantics import PARTITION_SHA256

Baseline = Literal["fh_bounded", "rag_frozen", "bot_style", "reflexion_style"]
BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
REGISTRY_PATH: Final = Path("data/phase13/authority/structural_checkpoint_registry_v1.json")
REGISTRY_HASH: Final = "a0da4b778a6892ec26da1073ee9ee1d7770e6985128468adadf85696b06ddc7a"


class StructuralAuthorityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _ArtifactRef(_StrictModel):
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _CheckpointRef(_StrictModel):
    checkpoint_id: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _StreamAuthority(_StrictModel):
    stream_id: Annotated[str, Field(min_length=1)]
    task: Annotated[str, Field(min_length=1)]
    seed_id: Annotated[int, Field(gt=0)]
    ordered_stream_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    checkpoints: dict[Baseline, _CheckpointRef]


class _Registry(_StrictModel):
    schema_version: Literal["phase13_structural_checkpoint_registry_v1"]
    registry_id: Literal["phase13-structural-checkpoint-registry-v1"]
    source_partition: _ArtifactRef
    streams: tuple[_StreamAuthority, ...]
    registry_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("streams", mode="before")
    @classmethod
    def _streams_tuple(cls, value: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(value)


@dataclass(frozen=True, slots=True)
class RegisteredCheckpoint:
    baseline: Baseline
    checkpoint_id: str
    sha256: str


def registered_checkpoints(stream_id: str, root: Path | None = None) -> tuple[RegisteredCheckpoint, ...]:
    authority_root = root or Path.cwd()
    try:
        raw = read_regular_nofollow(authority_root / REGISTRY_PATH)
        payload = json.loads(raw)
        registry = _Registry.model_validate(payload)
    except (AuthorityFileError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise StructuralAuthorityError("CHECKPOINT_REGISTRY_INVALID") from error
    unsigned = dict(payload)
    unsigned.pop("registry_hash", None)
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != registry.registry_hash or registry.registry_hash != REGISTRY_HASH:
        raise StructuralAuthorityError("CHECKPOINT_REGISTRY_AUTHORITY_MISMATCH")
    if registry.source_partition.sha256 != PARTITION_SHA256:
        raise StructuralAuthorityError("SOURCE_AUTHORITY_HASH_MISMATCH")
    try:
        partition_raw = read_regular_nofollow(authority_root / registry.source_partition.path)
    except AuthorityFileError as error:
        raise StructuralAuthorityError("SOURCE_AUTHORITY_HASH_MISMATCH") from error
    if hashlib.sha256(partition_raw).hexdigest() != registry.source_partition.sha256:
        raise StructuralAuthorityError("SOURCE_AUTHORITY_HASH_MISMATCH")
    stream = next((row for row in registry.streams if row.stream_id == stream_id), None)
    if stream is None:
        raise StructuralAuthorityError("CHECKPOINT_STREAM_UNREGISTERED")
    _validate_stream_source(stream, partition_raw)
    if set(stream.checkpoints) != set(BASELINES):
        raise StructuralAuthorityError("CHECKPOINT_REGISTRY_INVALID")
    return tuple(
        RegisteredCheckpoint(baseline, stream.checkpoints[baseline].checkpoint_id, stream.checkpoints[baseline].sha256)
        for baseline in BASELINES
    )


def _validate_stream_source(stream: _StreamAuthority, partition_raw: bytes) -> None:
    try:
        partition = json.loads(partition_raw)
        task = partition["tasks"][stream.task]
        trajectory = next(row for row in task["trajectories"] if row["seed_id"] == stream.seed_id)
    except (KeyError, TypeError, StopIteration, json.JSONDecodeError) as error:
        raise StructuralAuthorityError("CHECKPOINT_STREAM_UNREGISTERED") from error
    if trajectory["ordered_stream_sha256"] != stream.ordered_stream_sha256:
        raise StructuralAuthorityError("SOURCE_AUTHORITY_HASH_MISMATCH")


__all__ = ("RegisteredCheckpoint", "StructuralAuthorityError", "registered_checkpoints")
