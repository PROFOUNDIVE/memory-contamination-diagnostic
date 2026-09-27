from __future__ import annotations

import hashlib
import json
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from memcontam.contamination.phase12.models import (
    CandidateRegistry,
    CandidateVariant,
    canonical_content_hash,
)
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry


LegacyTask = Literal["game24", "math_equation_balancer", "word_sorting"]
LegacyRole = Literal["false", "correct", "irrelevant"]
_TASKS: Final = frozenset({"game24", "math_equation_balancer", "word_sorting"})
_REGISTRY_SHA256S: Final = frozenset(
    {
        "5c25b567b2d8edf6ee19c6bf9c83e8c480ada06eb41787e385994a4319bfaed0",
        "e80c895f1cab313980e43531d1da22875956f1ab238a7c42ae536059f8b83292",
    }
)


class LegacyDcRsRegistryError(ValueError):
    def __init__(self) -> None:
        super().__init__("LEGACY_DC_RS_REGISTRY_INVALID")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuthorityBinding(_FrozenModel):
    authority_id: str
    sha256: str


class LegacyDcRsRecord(_FrozenModel):
    role: LegacyRole
    candidate_id: str
    query: str
    response: str
    query_sha256: str
    response_sha256: str
    serialized_content_sha256: str
    render_id: str

    @property
    def serialized_content(self) -> str:
        return json.dumps(
            {"input": self.query, "raw_output": self.response},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        if (
            self.query_sha256 != hashlib.sha256(self.query.encode()).hexdigest()
            or self.response_sha256 != hashlib.sha256(self.response.encode()).hexdigest()
            or self.serialized_content_sha256
            != hashlib.sha256(self.serialized_content.encode()).hexdigest()
            or self.render_id
            not in {
                f"legacy-dc-rs-render-v1::{self.candidate_id}::{self.serialized_content_sha256[:16]}",
                f"legacy-dc-rs-render-v2::{self.candidate_id}::{self.serialized_content_sha256[:16]}",
            }
        ):
            raise LegacyDcRsRegistryError()
        return self


class LegacyDcRsTaskRecords(_FrozenModel):
    task: LegacyTask
    triplet_id: str
    archive_position: Literal["append_at_tau_star"]
    records: tuple[LegacyDcRsRecord, LegacyDcRsRecord, LegacyDcRsRecord]

    @model_validator(mode="after")
    def validate_matching(self) -> Self:
        if (
            {record.role for record in self.records} != {"false", "correct", "irrelevant"}
            or len({record.query for record in self.records}) != 1
        ):
            raise LegacyDcRsRegistryError()
        return self


class LegacyDcRsRegistry(_FrozenModel):
    schema_version: Literal[
        "phase13_legacy_dc_rs_intervention_registry_v1",
        "phase13_legacy_dc_rs_intervention_registry_v2",
    ]
    registry_id: Literal[
        "phase13-main-a-legacy-dc-rs-interventions-v1",
        "phase13-main-a-legacy-dc-rs-interventions-v2",
    ]
    renderer_id: Literal[
        "legacy-dc-rs-canonical-renderer-v1",
        "legacy-dc-rs-canonical-renderer-v2",
    ]
    serialization_id: Literal["dc-rs-input-then-response-json-v1"]
    candidate_registry_sha256: str
    construction_provenance: Literal[
        "controlled_protocol_intervention_a03_v1",
        "controlled_protocol_intervention_a03_v2",
    ]
    frozen_at: Literal["2026-09-13T00:00:00Z", "2026-09-23T00:00:00Z"]
    authority_bindings: tuple[AuthorityBinding, AuthorityBinding, AuthorityBinding]
    tasks: tuple[LegacyDcRsTaskRecords, LegacyDcRsTaskRecords, LegacyDcRsTaskRecords]
    registry_sha256: str

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        if (
            {task.task for task in self.tasks} != _TASKS
            or len({task.triplet_id for task in self.tasks}) != 3
            or self.registry_sha256 != _registry_hash(self)
        ):
            raise LegacyDcRsRegistryError()
        return self

    def render(self, task: LegacyTask, candidate: CandidateVariant) -> NativeEntry:
        task_records = next((row for row in self.tasks if row.task == task), None)
        record = None if task_records is None else next(
            (row for row in task_records.records if row.candidate_id == candidate.candidate_id),
            None,
        )
        if (
            record is None
            or record.role != candidate.role
            or record.response != candidate.content
        ):
            raise LegacyDcRsRegistryError()
        content = record.serialized_content
        return NativeEntry(
            entry_id=candidate.candidate_id,
            semantic_kind="dc_rs_io_pair",
            schema_version=NATIVE_ENTRY_V1,
            native_component="archive",
            content=content,
            content_hash=canonical_content_hash(content),
            render_id=record.render_id,
        )


def load_legacy_dc_rs_registry(
    raw: bytes,
    candidates: CandidateRegistry,
    candidate_registry_sha256: str,
) -> LegacyDcRsRegistry:
    if hashlib.sha256(raw).hexdigest() not in _REGISTRY_SHA256S:
        raise LegacyDcRsRegistryError()
    try:
        registry = LegacyDcRsRegistry.model_validate_json(raw)
    except (ValidationError, LegacyDcRsRegistryError) as error:
        raise LegacyDcRsRegistryError() from error
    if (
        raw != _canonical_bytes(registry)
        or registry.candidate_registry_sha256 != candidate_registry_sha256
    ):
        raise LegacyDcRsRegistryError()
    triplets = {triplet.task: triplet for triplet in candidates.triplets}
    for task_records in registry.tasks:
        triplet = triplets.get(task_records.task)
        if triplet is None or triplet.triplet_id != task_records.triplet_id:
            raise LegacyDcRsRegistryError()
        variants = (triplet.false_candidate, triplet.correct_twin, triplet.irrelevant_control)
        if {record.candidate_id for record in task_records.records} != {
            variant.candidate_id for variant in variants
        }:
            raise LegacyDcRsRegistryError()
        for variant in variants:
            registry.render(task_records.task, variant)
    return registry


def _canonical_bytes(registry: LegacyDcRsRegistry) -> bytes:
    return (
        json.dumps(
            registry.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _registry_hash(registry: LegacyDcRsRegistry) -> str:
    payload = registry.model_dump(mode="json", exclude={"registry_sha256"})
    raw = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = ["LegacyDcRsRegistry", "LegacyDcRsRegistryError", "load_legacy_dc_rs_registry"]
