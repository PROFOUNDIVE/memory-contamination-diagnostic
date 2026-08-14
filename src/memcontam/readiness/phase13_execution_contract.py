from __future__ import annotations

import hashlib
import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority import ArtifactRef, Identifier, Sha256


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class Phase13ExecutionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExecutionTemplate(_StrictModel):
    template_id: Identifier
    task: Identifier
    baseline: Identifier
    arm: Identifier
    nominal_semantic_calls_per_trial: NonNegativeInt
    maximum_semantic_calls_per_trial: NonNegativeInt

    @model_validator(mode="after")
    def _ordered_call_limits(self) -> ExecutionTemplate:
        if self.nominal_semantic_calls_per_trial > self.maximum_semantic_calls_per_trial:
            raise Phase13ExecutionError("TEMPLATE_CALL_LIMIT_INVALID")
        return self


class CapacityContract(_StrictModel):
    prefix_nominal_calls_per_seed: NonNegativeInt
    prefix_maximum_calls_per_seed: NonNegativeInt
    reserve_percent: NonNegativeInt
    maximum_transport_attempts_per_semantic_call: PositiveInt
    maximum_input_tokens_per_transport_attempt: PositiveInt
    maximum_output_tokens_per_transport_attempt: PositiveInt


class ExecutionRegistry(_StrictModel):
    schema_version: Annotated[str, Field(pattern=r"^phase13_execution_registry_v[0-9]+$")]
    registry_id: Identifier
    authority_freeze_sha256: Sha256
    backbone_id: Identifier
    H_run: PositiveInt
    tasks: tuple[Identifier, ...]
    baselines: tuple[Identifier, ...]
    arms: tuple[Identifier, ...]
    rag_corpus: ArtifactRef
    execution_owner_id: Identifier
    templates: tuple[ExecutionTemplate, ...]
    capacity: CapacityContract
    registry_hash: Sha256

    @field_validator("tasks", "baselines", "arms", mode="before")
    @classmethod
    def _identifiers(cls, value: list[str]) -> tuple[str, ...]:
        return tuple(value)

    @field_validator("templates", mode="before")
    @classmethod
    def _templates(cls, value: list[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
        return tuple(value)

    @model_validator(mode="after")
    def _declared_dimensions(self) -> ExecutionRegistry:
        for values in (self.tasks, self.baselines, self.arms):
            if not values or len(values) != len(set(values)):
                raise Phase13ExecutionError("EXECUTION_DIMENSION_INVALID")
        ids = tuple(row.template_id for row in self.templates)
        keys = tuple((row.task, row.baseline, row.arm) for row in self.templates)
        if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
            raise Phase13ExecutionError("DUPLICATE_EXECUTION_TEMPLATE")
        if any(
            row.task not in self.tasks
            or row.baseline not in self.baselines
            or row.arm not in self.arms
            for row in self.templates
        ):
            raise Phase13ExecutionError("TEMPLATE_DIMENSION_UNDECLARED")
        payload = self.model_dump(mode="json", exclude={"registry_hash"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest() != self.registry_hash:
            raise Phase13ExecutionError("REGISTRY_HASH_MISMATCH")
        return self


def parse_execution_registry(raw_json: bytes | str) -> ExecutionRegistry:
    try:
        return ExecutionRegistry.model_validate_json(raw_json)
    except ValidationError as error:
        message = str(error)
        for code in (
            "TEMPLATE_DIMENSION_UNDECLARED",
            "TEMPLATE_CALL_LIMIT_INVALID",
            "EXECUTION_DIMENSION_INVALID",
            "DUPLICATE_EXECUTION_TEMPLATE",
            "REGISTRY_HASH_MISMATCH",
        ):
            if code in message:
                raise Phase13ExecutionError(code) from error
        raise Phase13ExecutionError("MALFORMED_EXECUTION_REGISTRY") from error
