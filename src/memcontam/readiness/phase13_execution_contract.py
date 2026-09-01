from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority import (
    ArtifactRef,
    Identifier,
    Sha256,
    parse_authority_freeze,
    parse_authority_requirements,
)
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


@dataclass(frozen=True, slots=True)
class CoreMainRegistry:
    tasks: tuple[str, ...]
    task_extensions: tuple[tuple[str, str], ...]
    memory_baselines: tuple[str, ...]
    arms: tuple[str, ...]
    call_ceilings: tuple[tuple[str, int, int], ...]
    nomem_policy: str
    backbone_id: str
    H_run: int
    H_primary: int
    primary_analysis_window_id: str
    capacity_unit: str
    capacity_law_id: str
    capacity_formula: str
    dc_rs_capacity_binding: str
    writer_max_output_tokens: int
    attempted_seed_count: int
    seed_policy: str
    rag_deadline: str
    rag_cutoff_status: str
    current_main_excluded_cells: tuple[tuple[str, str], ...]
    prospective_rag_extension_id: str
    authority_router_sha256: str
    post_cutoff_addendum_path: str
    post_cutoff_rule_id: str
    current_main_package_id: str
    authority_stack: tuple[tuple[str, str], ...]


CORE_MAIN_REGISTRY = CoreMainRegistry(
    tasks=(
        "game24",
        "math_equation_balancer",
        "word_sorting",
        "mmlu_pro_engineering",
        "mmlu_pro_physics",
    ),
    task_extensions=(
        (
            "gpqa_diamond",
            "DEFERRED_TO_PROSPECTIVE_EXTENSION -- "
            "CORE_CONTAMINATION_CANDIDATE_FAMILY_NOT_READY",
        ),
    ),
    memory_baselines=(
        "fh_bounded",
        "rag_frozen",
        "bot_style",
        "reflexion_style",
        "dc_rs",
    ),
    arms=("clean", "correct", "irrelevant", "contam"),
    call_ceilings=(
        ("fh_bounded", 1, 1),
        ("rag_frozen", 1, 1),
        ("bot_style", 3, 3),
        ("reflexion_style", 1, 4),
        ("dc_rs", 2, 2),
        ("nomem", 1, 1),
    ),
    nomem_policy="singleton_per_task_seed",
    backbone_id="gpt-5.6-luna",
    H_run=50,
    H_primary=50,
    primary_analysis_window_id="core_prefix_50",
    capacity_unit="registered_serialized_tokens",
    capacity_law_id="luna_common_visible_memory_capacity_v1",
    capacity_formula="min(B_FH_feasible,B_DC_feasible)",
    dc_rs_capacity_binding="L_DC_tokens=B_mem_tokens",
    writer_max_output_tokens=8192,
    attempted_seed_count=10,
    seed_policy="exactly_10_no_replacement_no_adaptive_augmentation",
    rag_deadline="2026-08-22T18:00:00+09:00",
    rag_cutoff_status="CONTINGENCY_FIRED",
    current_main_excluded_cells=(
        ("mmlu_pro_engineering", "rag_frozen"),
        ("mmlu_pro_physics", "rag_frozen"),
    ),
    prospective_rag_extension_id="new_mcq_rag_prospective_extension_v1",
    authority_router_sha256="76af55a04ce47f5bfe82795a99b0654b509becc7e202b3e05db29550297cf430",
    post_cutoff_addendum_path="2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum.md",
    post_cutoff_rule_id="experiment_v8_two_retained_new_mcq_rag_deadline_contingency",
    current_main_package_id="phase13_main_a_post_cutoff_partial_crossed_v1",
    authority_stack=(
        (
            "theory",
            "34f63f37a49e92607c78ced038c4c70b4c9d5e3fa8fc57d6e97de1ee79db59a8",
        ),
        (
            "baseline",
            "0bacce62718a93c14ce4da0c1b426e3823b75cf70b362f8f9a0632e83f4166c1",
        ),
        (
            "protocol",
            "022879f559b145e30e645b6ccbd139e9927899d370f1956d27a0562580acf85f",
        ),
        (
            "post_cutoff_addendum",
            "6ab2db4041333a26e9c76a034266c2d7de64491c5c109634c97ecebef9b2f59b",
        ),
        (
            "experiment_design",
            "174002b407c2c4055c42298bd16ca13463e80af5c935f9593f556c3f71d7671c",
        ),
    ),
)


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

    @model_validator(mode="after")
    def _ordered_prefix_limits(self) -> CapacityContract:
        if self.prefix_nominal_calls_per_seed > self.prefix_maximum_calls_per_seed:
            raise Phase13ExecutionError("PREFIX_CALL_LIMIT_INVALID")
        return self


class ExecutionRegistry(_StrictModel):
    schema_version: Annotated[str, Field(pattern=r"^phase13_execution_registry_v[0-9]+$")]
    registry_id: Identifier
    authority_freeze_id: Identifier
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
            "PREFIX_CALL_LIMIT_INVALID",
            "EXECUTION_DIMENSION_INVALID",
            "DUPLICATE_EXECUTION_TEMPLATE",
            "REGISTRY_HASH_MISMATCH",
        ):
            if code in message:
                raise Phase13ExecutionError(code) from error
        raise Phase13ExecutionError("MALFORMED_EXECUTION_REGISTRY") from error


def validate_execution_closure(
    freeze_json: bytes | str,
    requirements_json: bytes | str,
    root: Path,
) -> ExecutionRegistry:
    requirements = parse_authority_requirements(requirements_json)
    freeze = parse_authority_freeze(freeze_json, requirements)
    references = tuple(row for row in freeze.registries if row.kind == "execution")
    if len(references) != 1:
        raise Phase13ExecutionError("EXECUTION_AUTHORITY_MISSING")
    reference = references[0]
    try:
        registry_json = read_regular_nofollow(root / reference.artifact.path)
    except AuthorityFileError as error:
        raise Phase13ExecutionError(error.code) from error
    if hashlib.sha256(registry_json).hexdigest() != reference.artifact.sha256:
        raise Phase13ExecutionError("EXECUTION_AUTHORITY_HASH_MISMATCH")
    registry = parse_execution_registry(registry_json)
    if (
        registry.registry_id != reference.registry_id
        or registry.authority_freeze_id != freeze.freeze_id
    ):
        raise Phase13ExecutionError("EXECUTION_AUTHORITY_IDENTITY_MISMATCH")
    return registry
