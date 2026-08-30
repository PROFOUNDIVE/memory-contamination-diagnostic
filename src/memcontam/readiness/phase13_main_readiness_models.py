from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Phase13MainReadinessError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactIdentity(_FrozenModel):
    path: str = Field(min_length=1)
    sha256: Sha256


class CallCeiling(_FrozenModel):
    nominal: int = Field(ge=0)
    maximum: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> CallCeiling:
        if self.nominal > self.maximum:
            raise Phase13MainReadinessError("MR_P4_CALL_CEILING_INVALID")
        return self


class ExecutionTemplates(_FrozenModel):
    tasks: tuple[str, ...]
    memory_baselines: tuple[str, ...]
    arms: tuple[str, ...]
    included_task_baseline_pairs: tuple[tuple[str, str], ...]
    nomem_tasks: tuple[str, ...]
    abstract_seed_slots_per_task: Literal[10]
    concrete_seed_registry_status: Literal["CONCRETE_MAIN_SEED_REGISTRY_FROZEN"]
    H_run: Literal[50]
    H_primary: Literal[50]
    primary_analysis_window_id: Literal["core_prefix_50"]
    tool_mode: Literal["text_only"]
    call_ceilings: dict[str, CallCeiling]

    @field_validator("tasks", "memory_baselines", "arms", "nomem_tasks", mode="before")
    @classmethod
    def _string_tuples(cls, value: list[str]) -> tuple[str, ...]:
        return tuple(value)

    @field_validator("included_task_baseline_pairs", mode="before")
    @classmethod
    def _pair_tuples(cls, value: list[list[str]]) -> tuple[tuple[str, str], ...]:
        return tuple((row[0], row[1]) for row in value)


class Level2Registry(_FrozenModel):
    anchor: Literal["fh_bounded"]
    contrast: Literal["difference_of_clean_minus_contam"]
    support: Literal["task_local_pairwise_common_structural_support"]
    multiplicity: Literal["task_local_holm_fwer"]
    alpha: float
    comparators_by_task: dict[str, tuple[str, ...]]

    @field_validator("comparators_by_task", mode="before")
    @classmethod
    def _comparators(cls, value: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
        return {task: tuple(comparators) for task, comparators in value.items()}


class ProviderRuntimeContract(_FrozenModel):
    api: Literal["OpenAI Responses API"]
    model: Literal["gpt-5.6-luna"]
    service_tier: Literal["default"]
    reasoning_mode: Literal["standard"]
    reasoning_effort: Literal["none"]
    reasoning_context: Literal["current_turn"]
    previous_response_id: None
    store: Literal[False]
    tools: tuple[()] = ()
    timeout_seconds: Literal[180]
    retries_after_initial_attempt: int = Field(ge=0, le=2)
    semantic_invalid_generic_retry: Literal[False]
    session_isolation: Literal["independent_per_trial_and_arm"]


class GateStates(_FrozenModel):
    synthetic_observability_conformance_status: Literal["PASS"]
    provider_session_retry_resource_contract_status: Literal["PASS"]
    static_readiness0_status: Literal["PASS"]
    tau_star_status: Literal["PASS"]
    cost_feasibility_status: Literal["PASS"]
    live_api_status: Literal["PASS"]


class MainReadinessManifest(_FrozenModel):
    schema_version: Literal["phase13_mr_p4_local_closure_manifest_v1"]
    status: Literal["MR_P4_CLOSED"]
    artifacts: dict[str, ArtifactIdentity]
    execution_templates: ExecutionTemplates
    level2_interactions: Level2Registry
    provider_runtime_contract: ProviderRuntimeContract
    gates: GateStates
    u_t_status: Literal["NOT_REGISTERED_FOR_CURRENT_MAIN"]
    mr_p4_closure_claimed: Literal[True]
    mr_p5_status: Literal["NOT_STARTED"]
    mr_p6_status: Literal["NOT_AUTHORIZED"]
    main_execution_authorized: Literal[False]
    main_a_measured_scientific_execution_count: Literal[0]
    closure_hash: Sha256

    @model_validator(mode="after")
    def _closure(self) -> MainReadinessManifest:
        _validate_execution_templates(self.execution_templates)
        _validate_level2(self.level2_interactions)
        payload = self.model_dump(mode="json", exclude={"closure_hash"})
        if _canonical_hash(payload) != self.closure_hash:
            raise Phase13MainReadinessError("MR_P4_CLOSURE_HASH_MISMATCH")
        return self


class MainReadinessReport(_FrozenModel):
    status: str
    manifest_sha256: Sha256
    execution_template_count: int
    level2_interaction_count: int
    abstract_seed_slots_per_task: int
    H_run: int
    H_primary: int
    synthetic_observability_conformance_status: str
    provider_session_retry_resource_contract_status: str
    u_t_status: str
    blockers: tuple[str, ...]
    f1c_status: str
    provider_calls_issued: int
    output_directory_created: bool
    scientific_result: bool
    main_result: bool
    mr_p4_status: str
    mr_p4_closure_claimed: bool
    mr_p5_status: str
    mr_p6_status: str
    main_a_status: str
    main_execution_authorized: bool
    main_a_measured_scientific_execution_count: int


def _validate_execution_templates(execution: ExecutionTemplates) -> None:
    expected_pairs = tuple(
        (task, baseline)
        for task in CORE_MAIN_REGISTRY.tasks
        for baseline in CORE_MAIN_REGISTRY.memory_baselines
        if (task, baseline) not in CORE_MAIN_REGISTRY.current_main_excluded_cells
    )
    expected_call_ceilings = {
        baseline: CallCeiling(nominal=nominal, maximum=maximum)
        for baseline, nominal, maximum in CORE_MAIN_REGISTRY.call_ceilings
    }
    if (
        execution.tasks != CORE_MAIN_REGISTRY.tasks
        or execution.memory_baselines != CORE_MAIN_REGISTRY.memory_baselines
        or execution.arms != CORE_MAIN_REGISTRY.arms
        or execution.included_task_baseline_pairs != expected_pairs
        or execution.nomem_tasks != CORE_MAIN_REGISTRY.tasks
        or execution.call_ceilings != expected_call_ceilings
    ):
        raise Phase13MainReadinessError("MR_P4_EXECUTION_CONTRACT_MISMATCH")


def _validate_level2(registry: Level2Registry) -> None:
    expected = {
        task: tuple(
            baseline
            for baseline in ("rag_frozen", "bot_style", "reflexion_style", "dc_rs")
            if (task, baseline) not in CORE_MAIN_REGISTRY.current_main_excluded_cells
        )
        for task in CORE_MAIN_REGISTRY.tasks
    }
    if registry.comparators_by_task != expected or registry.alpha != 0.05:
        raise Phase13MainReadinessError("MR_P4_LEVEL2_REGISTRY_MISMATCH")


def _canonical_hash(value: dict[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "ArtifactIdentity",
    "MainReadinessManifest",
    "MainReadinessReport",
    "Phase13MainReadinessError",
]
