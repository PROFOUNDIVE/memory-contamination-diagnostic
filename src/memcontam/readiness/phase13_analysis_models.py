from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Baseline = Literal["fh_bounded", "rag_frozen", "bot_style", "reflexion_style"]
Task = Literal["game24", "math_equation_balancer", "word_sorting"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExecutionAuthority(StrictModel):
    path: Literal["data/phase13/authority/execution_registry_v1.json"]
    file_sha256: Sha256
    registry_id: Literal["phase13-execution-registry-v1"]
    registry_hash: Sha256
    analysis_window_registry_id: Literal["phase13-analysis-window-registry-v1"]


class Level1(StrictModel):
    baseline: Baseline
    support_population_id: Identifier
    status: Literal["baseline_local"]


class Level2(StrictModel):
    pair_id: Literal["P01", "P02", "P03", "P04", "P05", "P06"]
    left_baseline: Baseline
    right_baseline: Baseline
    support_population_id: Identifier
    status: Literal["required_confirmatory", "planned_secondary"]
    route_gating: bool


class Level3(StrictModel):
    support_population_id: Identifier
    baselines: tuple[Baseline, ...]
    status: Literal["sensitivity_only"]
    route_gating: Literal[False]

    @field_validator("baselines", mode="before")
    @classmethod
    def _tuple_field(cls, value: list[str]) -> tuple[str, ...]:
        return tuple(value)


class SupportRegistry(StrictModel):
    level_1: tuple[Level1, ...]
    level_2: tuple[Level2, ...]
    level_3: Level3

    @field_validator("level_1", "level_2", mode="before")
    @classmethod
    def _tuple_fields(cls, value: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(value)


class Target(StrictModel):
    route: Literal["3w", "5w"]
    level_1: int
    level_2: int
    level_3: int | None


class DeterministicPlanning(StrictModel):
    classification: Literal["deterministic_structural"]
    required_state: Literal["conformance_passed"]
    planning_value: Literal["1.000"]


class StochasticPlanning(StrictModel):
    classification: Literal["trajectory_stochastic"]
    interval_id: Literal["support-planning-cp95-one-sided-v1"]
    method: Literal["clopper_pearson_exact_binomial"]
    side: Literal["lower_one_sided"]
    confidence: Literal["0.95"]
    rounding: Literal["floor_to_3_decimals"]
    zero_success_value: Literal["0.000"]
    denominator: Literal[12]


class PlanningRegistry(StrictModel):
    calibration_seeds: tuple[int, ...]
    targets: tuple[Target, ...]
    deterministic: DeterministicPlanning
    stochastic: StochasticPlanning
    main_outcomes_prohibited: Literal[True]

    @field_validator("calibration_seeds", "targets", mode="before")
    @classmethod
    def _tuple_fields(cls, value: list[object]) -> tuple[object, ...]:
        return tuple(value)


class Slot(StrictModel):
    order: int
    estimand_id: Identifier
    analysis_window_id: Literal["accuracy-h5-primary"]
    support_level: Literal["L1", "L2"]
    pair_id: Literal["P01", "P02", "P03"] | None


class Family(StrictModel):
    family_id: Identifier
    task: Task
    slots: tuple[Slot, ...]

    @field_validator("slots", mode="before")
    @classmethod
    def _tuple_field(cls, value: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(value)


class Bootstrap(StrictModel):
    unit: Literal["trajectory_seed"]
    dependence: Literal["joint_complete_seed_cluster"]
    interval: Literal["two_sided_95_percentile"]
    test: Literal["null_imposed_paired_cluster"]
    replicates: Literal[20000]
    rng_seed: Literal[13]


class Holm(StrictModel):
    method: Literal["holm_step_down"]
    alpha: Literal["0.05"]
    family_scope: Literal["within_task"]


class NotEstimable(StrictModel):
    sentinel: Literal["NOT_ESTIMABLE"]
    p_value: Literal["1.0"]
    reason_required: Literal[True]
    retain_family_slot: Literal[True]
    reject_null: Literal[False]
    shrink_family: Literal[False]
    renormalize_weights: Literal[False]


class InferenceRegistry(StrictModel):
    H_primary: Literal[5]
    seed_summary: Literal["mean_verified_accuracy_k0_k4"]
    level_1_estimator: Literal["paired_seed_risk_difference"]
    level_2_estimator: Literal["paired_seed_difference_in_differences"]
    interval_id: Literal["main-paired-seed-bootstrap95-v1"]
    interval_method: Literal["paired_seed_percentile_bootstrap"]
    bootstrap: Bootstrap
    holm: Holm
    families: tuple[Family, ...]
    cross_task_family: Literal[False]
    not_estimable: NotEstimable

    @field_validator("families", mode="before")
    @classmethod
    def _tuple_field(cls, value: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(value)


class NonPrimaryWindow(StrictModel):
    analysis_window_id: Identifier
    inference_status: Literal["estimation_only"]


class OfflineRow(StrictModel):
    operation: Literal["prefix_derivation", "paired_seed_bootstrap", "report_rendering"]
    owner_id: Literal["phase13-offline-compute-owner-v1"]
    provider_calls: Literal[0]
    task_presentations: Literal[0]
    memory_evolutions: Literal[0]


class OfflineCompute(StrictModel):
    owner_id: Literal["phase13-offline-compute-owner-v1"]
    rows: tuple[OfflineRow, ...]

    @field_validator("rows", mode="before")
    @classmethod
    def _tuple_field(cls, value: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(value)


class AnalysisRegistry(StrictModel):
    schema_version: Literal["phase13_analysis_registry_v1"]
    registry_id: Literal["phase13-analysis-registry-v1"]
    execution_authority: ExecutionAuthority
    support: SupportRegistry
    planning: PlanningRegistry
    inference: InferenceRegistry
    non_primary_windows: tuple[NonPrimaryWindow, ...]
    excluded_conditions: tuple[Literal["nomem", "filter_challenge"], ...]
    offline_compute: OfflineCompute
    registry_hash: Sha256

    @field_validator("non_primary_windows", "excluded_conditions", mode="before")
    @classmethod
    def _tuple_fields(cls, value: list[str] | list[dict[str, str]]) -> tuple[str | dict[str, str], ...]:
        return tuple(value)
