from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Baseline = Literal["fh_bounded", "rag_frozen", "bot_style", "reflexion_style"]
Task = Literal["game24", "math_equation_balancer", "word_sorting"]
EXECUTION_PATH: Final = "data/phase13/authority/execution_registry_v1.json"
EXECUTION_FILE_SHA256: Final = "7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970"
EXECUTION_REGISTRY_HASH: Final = "acb769e1e1adbc3eb69e4302322c8eac81829dc836611519caea2ba960900c38"
TASKS: Final = ("game24", "math_equation_balancer", "word_sorting")
BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
PAIRS: Final = (
    ("P01", "fh_bounded", "rag_frozen", "required_confirmatory", True),
    ("P02", "fh_bounded", "bot_style", "required_confirmatory", True),
    ("P03", "fh_bounded", "reflexion_style", "required_confirmatory", True),
    ("P04", "rag_frozen", "bot_style", "planned_secondary", False),
    ("P05", "rag_frozen", "reflexion_style", "planned_secondary", False),
    ("P06", "bot_style", "reflexion_style", "planned_secondary", False),
)
WINDOWS: Final = (
    "accuracy-h2-sensitivity", "recurrence-h2-descriptive", "recurrence-h5-secondary",
    "persistence-h5-secondary", "propagation-h5-conditional", "collapse-h5-exploratory",
    "accuracy-h10-sensitivity", "recurrence-h10-descriptive", "persistence-h10-descriptive",
    "propagation-h10-conditional", "collapse-h10-exploratory",
)


class Phase13AnalysisError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


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
    interval_id: Identifier
    side: Literal["lower_one_sided"]
    confidence: Literal["0.95"]
    rounding: Literal["floor_to_3_decimals"]
    zero_success_value: Literal["0.000"]
    denominator: int


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
    replicates: int
    rng_seed: int


class Holm(StrictModel):
    method: Literal["holm_step_down"]
    alpha: Literal["0.05"]
    family_scope: Literal["within_task"]


class NotEstimable(StrictModel):
    sentinel: Literal["NOT_ESTIMABLE"]
    retain_family_slot: Literal[True]
    reject_null: Literal[False]
    shrink_family: Literal[False]
    renormalize_weights: Literal[False]


class InferenceRegistry(StrictModel):
    H_primary: Literal[5]
    seed_summary: Literal["mean_verified_accuracy_k0_k4"]
    level_1_estimator: Literal["paired_seed_risk_difference"]
    level_2_estimator: Literal["paired_seed_difference_in_differences"]
    interval_id: Identifier
    bootstrap: Bootstrap
    holm: Holm
    families: tuple[Family, ...]
    cross_task_family: bool
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
    owner_id: Identifier
    provider_calls: int
    task_presentations: int
    memory_evolutions: int


class OfflineCompute(StrictModel):
    owner_id: Identifier
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

    @field_validator(
        "non_primary_windows", "excluded_conditions", mode="before",
    )
    @classmethod
    def _tuple_fields(cls, value: list[str] | list[dict[str, str]]) -> tuple[str | dict[str, str], ...]:
        return tuple(value)


def _canonical_hash(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("registry_hash", None)
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _expected_slots() -> tuple[tuple[int, str, str, str | None], ...]:
    return tuple(
        (index, estimand, "L1" if index <= 4 else "L2", None if index <= 4 else f"P0{index - 4}")
        for index, estimand in enumerate(
            (*[f"l1-{baseline}-clean-contam" for baseline in BASELINES],
             *[f"l2-p0{index}-clean-contam-did" for index in range(1, 4)]), start=1
        )
    )


def _validate_execution(registry: AnalysisRegistry, root: Path) -> None:
    reference = registry.execution_authority
    if (reference.file_sha256, reference.registry_hash) != (EXECUTION_FILE_SHA256, EXECUTION_REGISTRY_HASH):
        raise Phase13AnalysisError("EXECUTION_AUTHORITY_MISMATCH")
    try:
        raw = read_regular_nofollow(root / EXECUTION_PATH)
    except AuthorityFileError as error:
        raise Phase13AnalysisError(str(error)) from error
    if hashlib.sha256(raw).hexdigest() != EXECUTION_FILE_SHA256:
        raise Phase13AnalysisError("EXECUTION_AUTHORITY_MISMATCH")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or (
        payload.get("registry_hash"), payload.get("registry_id"),
        payload.get("analysis_window_registry_id"),
    ) != (EXECUTION_REGISTRY_HASH, reference.registry_id, reference.analysis_window_registry_id):
        raise Phase13AnalysisError("EXECUTION_AUTHORITY_MISMATCH")


def _validate_semantics(registry: AnalysisRegistry) -> None:
    l1 = tuple((row.baseline, row.support_population_id, row.status) for row in registry.support.level_1)
    l2 = tuple((row.pair_id, row.left_baseline, row.right_baseline, row.status, row.route_gating) for row in registry.support.level_2)
    families = tuple((family.task, family.family_id, tuple((slot.order, slot.estimand_id, slot.support_level, slot.pair_id) for slot in family.slots)) for family in registry.inference.families)
    expected_families = tuple((task, f"{task}-h5-primary-holm-v1", _expected_slots()) for task in TASKS)
    if (
        l1 != tuple((baseline, f"l1-{baseline}-structural-support", "baseline_local") for baseline in BASELINES)
        or l2 != PAIRS
        or (registry.support.level_3.support_population_id, registry.support.level_3.baselines) != ("l3-all-primary-baselines-structural-support", BASELINES)
        or tuple((row.route, row.level_1, row.level_2, row.level_3) for row in registry.planning.targets) != (("3w", 10, 10, None), ("5w", 16, 16, 8))
        or registry.planning.calibration_seeds != tuple(range(10000, 10012))
        or registry.planning.stochastic.denominator != 12
        or registry.planning.stochastic.interval_id == registry.inference.interval_id
        or (registry.inference.bootstrap.replicates, registry.inference.bootstrap.rng_seed) != (20_000, 13)
        or registry.inference.cross_task_family
        or families != expected_families
        or tuple(row.analysis_window_id for row in registry.non_primary_windows) != WINDOWS
        or registry.excluded_conditions != ("nomem", "filter_challenge")
        or tuple(row.operation for row in registry.offline_compute.rows) != ("prefix_derivation", "paired_seed_bootstrap", "report_rendering")
        or any(row.owner_id != registry.offline_compute.owner_id or row.provider_calls or row.task_presentations or row.memory_evolutions for row in registry.offline_compute.rows)
    ):
        raise Phase13AnalysisError("ANALYSIS_SEMANTICS_INVALID")


def parse_analysis_registry(raw: bytes, root: Path) -> AnalysisRegistry:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise Phase13AnalysisError("MALFORMED_REGISTRY")
        registry = AnalysisRegistry.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise Phase13AnalysisError("ANALYSIS_SEMANTICS_INVALID") from error
    if _canonical_hash(payload) != registry.registry_hash:
        raise Phase13AnalysisError("REGISTRY_HASH_MISMATCH")
    _validate_execution(registry, root)
    _validate_semantics(registry)
    return registry


def load_analysis_registry(path: Path, root: Path) -> AnalysisRegistry:
    try:
        raw = read_regular_nofollow(path)
    except AuthorityFileError as error:
        raise Phase13AnalysisError(str(error)) from error
    return parse_analysis_registry(raw, root)
