from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path
from typing import Final, Literal, assert_never

from memcontam.manifests.phase13 import NotExchangeable, PrefixDerivationArtifact
from memcontam.readiness.phase13_analysis_contract import (
    Phase13AnalysisError,
    load_analysis_registry,
)
from memcontam.readiness.phase13_execution_contract import (
    Phase13ExecutionError,
    load_execution_registry,
)
from memcontam.readiness.phase13_calibration_v2_runtime_models import (
    CompletedTrajectory,
    TrajectoryRequest,
)
from memcontam.readiness.phase13_route_capacity import (
    CapacityPlan,
    CapacityPlanningError,
    recompute_capacity,
)
from memcontam.readiness.phase13_support_authority import (
    SupportAuthorityError,
    authenticate_conformance,
)
from memcontam.readiness.phase13_support_inputs import (
    SupportInputError,
    validate_route_support_inputs,
)


ANALYSIS_PATH: Final = Path("data/phase13/authority/analysis_registry_v1.json")
EXECUTION_PATH: Final = Path("data/phase13/authority/execution_registry_v1.json")
ANALYSIS_FILE_SHA256: Final = "b58e6aec8acc040fb934e9b25842eb68c702d098a08b41ba0eab9502a198a0f3"
EXECUTION_FILE_SHA256: Final = "7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970"
BISECTION_STEPS: Final = 96
FLOOR_QUANTUM: Final = Decimal("0.001")


class PlanningError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ResolvedSeed:
    seed_id: int
    passed: bool


@dataclass(frozen=True, slots=True)
class UnresolvedSeed:
    seed_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class AbsentSeed:
    seed_id: int


SeedRow = ResolvedSeed | UnresolvedSeed | AbsentSeed


@dataclass(frozen=True, slots=True)
class DeterministicSupportInput:
    task: str
    support_population_id: str
    certificate: PrefixDerivationArtifact | NotExchangeable
    authority_request: TrajectoryRequest | None = None
    authority_source: CompletedTrajectory | None = None


@dataclass(frozen=True, slots=True)
class StochasticSupportInput:
    task: str
    support_population_id: str
    seeds: tuple[SeedRow, ...]
    method: str = "clopper_pearson_exact_binomial"
    rounding: str = "floor_to_3_decimals"
    denominator: int = 12


SupportInput = DeterministicSupportInput | StochasticSupportInput


@dataclass(frozen=True, slots=True)
class SupportPlan:
    task: str
    support_population_id: str
    method: Literal["deterministic_structural", "clopper_pearson_exact_binomial"]
    successes: int
    denominator: int
    planning_value: Decimal


@dataclass(frozen=True, slots=True)
class RoutePlanningRequest:
    route: Literal["3w", "5w"]
    support_inputs: tuple[StochasticSupportInput, ...]
    historical_route_calls: int | None = None
    budget_layer: Literal["phase13", "phase12"] = "phase13"


@dataclass(frozen=True, slots=True)
class RoutePlan:
    route: Literal["3w", "5w"]
    attempted_seeds: tuple[tuple[str, int], ...]
    support: tuple[SupportPlan, ...]
    informational_population_ids: tuple[str, ...]
    capacity: CapacityPlan


def _survival(successes: int, trials: int, probability: Decimal) -> Decimal:
    return sum(
        (
            Decimal(math.comb(trials, count))
            * probability**count
            * (Decimal(1) - probability) ** (trials - count)
            for count in range(successes, trials + 1)
        ),
        start=Decimal(0),
    )


def clopper_pearson_lower(successes: int, trials: int) -> Decimal:
    if trials != 12:
        raise PlanningError("DENOMINATOR_INVALID")
    if successes < 0 or successes > trials:
        raise PlanningError("BINOMIAL_COUNT_INVALID")
    if successes == 0:
        return Decimal("0.000")
    with localcontext() as context:
        context.prec = 60
        lower, upper = Decimal(0), Decimal(1)
        for _ in range(BISECTION_STEPS):
            midpoint = (lower + upper) / 2
            if _survival(successes, trials, midpoint) <= Decimal("0.05"):
                lower = midpoint
            else:
                upper = midpoint
        return lower.quantize(FLOOR_QUANTUM, rounding=ROUND_FLOOR)


def _authorities(root: Path):  # noqa: ANN202
    if hashlib.sha256((root / ANALYSIS_PATH).read_bytes()).hexdigest() != ANALYSIS_FILE_SHA256:
        raise PlanningError("ANALYSIS_AUTHORITY_INVALID")
    if hashlib.sha256((root / EXECUTION_PATH).read_bytes()).hexdigest() != EXECUTION_FILE_SHA256:
        raise PlanningError("CAPACITY_AUTHORITY_INVALID")
    try:
        analysis = load_analysis_registry(root / ANALYSIS_PATH, root)
        execution = load_execution_registry(root / EXECUTION_PATH, root)
    except Phase13AnalysisError as error:
        raise PlanningError("ANALYSIS_AUTHORITY_INVALID") from error
    except Phase13ExecutionError as error:
        raise PlanningError("CAPACITY_AUTHORITY_INVALID") from error
    return analysis, execution


def _registered_populations(analysis) -> set[str]:  # noqa: ANN001
    return {
        *(row.support_population_id for row in analysis.support.level_1),
        *(row.support_population_id for row in analysis.support.level_2),
        analysis.support.level_3.support_population_id,
    }


def _stochastic_plan(evidence: StochasticSupportInput) -> SupportPlan:
    if evidence.method != "clopper_pearson_exact_binomial":
        raise PlanningError("POINT_ESTIMATE_FORBIDDEN")
    if evidence.rounding != "floor_to_3_decimals":
        raise PlanningError("ROUNDING_MODE_INVALID")
    if evidence.denominator != 12:
        raise PlanningError("DENOMINATOR_INVALID")
    if len(evidence.seeds) != 12:
        raise PlanningError("FINAL_SEED_ROW_COUNT_INVALID")
    if tuple(row.seed_id for row in evidence.seeds) != tuple(range(10000, 10012)):
        raise PlanningError("FINAL_SEED_IDENTITY_INVALID")
    resolved: list[ResolvedSeed] = []
    for row in evidence.seeds:
        match row:
            case ResolvedSeed():
                resolved.append(row)
            case UnresolvedSeed():
                raise PlanningError("SEED_UNRESOLVED")
            case AbsentSeed():
                raise PlanningError("SEED_ABSENT")
            case unreachable:
                assert_never(unreachable)
    successes = sum(row.passed for row in resolved)
    return SupportPlan(
        evidence.task,
        evidence.support_population_id,
        "clopper_pearson_exact_binomial",
        successes,
        12,
        clopper_pearson_lower(successes, 12),
    )


def plan_support(evidence: SupportInput, root: Path) -> SupportPlan:
    analysis, _ = _authorities(root)
    if evidence.support_population_id not in _registered_populations(analysis):
        raise PlanningError("SUPPORT_POPULATION_UNREGISTERED")
    match evidence:
        case DeterministicSupportInput():
            match evidence.certificate:
                case PrefixDerivationArtifact() as certificate:
                    try:
                        authenticate_conformance(
                            certificate,
                            evidence.authority_request,
                            evidence.authority_source,
                        )
                    except SupportAuthorityError as error:
                        raise PlanningError(error.code) from error
                    return SupportPlan(
                        evidence.task, evidence.support_population_id,
                        "deterministic_structural", 12, 12, Decimal("1.000"),
                    )
                case NotExchangeable():
                    raise PlanningError("CONFORMANCE_NOT_PASSED")
                case unreachable:
                    assert_never(unreachable)
        case StochasticSupportInput():
            return _stochastic_plan(evidence)
        case unreachable:
            assert_never(unreachable)


def plan_route(request: RoutePlanningRequest, root: Path) -> RoutePlan:
    if request.historical_route_calls is not None:
        raise PlanningError("HISTORICAL_ROUTE_MERGE_FORBIDDEN")
    if request.budget_layer != "phase13":
        raise PlanningError("BUDGET_LAYER_INVALID")
    analysis, execution = _authorities(root)
    try:
        validate_route_support_inputs(request.support_inputs, analysis)
    except SupportInputError as error:
        raise PlanningError(error.code) from error
    target = next(row for row in analysis.planning.targets if row.route == request.route)
    required = {
        *(row.support_population_id for row in analysis.support.level_1),
        *(row.support_population_id for row in analysis.support.level_2 if row.route_gating),
    }
    plans = tuple(_stochastic_plan(item) for item in request.support_inputs)
    by_key = {(row.task, row.support_population_id): row for row in plans}
    attempted: list[tuple[str, int]] = []
    for task in ("game24", "math_equation_balancer", "word_sorting"):
        task_rows = tuple(by_key.get((task, population)) for population in required)
        if any(row is None for row in task_rows):
            raise PlanningError("REQUIRED_SUPPORT_MISSING")
        values = tuple(row.planning_value for row in task_rows if row is not None)
        if any(value == 0 for value in values):
            raise PlanningError("REQUIRED_SUPPORT_ZERO")
        ceiling = max(target.level_1, target.level_2)
        attempted.append((task, max(math.ceil(Decimal(ceiling) / value) for value in values)))
    informational = tuple(
        row.support_population_id for row in plans if row.support_population_id not in required
    )
    try:
        capacity = recompute_capacity(execution, tuple(attempted))
    except CapacityPlanningError as error:
        raise PlanningError(error.code) from error
    return RoutePlan(request.route, tuple(attempted), plans, informational, capacity)


__all__ = (
    "AbsentSeed", "DeterministicSupportInput", "PlanningError",
    "ResolvedSeed", "RoutePlan", "RoutePlanningRequest", "StochasticSupportInput",
    "UnresolvedSeed", "clopper_pearson_lower", "plan_route", "plan_support",
)
