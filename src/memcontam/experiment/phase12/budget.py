from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Literal, Mapping

from memcontam.experiment.phase12.contracts import (
    PrefixTemplateSpec,
    RunTemplateSpec,
    canonical_json_hash,
)


class PlanningError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CallCostRegistry:
    registry_id: str
    required_call_keys: tuple[str, ...]
    call_costs: Mapping[str, int]
    rerun_allowance_numerator: int
    rerun_allowance_denominator: int
    artifact_hash: str

    def __post_init__(self) -> None:
        if (
            not self.required_call_keys
            or len(set(self.required_call_keys)) != len(self.required_call_keys)
            or set(self.call_costs) != set(self.required_call_keys)
            or any(type(cost) is not int or cost < 0 for cost in self.call_costs.values())
            or self.rerun_allowance_numerator < 0
            or self.rerun_allowance_denominator <= 0
        ):
            raise PlanningError("INVALID_CALL_COST_REGISTRY")


@dataclass(frozen=True)
class ConditionalCallScope:
    scope_id: str
    model_snapshot: str
    task_family: str
    baseline_condition_id: str
    activation: Literal["on_attempt_1_failure", "on_retry_failure"]
    candidate_routes: tuple[Literal["3w", "5w"], ...]
    run_families: tuple[str, ...]
    template_domain_hash: str
    artifact_hash: str


@dataclass(frozen=True)
class ConditionalCallScopeRegistry:
    registry_id: str
    scopes: tuple[ConditionalCallScope, ...]
    frozen_before_main_unblinding: Literal[True]
    artifact_hash: str


@dataclass(frozen=True)
class PilotCallSufficientStatistic:
    statistic_id: str
    scope_id: str
    scope_hash: str
    eligible_opportunities: int
    activated_calls: int
    source_run_ids: tuple[str, ...]
    source_manifest_hash: str
    artifact_hash: str


@dataclass(frozen=True)
class PilotCallStatisticsManifest:
    manifest_id: str
    statistics: tuple[PilotCallSufficientStatistic, ...]
    frozen_at: str
    artifact_hash: str


@dataclass(frozen=True)
class ConservativeRateUpperBound:
    bound_id: str
    scope_id: str
    scope_hash: str
    numerator: int
    denominator: int
    rationale_code: str
    source_path: str
    source_hash: str
    artifact_hash: str


@dataclass(frozen=True)
class ConservativeRateUpperBoundRegistry:
    registry_id: str
    bounds: tuple[ConservativeRateUpperBound, ...]
    frozen_before_main_unblinding: Literal[True]
    artifact_hash: str


Activation = Literal["always", "once_per_seed", "on_attempt_1_failure", "on_retry_failure"]


@dataclass(frozen=True)
class CallActivationRate:
    rate_id: str
    scope_id_or_none: str | None
    scope_hash_or_none: str | None
    activation: Activation
    numerator: int
    denominator: int
    source_kind: Literal["deterministic", "pilot_b", "registered_upper_bound"]
    source_id: str
    source_hash: str

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class ConditionalCallRateRegistry:
    registry_id: str
    rates: tuple[CallActivationRate, ...]
    frozen_before_main_unblinding: Literal[True]
    artifact_hash: str


@dataclass(frozen=True)
class RequestedCountVector:
    core_counts: Mapping[str, int | Literal["not_feasible"]]
    extension_counts: Mapping[str, int | Literal["not_feasible"]]


@dataclass(frozen=True)
class RunTemplateCallComponent:
    component_id: str
    owner_kind: Literal["prefix", "run_template"]
    owner_id: str
    phase: Literal["burn", "init", "trial"]
    call_category: str
    calls_per_activation: int
    activation_rate_id: str


@dataclass(frozen=True)
class RunTemplateRegistry:
    registry_id: str
    artifact_hash: str
    repository_commit: str
    candidate_route: Literal["3w", "5w", "exploratory"]
    candidate_template_set_id: str
    candidate_template_set_hash: str
    template_package_hash: str
    conditional_call_scope_registry_id: str
    conditional_call_scope_registry_hash: str
    inv03_equivalence_registry_id: str
    inv03_equivalence_registry_hash: str
    component_policy_version: str
    run_templates: tuple[RunTemplateSpec, ...]
    prefix_templates: tuple[PrefixTemplateSpec, ...]
    abstract_slots: tuple[str, ...]
    call_components: tuple[RunTemplateCallComponent, ...]


@dataclass(frozen=True)
class RationalCallCount:
    numerator: int
    denominator: int

    @classmethod
    def from_fraction(cls, value: Fraction) -> RationalCallCount:
        return cls(value.numerator, value.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class CallBudgetBreakdown:
    breakdown_id: str
    artifact_hash: str
    run_template_registry_id: str
    run_template_registry_hash: str
    call_cost_registry_id: str
    call_cost_registry_hash: str
    conditional_call_rate_registry_id: str
    conditional_call_rate_registry_hash: str
    requested_core_counts: Mapping[str, int]
    requested_extension_counts: Mapping[str, int]
    run_template_contributions: Mapping[str, RationalCallCount]
    prefix_template_contributions: Mapping[str, RationalCallCount]
    expected_base_calls: RationalCallCount
    base_calls: int
    rerun_allowance_calls: int
    total_calls: int


def reserve_rational_calls(expected: Fraction, rerun_allowance: Fraction) -> CallBudgetBreakdown:
    if expected < 0 or rerun_allowance < 0:
        raise PlanningError("INVALID_CALL_BUDGET")
    base_calls = ceil(expected)
    total_calls = ceil(expected * (1 + rerun_allowance))
    return CallBudgetBreakdown(
        breakdown_id="reservation",
        artifact_hash="",
        run_template_registry_id="",
        run_template_registry_hash="",
        call_cost_registry_id="",
        call_cost_registry_hash="",
        conditional_call_rate_registry_id="",
        conditional_call_rate_registry_hash="",
        requested_core_counts={},
        requested_extension_counts={},
        run_template_contributions={},
        prefix_template_contributions={},
        expected_base_calls=RationalCallCount.from_fraction(expected),
        base_calls=base_calls,
        rerun_allowance_calls=total_calls - base_calls,
        total_calls=total_calls,
    )


def evaluate_route_call_budget(
    registry: RunTemplateRegistry,
    costs: CallCostRegistry,
    rates: ConditionalCallRateRegistry,
    counts: RequestedCountVector,
) -> CallBudgetBreakdown:
    _validate_rates(rates)
    rate_by_id = {rate.rate_id: rate for rate in rates.rates}
    components_by_owner: dict[str, list[RunTemplateCallComponent]] = {}
    for component in registry.call_components:
        if component.calls_per_activation < 0 or component.call_category not in costs.call_costs:
            raise PlanningError("INVALID_CALL_COMPONENT")
        if component.activation_rate_id not in rate_by_id:
            raise PlanningError("CONDITIONAL_CALL_RATE_REQUIRED")
        components_by_owner.setdefault(component.owner_id, []).append(component)

    run_contributions: dict[str, RationalCallCount] = {}
    prefix_contributions: dict[str, RationalCallCount] = {}
    expected = Fraction()
    clean_counts = _integer_counts(counts.core_counts, counts.extension_counts)
    for template in registry.run_templates:
        multiplicity = clean_counts.get(template.task_family, 0)
        component_cost = sum(
            (
                _component_cost(component, costs, rate_by_id)
                * (template.horizon if component.phase == "trial" else 1)
                for component in components_by_owner.get(template.run_template_id, [])
            ),
            Fraction(),
        )
        contribution = multiplicity * component_cost
        run_contributions[template.run_template_id] = RationalCallCount.from_fraction(
            Fraction(contribution)
        )
        expected += contribution
    for prefix in registry.prefix_templates:
        matching_counts = [
            clean_counts.get(template.task_family, 0)
            for template in registry.run_templates
            if template.prefix_template_key_or_none == prefix.prefix_template_key
        ]
        contribution = max(matching_counts, default=0) * sum(
            (
                _component_cost(component, costs, rate_by_id)
                for component in components_by_owner.get(prefix.prefix_template_key, [])
            ),
            Fraction(),
        )
        prefix_contributions[prefix.prefix_template_key] = RationalCallCount.from_fraction(
            Fraction(contribution)
        )
        expected += contribution
    reservation = reserve_rational_calls(
        expected,
        Fraction(costs.rerun_allowance_numerator, costs.rerun_allowance_denominator),
    )
    payload = {
        "registry": registry.registry_id,
        "costs": costs.registry_id,
        "rates": rates.registry_id,
        "expected": [expected.numerator, expected.denominator],
        "total": reservation.total_calls,
    }
    return CallBudgetBreakdown(
        breakdown_id=f"budget-{registry.candidate_route}",
        artifact_hash=canonical_json_hash(payload),
        run_template_registry_id=registry.registry_id,
        run_template_registry_hash=registry.artifact_hash,
        call_cost_registry_id=costs.registry_id,
        call_cost_registry_hash=costs.artifact_hash,
        conditional_call_rate_registry_id=rates.registry_id,
        conditional_call_rate_registry_hash=rates.artifact_hash,
        requested_core_counts={key: value for key, value in counts.core_counts.items() if type(value) is int},
        requested_extension_counts={
            key: value for key, value in counts.extension_counts.items() if type(value) is int
        },
        run_template_contributions=run_contributions,
        prefix_template_contributions=prefix_contributions,
        expected_base_calls=RationalCallCount.from_fraction(expected),
        base_calls=reservation.base_calls,
        rerun_allowance_calls=reservation.rerun_allowance_calls,
        total_calls=reservation.total_calls,
    )


def _integer_counts(*count_maps: Mapping[str, int | Literal["not_feasible"]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for count_map in count_maps:
        for task, count in count_map.items():
            if type(count) is not int or count < 0:
                if count != "not_feasible":
                    raise PlanningError("INVALID_REQUESTED_COUNT")
                continue
            counts[task] = max(counts.get(task, 0), count)
    return counts


def _component_cost(
    component: RunTemplateCallComponent,
    costs: CallCostRegistry,
    rates: Mapping[str, CallActivationRate],
) -> Fraction:
    return component.calls_per_activation * costs.call_costs[component.call_category] * rates[
        component.activation_rate_id
    ].fraction


def _validate_rates(registry: ConditionalCallRateRegistry) -> None:
    if len({rate.rate_id for rate in registry.rates}) != len(registry.rates):
        raise PlanningError("INVALID_CONDITIONAL_CALL_RATE")
    for rate in registry.rates:
        if rate.denominator <= 0 or rate.numerator < 0 or rate.numerator > rate.denominator:
            raise PlanningError("INVALID_CONDITIONAL_CALL_RATE")
        deterministic = rate.activation in {"always", "once_per_seed"}
        if deterministic and (
            rate.scope_id_or_none is not None
            or rate.scope_hash_or_none is not None
            or rate.fraction != 1
        ):
            raise PlanningError("INVALID_CONDITIONAL_CALL_RATE")
        if not deterministic and (rate.scope_id_or_none is None or rate.scope_hash_or_none is None):
            raise PlanningError("CONDITIONAL_CALL_SCOPE_MISMATCH")


__all__ = [
    "CallActivationRate",
    "CallBudgetBreakdown",
    "CallCostRegistry",
    "ConditionalCallRateRegistry",
    "ConditionalCallScope",
    "ConditionalCallScopeRegistry",
    "ConservativeRateUpperBound",
    "ConservativeRateUpperBoundRegistry",
    "PilotCallStatisticsManifest",
    "PilotCallSufficientStatistic",
    "PlanningError",
    "RationalCallCount",
    "RequestedCountVector",
    "RunTemplateCallComponent",
    "RunTemplateRegistry",
    "evaluate_route_call_budget",
    "reserve_rational_calls",
]
