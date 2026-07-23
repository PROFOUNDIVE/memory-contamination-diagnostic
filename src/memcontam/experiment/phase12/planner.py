from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Literal, Mapping, Sequence, cast

from memcontam.experiment.phase12.budget import (
    CallActivationRate,
    CallCostRegistry,
    ConditionalCallRateRegistry,
    ConditionalCallScope,
    ConditionalCallScopeRegistry,
    ConservativeRateUpperBound,
    ConservativeRateUpperBoundRegistry,
    PilotCallStatisticsManifest,
    PilotCallSufficientStatistic,
    PlanningError,
    RequestedCountVector,
    RunTemplateCallComponent,
    RunTemplateRegistry,
    evaluate_route_call_budget,
)
from memcontam.experiment.phase12.contracts import (
    BehaviorTestRegistry,
    CandidateTemplateSet,
    CodeMatrixPlan,
    ExploratoryActivationManifest,
    Inv03EquivalenceRegistry,
    MemoryArmExecutionKey,
    MftManifest,
    PilotBManifest,
    RouteFeasibilityReport,
    RouteSelectionManifest,
    RouteCandidateId,
    RunTemplateSpec,
    SeedAllocationManifest,
    SelectedPackageResourceManifest,
    ValidatedExploratoryActivation,
    ValidatedRouteSelection,
    canonical_json_hash,
)
from memcontam.experiment.phase12.eligibility import JointEligibilityResult


def build_conditional_call_scope_registry(
    template_sets: Sequence[CandidateTemplateSet], frozen_at: str
) -> ConditionalCallScopeRegistry:
    if {template_set.candidate_route for template_set in template_sets} != {"3w", "5w"}:
        raise PlanningError("REQUIRED_CONFIG_CELL_MISSING")
    grouped: dict[
        tuple[str, str, str, Literal["on_attempt_1_failure", "on_retry_failure"]],
        list[tuple[RouteCandidateId, RunTemplateSpec]],
    ] = defaultdict(list)
    for template_set in template_sets:
        for template in template_set.run_templates:
            if not template.baseline_condition_id.startswith("reflexion"):
                continue
            for activation in ("on_attempt_1_failure", "on_retry_failure"):
                grouped[(template.model_snapshot, template.task_family, template.baseline_condition_id, activation)].append(
                    (template_set.candidate_route, template)
                )
    scopes = tuple(
        _scope(key, records)
        for key, records in sorted(grouped.items())
    )
    payload = {"frozen_at": frozen_at, "scopes": [asdict(scope) for scope in scopes]}
    return ConditionalCallScopeRegistry(
        registry_id=f"conditional-scopes-{canonical_json_hash(payload)[:12]}",
        scopes=scopes,
        frozen_before_main_unblinding=True,
        artifact_hash=canonical_json_hash(payload),
    )


def generate_candidate_route_registries(
    template_sets: Sequence[CandidateTemplateSet],
    scopes: ConditionalCallScopeRegistry,
    *,
    components_by_route: Mapping[str, Sequence[RunTemplateCallComponent]] | None = None,
) -> tuple[RunTemplateRegistry, ...]:
    return tuple(
        _registry(
            template_set,
            scopes,
            components=()
            if components_by_route is None
            else tuple(components_by_route.get(template_set.candidate_route, ())),
        )
        for template_set in sorted(template_sets, key=lambda item: item.candidate_route)
    )


def generate_candidate_run_registry(
    template_set: CandidateTemplateSet,
    costs: CallCostRegistry,
    scopes: ConditionalCallScopeRegistry,
    rates: ConditionalCallRateRegistry,
    inv03: Inv03EquivalenceRegistry,
    behavior: BehaviorTestRegistry,
) -> RunTemplateRegistry:
    registry = _registry(template_set, scopes, costs=costs, rates=rates)
    validate_complete_template_registry(registry, template_set, scopes, inv03, behavior)
    return registry


def validate_complete_template_registry(
    registry: RunTemplateRegistry,
    template_set: CandidateTemplateSet,
    scopes: ConditionalCallScopeRegistry,
    inv03: Inv03EquivalenceRegistry,
    behavior: BehaviorTestRegistry,
) -> None:
    if (
        registry.candidate_template_set_id != template_set.template_set_id
        or registry.candidate_template_set_hash != template_set.artifact_hash
        or registry.template_package_hash != template_set.template_package_hash
        or registry.run_templates != template_set.run_templates
        or registry.prefix_templates != template_set.prefix_templates
        or registry.abstract_slots != template_set.abstract_slots
    ):
        raise PlanningError("CONFIG_TEMPLATE_SET_MISMATCH")
    if registry.repository_commit != template_set.repository_commit:
        raise PlanningError("REPOSITORY_COMMIT_MISMATCH")
    if (
        registry.inv03_equivalence_registry_id != inv03.registry_id
        or registry.inv03_equivalence_registry_hash != inv03.artifact_hash
        or template_set.inv03_equivalence_registry_id != inv03.registry_id
        or template_set.inv03_equivalence_registry_hash != inv03.artifact_hash
    ):
        raise PlanningError("INV03_EQUIVALENCE_CONTRACT_MISMATCH")
    if (
        template_set.behavior_test_registry_id != behavior.registry_id
        or template_set.behavior_test_registry_hash != behavior.artifact_hash
        or behavior.source_experiment_design_sha256
        != template_set.authoritative_experiment_design_sha256
        or set(behavior.required_test_ids) != {row.test_id for row in behavior.rows}
    ):
        raise PlanningError("BEHAVIOR_REGISTRY_INCOMPLETE")
    prefix_keys = {prefix.prefix_template_key for prefix in template_set.prefix_templates}
    if len(prefix_keys) != len(template_set.prefix_templates):
        raise PlanningError("PREFIX_EXECUTION_KEY_REQUIRED")
    for template in template_set.run_templates:
        _validate_run_template(template, prefix_keys)
        versions = template.corpus_index_filter_versions
        content = versions.get("branch_corpus_hash")
        index = versions.get("index_hash")
        if template.baseline_condition_id == "rag_frozen" and content and index:
            if (
                isinstance(template.execution_key, MemoryArmExecutionKey)
                and template.execution_key.arm == "filter"
                and versions.get("quarantine_in_index") == "true"
            ):
                raise PlanningError("FILTER_INDEX_CONTAINS_QUARANTINE")
    if registry.conditional_call_scope_registry_id != scopes.registry_id or (
        registry.conditional_call_scope_registry_hash != scopes.artifact_hash
    ):
        raise PlanningError("CONDITIONAL_CALL_SCOPE_MISMATCH")
    init_owner_ids = {
        component.owner_id for component in registry.call_components if component.phase == "init"
    }
    if {template.run_template_id for template in template_set.run_templates} - init_owner_ids:
        raise PlanningError("INIT_COMPONENT_MISSING")


def freeze_pilot_call_statistics(
    scope_registry: ConditionalCallScopeRegistry,
    statistics: Sequence[PilotCallSufficientStatistic],
    frozen_at: str,
) -> PilotCallStatisticsManifest:
    scope_by_id = {scope.scope_id: scope for scope in scope_registry.scopes}
    if len({statistic.scope_id for statistic in statistics}) != len(statistics):
        raise PlanningError("PILOT_CALL_STATISTIC_SCOPE_MISMATCH")
    for statistic in statistics:
        scope = scope_by_id.get(statistic.scope_id)
        if (
            scope is None
            or statistic.scope_hash != scope.artifact_hash
            or statistic.eligible_opportunities <= 0
            or statistic.activated_calls < 0
            or statistic.activated_calls > statistic.eligible_opportunities
        ):
            raise PlanningError("PILOT_CALL_STATISTIC_SCOPE_MISMATCH")
    payload = {"statistics": [asdict(item) for item in statistics], "frozen_at": frozen_at}
    return PilotCallStatisticsManifest(
        manifest_id=f"pilot-call-statistics-{canonical_json_hash(payload)[:12]}",
        statistics=tuple(statistics),
        frozen_at=frozen_at,
        artifact_hash=canonical_json_hash(payload),
    )


def freeze_conservative_rate_upper_bounds(
    scope_registry: ConditionalCallScopeRegistry,
    bounds: Sequence[ConservativeRateUpperBound],
    frozen_at: str,
) -> ConservativeRateUpperBoundRegistry:
    scope_by_id = {scope.scope_id: scope for scope in scope_registry.scopes}
    if len({bound.scope_id for bound in bounds}) != len(bounds):
        raise PlanningError("CONSERVATIVE_RATE_SCOPE_MISMATCH")
    for bound in bounds:
        scope = scope_by_id.get(bound.scope_id)
        if (
            scope is None
            or bound.scope_hash != scope.artifact_hash
            or bound.numerator < 0
            or bound.denominator <= 0
            or bound.numerator > bound.denominator
            or not bound.rationale_code
            or not bound.source_hash
        ):
            raise PlanningError("CONSERVATIVE_RATE_SCOPE_MISMATCH")
    payload = {"bounds": [asdict(item) for item in bounds], "frozen_at": frozen_at}
    return ConservativeRateUpperBoundRegistry(
        registry_id=f"conservative-rates-{canonical_json_hash(payload)[:12]}",
        bounds=tuple(bounds),
        frozen_before_main_unblinding=True,
        artifact_hash=canonical_json_hash(payload),
    )


def freeze_conditional_call_rate_registry(
    scope_registry: ConditionalCallScopeRegistry,
    statistics: PilotCallStatisticsManifest,
    upper_bounds: ConservativeRateUpperBoundRegistry,
    deterministic_rates: Sequence[CallActivationRate],
    frozen_at: str,
) -> ConditionalCallRateRegistry:
    statistics_by_scope = {item.scope_id: item for item in statistics.statistics}
    bounds_by_scope = {item.scope_id: item for item in upper_bounds.bounds}
    rates = list(deterministic_rates)
    for scope in scope_registry.scopes:
        statistic = statistics_by_scope.get(scope.scope_id)
        bound = bounds_by_scope.get(scope.scope_id)
        if bool(statistic) == bool(bound):
            raise PlanningError("CONDITIONAL_CALL_SCOPE_MISMATCH")
        if statistic:
            numerator, denominator = statistic.activated_calls, statistic.eligible_opportunities
            source_kind, source_id, source_hash = "pilot_b", statistic.statistic_id, statistic.artifact_hash
        else:
            assert bound is not None
            numerator, denominator = bound.numerator, bound.denominator
            source_kind, source_id, source_hash = "registered_upper_bound", bound.bound_id, bound.artifact_hash
        rates.append(
            CallActivationRate(
                rate_id=f"rate-{scope.scope_id}",
                scope_id_or_none=scope.scope_id,
                scope_hash_or_none=scope.artifact_hash,
                activation=scope.activation,
                numerator=numerator,
                denominator=denominator,
                source_kind=source_kind,
                source_id=source_id,
                source_hash=source_hash,
            )
        )
    _validate_rate_rows(rates, scope_registry)
    payload = {"rates": [asdict(item) for item in rates], "frozen_at": frozen_at}
    return ConditionalCallRateRegistry(
        registry_id=f"conditional-rates-{canonical_json_hash(payload)[:12]}",
        rates=tuple(rates),
        frozen_before_main_unblinding=True,
        artifact_hash=canonical_json_hash(payload),
    )


def build_pilot_b_manifest(
    attempted_seed_counts: Mapping[str, int],
    costs: CallCostRegistry,
    scope_registry: ConditionalCallScopeRegistry,
    statistics: PilotCallStatisticsManifest,
    upper_bounds: ConservativeRateUpperBoundRegistry,
    rates: ConditionalCallRateRegistry,
    eligibility: JointEligibilityResult,
    variance_summary_hash: str,
    frozen_at: str,
):
    from memcontam.experiment.phase12.contracts import PilotBManifest

    if any(type(value) is not int or value < 0 for value in attempted_seed_counts.values()):
        raise PlanningError("INVALID_ATTEMPTED_SEED_COUNT")
    payload = {"attempted_seed_counts": dict(attempted_seed_counts), "frozen_at": frozen_at}
    return PilotBManifest(
        manifest_id=f"pilot-b-{canonical_json_hash(payload)[:12]}",
        artifact_hash=canonical_json_hash(payload),
        completed_before_main_unblinding=True,
        attempted_seed_counts=attempted_seed_counts,
        cost_registry_hash=costs.artifact_hash,
        conditional_call_scope_registry_id=scope_registry.registry_id,
        conditional_call_scope_registry_hash=scope_registry.artifact_hash,
        pilot_call_statistics_manifest_id=statistics.manifest_id,
        pilot_call_statistics_manifest_hash=statistics.artifact_hash,
        conservative_rate_upper_bound_registry_id=upper_bounds.registry_id,
        conservative_rate_upper_bound_registry_hash=upper_bounds.artifact_hash,
        conditional_call_rate_registry_id=rates.registry_id,
        conditional_call_rate_registry_hash=rates.artifact_hash,
        joint_eligibility_summary_hash=canonical_json_hash(
            {"indices": eligibility.joint_eligible_indices, "horizon": eligibility.horizon}
        ),
        variance_summary_hash=variance_summary_hash,
        frozen_at=frozen_at,
    )


def estimate_route_feasibility(
    registry: RunTemplateRegistry,
    costs: CallCostRegistry,
    rates: ConditionalCallRateRegistry,
    counts: RequestedCountVector,
    pilot_b: PilotBManifest,
    mft: MftManifest,
    call_capacity: int,
) -> RouteFeasibilityReport:
    if call_capacity < 0:
        raise PlanningError("INVALID_CALL_CAPACITY")
    if (
        pilot_b.cost_registry_hash != costs.artifact_hash
        or pilot_b.conditional_call_rate_registry_id != rates.registry_id
        or pilot_b.conditional_call_rate_registry_hash != rates.artifact_hash
    ):
        raise PlanningError("CONDITIONAL_CALL_RATE_HASH_MISMATCH")
    budget = evaluate_route_call_budget(registry, costs, rates, counts)
    feasible = (
        mft.all_registered_cases_attempted
        and mft.mft04_status == "pass"
        and mft.route_gate_status == "pass"
        and budget.total_calls <= call_capacity
    )
    reasons: list[str] = []
    if not mft.all_registered_cases_attempted or mft.mft04_status != "pass":
        reasons.append("MFT_GATE_INCOMPLETE")
    if mft.route_gate_status != "pass":
        reasons.append("MFT_ROUTE_GATE_BLOCKED")
    if budget.total_calls > call_capacity:
        reasons.append("CALL_CAPACITY_EXCEEDED")
    payload = {
        "candidate_route": registry.candidate_route,
        "registry": registry.artifact_hash,
        "budget": budget.artifact_hash,
        "capacity": call_capacity,
        "pilot": pilot_b.artifact_hash,
        "mft": mft.artifact_hash,
    }
    return RouteFeasibilityReport(
        report_id=f"route-report-{registry.candidate_route}-{canonical_json_hash(payload)[:12]}",
        candidate_route=cast(RouteCandidateId, registry.candidate_route),
        run_template_registry_id=registry.registry_id,
        run_template_registry_hash=registry.artifact_hash,
        requested_core_counts=counts.core_counts,
        requested_extension_counts=counts.extension_counts,
        estimated_calls=budget.total_calls,
        call_budget_breakdown_id=budget.breakdown_id,
        call_budget_breakdown_hash=budget.artifact_hash,
        call_capacity=call_capacity,
        feasible=feasible,
        reasons=tuple(reasons),
        pilot_b_manifest_id=pilot_b.manifest_id,
        pilot_b_manifest_hash=pilot_b.artifact_hash,
        mft_manifest_id=mft.manifest_id,
        mft_manifest_hash=mft.artifact_hash,
        artifact_hash=canonical_json_hash(payload),
    )


def validate_route_selection(
    reports: Sequence[RouteFeasibilityReport],
    pilot_b: PilotBManifest,
    mft: MftManifest,
    selection: RouteSelectionManifest,
    allocation: SeedAllocationManifest,
) -> ValidatedRouteSelection:
    if (
        not mft.all_registered_cases_attempted
        or mft.mft04_status != "pass"
        or mft.route_gate_status != "pass"
    ):
        raise PlanningError("MFT_GATE_NOT_PASS")
    reports_by_id = {report.report_id: report for report in reports}
    if set(report.candidate_route for report in reports) != {"3w", "5w"}:
        raise PlanningError("ROUTE_REPORT_MISSING")
    selected = reports_by_id.get(selection.selected_feasibility_report_id)
    if (
        not pilot_b.completed_before_main_unblinding
        or mft.mft04_status != "pass"
        or mft.route_gate_status != "pass"
        or selected is None
        or not selected.feasible
        or selected.artifact_hash != selection.selected_feasibility_report_hash
        or selected.candidate_route != selection.selected_route
        or selected.pilot_b_manifest_id != pilot_b.manifest_id
        or selected.pilot_b_manifest_hash != pilot_b.artifact_hash
        or selected.mft_manifest_id != mft.manifest_id
        or selected.mft_manifest_hash != mft.artifact_hash
    ):
        raise PlanningError("ROUTE_SELECTION_INVALID")
    if (
        allocation.selected_route != selection.selected_route
        or allocation.selected_feasibility_report_id != selected.report_id
        or allocation.selected_feasibility_report_hash != selected.artifact_hash
        or allocation.run_template_registry_id != selected.run_template_registry_id
        or allocation.run_template_registry_hash != selected.run_template_registry_hash
        or dict(allocation.requested_core_counts) != dict(selected.requested_core_counts)
        or dict(allocation.requested_extension_counts) != dict(selected.requested_extension_counts)
        or len(set(allocation.slot_to_seed.values())) != len(allocation.slot_to_seed)
    ):
        raise PlanningError("SEED_ASSIGNMENT_MISMATCH")
    payload = {"selection": selection.artifact_hash, "allocation": allocation.artifact_hash}
    return ValidatedRouteSelection(
        validation_hash=canonical_json_hash(payload),
        selected_route=selection.selected_route,
        route_selection_manifest_id=selection.manifest_id,
        route_selection_manifest_hash=selection.artifact_hash,
        seed_allocation_manifest_id=allocation.manifest_id,
        seed_allocation_manifest_hash=allocation.artifact_hash,
        selected_feasibility_report_id=selected.report_id,
        selected_feasibility_report_hash=selected.artifact_hash,
        run_template_registry_id=selected.run_template_registry_id,
        run_template_registry_hash=selected.run_template_registry_hash,
        pilot_b_manifest_id=pilot_b.manifest_id,
        pilot_b_manifest_hash=pilot_b.artifact_hash,
        mft_manifest_id=mft.manifest_id,
        mft_manifest_hash=mft.artifact_hash,
        slot_to_seed=allocation.slot_to_seed,
    )


def validate_exploratory_activation(
    plan: CodeMatrixPlan,
    resource: SelectedPackageResourceManifest,
    activation: ExploratoryActivationManifest,
    route_selection: ValidatedRouteSelection,
) -> ValidatedExploratoryActivation:
    if (
        resource.route_selection_manifest_id != route_selection.route_selection_manifest_id
        or resource.route_selection_manifest_hash != route_selection.route_selection_manifest_hash
        or resource.seed_allocation_manifest_id != route_selection.seed_allocation_manifest_id
        or resource.seed_allocation_manifest_hash != route_selection.seed_allocation_manifest_hash
        or resource.exploratory_plan_id != plan.plan_id
        or resource.exploratory_plan_hash != plan.artifact_hash
        or activation.resource_manifest_id != resource.manifest_id
        or activation.resource_manifest_hash != resource.artifact_hash
        or activation.exploratory_plan_id != plan.plan_id
        or activation.exploratory_plan_hash != plan.artifact_hash
        or activation.exploratory_run_template_registry_id
        != plan.exploratory_run_template_registry_id
        or activation.exploratory_run_template_registry_hash
        != plan.exploratory_run_template_registry_hash
        or set(activation.exploratory_slot_to_seed) != set(plan.abstract_slots)
        or len(set(activation.exploratory_slot_to_seed.values())) != len(activation.exploratory_slot_to_seed)
        or resource.mandatory_package_status != "fully_resourced"
        or plan.estimated_exploratory_calls > resource.exploratory_call_budget
        or resource.exploratory_call_budget + resource.reproducibility_reserve
        > resource.remaining_call_capacity
    ):
        raise PlanningError("EXPLORATORY_ACTIVATION_INVALID")
    payload = {"activation": activation.artifact_hash, "resource": resource.artifact_hash}
    return ValidatedExploratoryActivation(
        validation_hash=canonical_json_hash(payload),
        exploratory_activation_manifest_id=activation.manifest_id,
        exploratory_activation_manifest_hash=activation.artifact_hash,
        resource_manifest_id=resource.manifest_id,
        resource_manifest_hash=resource.artifact_hash,
        exploratory_plan_id=plan.plan_id,
        exploratory_plan_hash=plan.artifact_hash,
        exploratory_run_template_registry_id=plan.exploratory_run_template_registry_id,
        exploratory_run_template_registry_hash=plan.exploratory_run_template_registry_hash,
        exploratory_slot_to_seed=activation.exploratory_slot_to_seed,
        route_selection_manifest_id=route_selection.route_selection_manifest_id,
        route_selection_manifest_hash=route_selection.route_selection_manifest_hash,
        seed_allocation_manifest_id=route_selection.seed_allocation_manifest_id,
        seed_allocation_manifest_hash=route_selection.seed_allocation_manifest_hash,
        estimated_exploratory_calls=plan.estimated_exploratory_calls,
        exploratory_call_budget=resource.exploratory_call_budget,
        reproducibility_reserve=resource.reproducibility_reserve,
        remaining_call_capacity=resource.remaining_call_capacity,
    )


def _scope(
    key: tuple[str, str, str, Literal["on_attempt_1_failure", "on_retry_failure"]],
    records: list[tuple[RouteCandidateId, RunTemplateSpec]],
) -> ConditionalCallScope:
    model, task, baseline, activation = key
    route_templates: dict[RouteCandidateId, list[str]] = defaultdict(list)
    run_families: set[str] = set()
    for route, template in records:
        route_templates[route].append(template.run_template_id)
        run_families.add(template.run_family)
    domain = tuple(sorted(template_id for ids in route_templates.values() for template_id in ids))
    payload = {"key": key, "routes": sorted(route_templates), "domain": domain}
    return ConditionalCallScope(
        scope_id=f"scope-{canonical_json_hash(payload)[:12]}",
        model_snapshot=model,
        task_family=task,
        baseline_condition_id=baseline,
        activation=activation,
        candidate_routes=tuple(sorted(route_templates)),
        run_families=tuple(sorted(run_families)),
        template_domain_hash=canonical_json_hash(domain),
        artifact_hash=canonical_json_hash(payload),
    )


def _registry(
    template_set: CandidateTemplateSet,
    scopes: ConditionalCallScopeRegistry,
    *,
    costs: CallCostRegistry | None = None,
    rates: ConditionalCallRateRegistry | None = None,
    components: tuple[RunTemplateCallComponent, ...] | None = None,
) -> RunTemplateRegistry:
    payload = {
        "candidate": template_set.candidate_route,
        "template_set": template_set.artifact_hash,
        "scopes": scopes.artifact_hash,
    }
    attached_components = components or ()
    if not attached_components and costs is not None and rates is not None:
        always_rate = next(
            (
                rate.rate_id
                for rate in rates.rates
                if rate.activation == "always" and rate.fraction == 1
            ),
            None,
        )
        if always_rate is None:
            raise PlanningError("CONDITIONAL_CALL_RATE_REQUIRED")
        local_category = costs.required_call_keys[0]
        attached_components = tuple(
            RunTemplateCallComponent(
                component_id=f"{template.run_template_id}:init",
                owner_kind="run_template",
                owner_id=template.run_template_id,
                phase="init",
                call_category=local_category,
                calls_per_activation=0,
                activation_rate_id=always_rate,
            )
            for template in template_set.run_templates
        )
    return RunTemplateRegistry(
        registry_id=f"run-templates-{template_set.candidate_route}-{canonical_json_hash(payload)[:12]}",
        artifact_hash=canonical_json_hash(payload),
        repository_commit=template_set.repository_commit,
        candidate_route=template_set.candidate_route,
        candidate_template_set_id=template_set.template_set_id,
        candidate_template_set_hash=template_set.artifact_hash,
        template_package_hash=template_set.template_package_hash,
        conditional_call_scope_registry_id=scopes.registry_id,
        conditional_call_scope_registry_hash=scopes.artifact_hash,
        inv03_equivalence_registry_id=template_set.inv03_equivalence_registry_id,
        inv03_equivalence_registry_hash=template_set.inv03_equivalence_registry_hash,
        component_policy_version="attached-by-task15",
        run_templates=template_set.run_templates,
        prefix_templates=template_set.prefix_templates,
        abstract_slots=template_set.abstract_slots,
        call_components=attached_components,
    )


def _validate_run_template(template: RunTemplateSpec, prefix_keys: set[str]) -> None:
    if template.evidence_layer not in {"build", "calibration", "main", "extension"}:
        raise PlanningError("INVALID_TEMPLATE_EVIDENCE_LAYER")
    if template.run_family in {"main_a", "main_b", "extension"} and template.model_snapshot != "gpt-4o-v1":
        raise PlanningError("INVALID_MODEL_ROLE_ASSIGNMENT")
    if template.run_family == "main_c" and (
        template.model_snapshot != "frontier-model-v1"
        or template.analysis_status != "exploratory_model_specificity"
    ):
        raise PlanningError("INVALID_MODEL_ROLE_ASSIGNMENT")
    if template.baseline_condition_id == "nomem":
        if template.execution_key.kind != "nomem_singleton" or template.prefix_template_key_or_none:
            raise PlanningError("INVALID_RUN_EXECUTION_KEY")
    elif (
        not isinstance(template.execution_key, MemoryArmExecutionKey)
        or template.prefix_template_key_or_none not in prefix_keys
    ):
        raise PlanningError("PREFIX_EXECUTION_KEY_REQUIRED")


def _validate_rate_rows(
    rates: Sequence[CallActivationRate], scope_registry: ConditionalCallScopeRegistry
) -> None:
    scope_hashes = {scope.scope_id: scope.artifact_hash for scope in scope_registry.scopes}
    if len({rate.rate_id for rate in rates}) != len(rates):
        raise PlanningError("INVALID_CONDITIONAL_CALL_RATE")
    for rate in rates:
        if rate.denominator <= 0 or rate.numerator < 0 or rate.numerator > rate.denominator:
            raise PlanningError("INVALID_CONDITIONAL_CALL_RATE")
        if rate.activation in {"always", "once_per_seed"}:
            if rate.scope_id_or_none is not None or rate.scope_hash_or_none is not None or rate.fraction != 1:
                raise PlanningError("INVALID_CONDITIONAL_CALL_RATE")
        elif rate.scope_id_or_none is None or (
            scope_hashes.get(rate.scope_id_or_none) != rate.scope_hash_or_none
        ):
            raise PlanningError("CONDITIONAL_CALL_SCOPE_MISMATCH")


__all__ = [
    "PlanningError",
    "build_conditional_call_scope_registry",
    "build_pilot_b_manifest",
    "estimate_route_feasibility",
    "freeze_conditional_call_rate_registry",
    "freeze_conservative_rate_upper_bounds",
    "freeze_pilot_call_statistics",
    "generate_candidate_route_registries",
    "generate_candidate_run_registry",
    "validate_complete_template_registry",
    "validate_exploratory_activation",
    "validate_route_selection",
]
