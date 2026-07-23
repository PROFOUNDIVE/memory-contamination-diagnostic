from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from memcontam.config.phase12 import (
    build_candidate_template_set,
    load_phase12_config,
    resolve_phase12_config,
)
from memcontam.experiment.phase12.budget import (
    CallActivationRate,
    CallCostRegistry,
    ConditionalCallRateRegistry,
    RequestedCountVector,
    RunTemplateCallComponent,
    evaluate_route_call_budget,
)
from memcontam.experiment.phase12.contracts import (
    BehaviorTestRegistry,
    Inv03EquivalenceRegistry,
    MftManifest,
    MemoryArmExecutionKey,
    PilotBManifest,
    RouteFeasibilityReport,
    RouteSelectionManifest,
    SeedAllocationManifest,
)
from memcontam.experiment.phase12.planner import (
    PlanningError,
    build_conditional_call_scope_registry,
    generate_candidate_route_registries,
    validate_complete_template_registry,
    validate_route_selection,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-CONFIG-001.json"


def _template_set_and_registry():
    resolved = resolve_phase12_config(load_phase12_config(FIXTURE_PATH))
    template_set = build_candidate_template_set(resolved, "3w")
    other = build_candidate_template_set(resolved, "5w")
    scopes = build_conditional_call_scope_registry((template_set, other), "2026-07-23T00:00:00Z")
    always = CallActivationRate(
        rate_id="always",
        scope_id_or_none=None,
        scope_hash_or_none=None,
        activation="always",
        numerator=1,
        denominator=1,
        source_kind="deterministic",
        source_id="deterministic",
        source_hash="deterministic-hash",
    )
    components = tuple(
        RunTemplateCallComponent(
            component_id=f"{template.run_template_id}:init",
            owner_kind="run_template",
            owner_id=template.run_template_id,
            phase="init",
            call_category="model_call",
            calls_per_activation=0,
            activation_rate_id="always",
        )
        for template in template_set.run_templates
    )
    registry = generate_candidate_route_registries(
        (template_set, other), scopes, components_by_route={"3w": components}
    )[0]
    rates = ConditionalCallRateRegistry(
        registry_id="rates",
        rates=(always,),
        frozen_before_main_unblinding=True,
        artifact_hash="rates-hash",
    )
    costs = CallCostRegistry(
        registry_id="costs",
        required_call_keys=("model_call",),
        call_costs={"model_call": 1},
        rerun_allowance_numerator=1,
        rerun_allowance_denominator=20,
        artifact_hash="costs-hash",
    )
    behavior = BehaviorTestRegistry.model_construct(
        registry_id=template_set.behavior_test_registry_id,
        artifact_hash=template_set.behavior_test_registry_hash,
        source_experiment_design_sha256=template_set.authoritative_experiment_design_sha256,
        required_test_ids=(),
        rows=(),
    )
    inv03 = Inv03EquivalenceRegistry.model_construct(
        registry_id=template_set.inv03_equivalence_registry_id,
        artifact_hash=template_set.inv03_equivalence_registry_hash,
    )
    return template_set, scopes, registry, costs, rates, behavior, inv03


def _report(candidate: Literal["3w", "5w"], *, feasible: bool) -> RouteFeasibilityReport:
    return RouteFeasibilityReport(
        report_id=f"report-{candidate}",
        candidate_route=candidate,
        run_template_registry_id="registry",
        run_template_registry_hash="registry-hash",
        requested_core_counts={},
        requested_extension_counts={},
        estimated_calls=1,
        call_budget_breakdown_id="budget",
        call_budget_breakdown_hash="budget-hash",
        call_capacity=10,
        feasible=feasible,
        reasons=(),
        pilot_b_manifest_id="pilot",
        pilot_b_manifest_hash="pilot-hash",
        mft_manifest_id="mft",
        mft_manifest_hash="mft-hash",
        artifact_hash=f"report-{candidate}-hash",
    )


def test_rejects_invalid_config_runtime_and_governance_variants() -> None:
    template_set, scopes, registry, costs, rates, behavior, inv03 = _template_set_and_registry()

    with pytest.raises(PlanningError, match="CONFIG_TEMPLATE_SET_MISMATCH"):
        validate_complete_template_registry(
            replace(registry, candidate_template_set_hash="stale"),
            template_set,
            scopes,
            inv03,
            behavior,
        )
    with pytest.raises(PlanningError, match="CONDITIONAL_CALL_SCOPE_MISMATCH"):
        validate_complete_template_registry(
            replace(registry, conditional_call_scope_registry_hash="stale"),
            template_set,
            scopes,
            inv03,
            behavior,
        )
    with pytest.raises(PlanningError, match="INIT_COMPONENT_MISSING"):
        validate_complete_template_registry(
            replace(registry, call_components=()), template_set, scopes, inv03, behavior
        )

    memory_template = next(
        template
        for template in template_set.run_templates
        if template.baseline_condition_id != "nomem"
    )
    invalid_prefix = memory_template.model_copy(update={"prefix_template_key_or_none": None})
    invalid_templates = tuple(
        invalid_prefix if template.run_template_id == invalid_prefix.run_template_id else template
        for template in template_set.run_templates
    )
    invalid_set = template_set.model_copy(update={"run_templates": invalid_templates})
    with pytest.raises(PlanningError, match="PREFIX_EXECUTION_KEY_REQUIRED"):
        validate_complete_template_registry(
            replace(registry, run_templates=invalid_templates), invalid_set, scopes, inv03, behavior
        )

    filter_template = next(
        template
        for template in template_set.run_templates
        if template.baseline_condition_id == "rag_frozen"
        and isinstance(template.execution_key, MemoryArmExecutionKey)
        and template.execution_key.arm == "filter"
    )
    quarantined = filter_template.model_copy(
        update={
            "corpus_index_filter_versions": {
                **filter_template.corpus_index_filter_versions,
                "quarantine_in_index": "true",
            }
        }
    )
    quarantined_templates = tuple(
        quarantined if template.run_template_id == quarantined.run_template_id else template
        for template in template_set.run_templates
    )
    quarantined_set = template_set.model_copy(update={"run_templates": quarantined_templates})
    with pytest.raises(PlanningError, match="FILTER_INDEX_CONTAINS_QUARANTINE"):
        validate_complete_template_registry(
            replace(registry, run_templates=quarantined_templates),
            quarantined_set,
            scopes,
            inv03,
            behavior,
        )

    missing_rate_component = RunTemplateCallComponent(
        component_id="missing-rate",
        owner_kind="run_template",
        owner_id=registry.run_templates[0].run_template_id,
        phase="trial",
        call_category="model_call",
        calls_per_activation=1,
        activation_rate_id="missing",
    )
    with pytest.raises(PlanningError, match="CONDITIONAL_CALL_RATE_REQUIRED"):
        evaluate_route_call_budget(
            replace(registry, call_components=(*registry.call_components, missing_rate_component)),
            costs,
            rates,
            counts=RequestedCountVector(core_counts={}, extension_counts={}),
        )

    failed_mft = MftManifest(
        manifest_id="mft",
        artifact_hash="mft-hash",
        all_registered_cases_attempted=True,
        mft04_status="fail",
        route_gate_status="blocked",
        case_status_ledger_hash="case-hash",
        pilot_allowance_ledger_hash="allowance-hash",
        frozen_at="2026-07-23T00:00:00Z",
    )
    selection = RouteSelectionManifest(
        manifest_id="selection",
        selected_route="3w",
        feasibility_report_ids=("report-3w", "report-5w"),
        selected_feasibility_report_id="report-3w",
        selected_feasibility_report_hash="report-3w-hash",
        seed_allocation_manifest_id="allocation",
        seed_allocation_manifest_hash="allocation-hash",
        selected_after_pilot_b=True,
        mft_gate_status="pass",
        approved_by="external",
        frozen_at="2026-07-23T00:00:00Z",
        artifact_hash="selection-hash",
    )
    allocation = SeedAllocationManifest(
        manifest_id="allocation",
        selected_route="3w",
        selected_feasibility_report_id="report-3w",
        selected_feasibility_report_hash="report-3w-hash",
        run_template_registry_id="registry",
        run_template_registry_hash="registry-hash",
        requested_core_counts={},
        requested_extension_counts={},
        slot_to_seed={},
        approved_by="external",
        frozen_at="2026-07-23T00:00:00Z",
        artifact_hash="allocation-hash",
    )
    pilot = PilotBManifest(
        manifest_id="pilot",
        artifact_hash="pilot-hash",
        completed_before_main_unblinding=True,
        attempted_seed_counts={},
        cost_registry_hash="costs-hash",
        conditional_call_scope_registry_id="scopes",
        conditional_call_scope_registry_hash="scopes-hash",
        pilot_call_statistics_manifest_id="statistics",
        pilot_call_statistics_manifest_hash="statistics-hash",
        conservative_rate_upper_bound_registry_id="bounds",
        conservative_rate_upper_bound_registry_hash="bounds-hash",
        conditional_call_rate_registry_id="rates",
        conditional_call_rate_registry_hash="rates-hash",
        joint_eligibility_summary_hash="eligibility-hash",
        variance_summary_hash="variance-hash",
        frozen_at="2026-07-23T00:00:00Z",
    )
    with pytest.raises(PlanningError, match="MFT_GATE_NOT_PASS"):
        validate_route_selection(
            (_report("3w", feasible=True), _report("5w", feasible=False)),
            pilot,
            failed_mft,
            selection,
            allocation,
        )
