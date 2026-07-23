from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from memcontam.experiment.phase12.contracts import MftManifest, PilotBManifest
from memcontam.experiment.phase12.planner import (
    build_conditional_call_scope_registry,
    estimate_route_feasibility,
    generate_candidate_route_registries,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase12"
FIXTURE_PATH = FIXTURE_ROOT / "FX-CONFIG-001.json"
ROUTE_FIXTURE_PATH = FIXTURE_ROOT / "FX-ROUTE-001.json"
FROZEN_AT = "2026-07-23T00:00:00Z"


def _route_fixture() -> dict[str, Any]:
    return json.loads(ROUTE_FIXTURE_PATH.read_text(encoding="utf-8"))


def _template_sets():
    resolved = resolve_phase12_config(load_phase12_config(FIXTURE_PATH))
    return resolved, tuple(
        build_candidate_template_set(resolved, candidate) for candidate in ("3w", "5w")
    )


def _costs() -> CallCostRegistry:
    return CallCostRegistry(
        registry_id="test-costs",
        required_call_keys=("model_call",),
        call_costs={"model_call": 1},
        rerun_allowance_numerator=1,
        rerun_allowance_denominator=20,
        artifact_hash="test-costs-hash",
    )


def _rate_and_components(registry, scopes, counts: RequestedCountVector, base: int):
    template = next(
        item
        for item in registry.run_templates
        if item.baseline_condition_id.startswith("reflexion") and item.population_layer == "core"
    )
    scope = next(
        item
        for item in scopes.scopes
        if item.model_snapshot == template.model_snapshot
        and item.task_family == template.task_family
        and item.activation == "on_attempt_1_failure"
    )
    multiplicity = counts.core_counts[template.task_family]
    assert type(multiplicity) is int
    rate = CallActivationRate(
        rate_id=f"rate-{registry.candidate_route}",
        scope_id_or_none=scope.scope_id,
        scope_hash_or_none=scope.artifact_hash,
        activation="on_attempt_1_failure",
        numerator=1,
        denominator=multiplicity * template.horizon,
        source_kind="registered_upper_bound",
        source_id="test-upper-bound",
        source_hash="test-upper-bound-hash",
    )
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
            component_id=f"{item.run_template_id}:init",
            owner_kind="run_template",
            owner_id=item.run_template_id,
            phase="init",
            call_category="model_call",
            calls_per_activation=0,
            activation_rate_id="always",
        )
        for item in registry.run_templates
    ) + (
        RunTemplateCallComponent(
            component_id=f"{template.run_template_id}:failure-gated",
            owner_kind="run_template",
            owner_id=template.run_template_id,
            phase="trial",
            call_category="model_call",
            calls_per_activation=base,
            activation_rate_id=rate.rate_id,
        ),
    )
    return ConditionalCallRateRegistry(
        registry_id=f"rates-{registry.candidate_route}",
        rates=(always, rate),
        frozen_before_main_unblinding=True,
        artifact_hash=f"rates-{registry.candidate_route}-hash",
    ), components


def _pilot_and_mft(costs, rates):
    pilot = PilotBManifest(
        manifest_id="pilot-b-test",
        artifact_hash="pilot-b-test-hash",
        completed_before_main_unblinding=True,
        attempted_seed_counts={"game24": 1},
        cost_registry_hash=costs.artifact_hash,
        conditional_call_scope_registry_id="scopes",
        conditional_call_scope_registry_hash="scopes-hash",
        pilot_call_statistics_manifest_id="statistics",
        pilot_call_statistics_manifest_hash="statistics-hash",
        conservative_rate_upper_bound_registry_id="bounds",
        conservative_rate_upper_bound_registry_hash="bounds-hash",
        conditional_call_rate_registry_id=rates.registry_id,
        conditional_call_rate_registry_hash=rates.artifact_hash,
        joint_eligibility_summary_hash="eligibility-hash",
        variance_summary_hash="variance-hash",
        frozen_at=FROZEN_AT,
    )
    mft = MftManifest(
        manifest_id="mft-test",
        artifact_hash="mft-test-hash",
        all_registered_cases_attempted=True,
        mft04_status="pass",
        route_gate_status="pass",
        case_status_ledger_hash="case-ledger-hash",
        pilot_allowance_ledger_hash="allowance-ledger-hash",
        frozen_at=FROZEN_AT,
    )
    return pilot, mft


def test_generates_both_candidate_registries_and_feasibility_reports() -> None:
    resolved, template_sets = _template_sets()
    route_config = {item.candidate_route: item for item in resolved.source.route_candidates}
    scopes = build_conditional_call_scope_registry(template_sets, frozen_at=FROZEN_AT)
    bare_registries = generate_candidate_route_registries(template_sets, scopes)
    counts = {
        candidate: RequestedCountVector(
            core_counts=route_config[candidate].requested_core_counts,
            extension_counts=route_config[candidate].requested_extension_counts,
        )
        for candidate in ("3w", "5w")
    }
    expected_base = {"3w": 14770, "5w": 28336}
    costs = _costs()
    rate_and_components = {
        registry.candidate_route: _rate_and_components(
            registry,
            scopes,
            counts[registry.candidate_route],
            expected_base[registry.candidate_route],
        )
        for registry in bare_registries
    }
    registries = generate_candidate_route_registries(
        template_sets,
        scopes,
        components_by_route={
            route: components for route, (_, components) in rate_and_components.items()
        },
    )
    reports = {
        registry.candidate_route: estimate_route_feasibility(
            registry,
            costs,
            rate_and_components[registry.candidate_route][0],
            counts[registry.candidate_route],
            *_pilot_and_mft(costs, rate_and_components[registry.candidate_route][0]),
            route_config[registry.candidate_route].max_calls,
        )
        for registry in registries
    }
    budgets = {
        registry.candidate_route: evaluate_route_call_budget(
            registry,
            costs,
            rate_and_components[registry.candidate_route][0],
            counts[registry.candidate_route],
        )
        for registry in registries
    }

    assert {registry.candidate_route for registry in registries} == {"3w", "5w"}
    assert all(
        registry.run_templates == template_set.run_templates
        and registry.prefix_templates == template_set.prefix_templates
        and registry.abstract_slots == template_set.abstract_slots
        for registry, template_set in zip(registries, template_sets, strict=True)
    )
    assert all(scope.template_domain_hash and scope.candidate_routes for scope in scopes.scopes)
    assert all(
        template.run_family != "main_c"
        or (
            template.model_snapshot == "frontier-model-v1"
            and template.analysis_status == "exploratory_model_specificity"
        )
        for registry in registries
        for template in registry.run_templates
    )
    frozen_reports = {
        item["candidate_route"]: item for item in _route_fixture()["feasibility_reports"]
    }
    assert reports["3w"].estimated_calls == frozen_reports["3w"]["estimated_calls"] == 15509
    assert reports["3w"].call_capacity == frozen_reports["3w"]["call_capacity"] == 17000
    assert reports["3w"].feasible is True
    assert reports["5w"].estimated_calls == frozen_reports["5w"]["estimated_calls"] == 29753
    assert reports["5w"].call_capacity == frozen_reports["5w"]["call_capacity"] == 29000
    assert reports["5w"].feasible is False
    assert reports["5w"].reasons == ("CALL_CAPACITY_EXCEEDED",)
    assert (budgets["3w"].base_calls, budgets["3w"].rerun_allowance_calls) == (14770, 739)
    assert (budgets["5w"].base_calls, budgets["5w"].rerun_allowance_calls) == (28336, 1417)
    assert all(
        report.call_budget_breakdown_id == f"budget-{route}" for route, report in reports.items()
    )
    assert all(
        report.call_budget_breakdown_hash and report.artifact_hash for report in reports.values()
    )
