from __future__ import annotations

import json
from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from memcontam.experiment.phase12.filter_challenge.mft_state import (
    MFT_STATE_IDS,
    MftGateResult,
    MftMachineObservation,
    MftStateContext,
    MftStateMutation,
    MftStateReport,
    mft_state_evidence_hash,
    run_mft_state_gates,
    write_mft_state_report,
)
from memcontam.experiment.phase12.filter_challenge.registry import validate_registry_closure
from memcontam.experiment.phase12.filter_challenge.executor_types import PairingIdentity
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12" / "filter_v5"
EXPECTED_IDS = (
    "MFT-FV5-01-PAIR-MATCH",
    "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE",
    "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE",
    "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT",
    "MFT-FV5-08-NO-WRITEBACK",
)


def _context() -> MftStateContext:
    search = SearchConfig.model_validate(
        yaml.safe_load(
            (FIXTURE_ROOT / "FilterChallengeSearchConfig.yaml").read_text(encoding="utf-8")
        )
    )
    inventory = ProbeInventoryRegistry.model_validate_json(
        (FIXTURE_ROOT / "probe_inventory_manifest.json").read_text(encoding="utf-8")
    )
    suite_registry = OperationalSuiteRegistry.model_validate_json(
        (FIXTURE_ROOT / "operational_suite_manifest.json").read_text(encoding="utf-8")
    )
    closure = validate_registry_closure(search, inventory, suite_registry)
    return MftStateContext(
        search_config_id=closure.search_config_id,
        search_config_hash=search.search_config_hash,
        calibration_probe_inventory_id=closure.calibration_probe_inventory_id,
        calibration_probe_inventory_manifest_hash=(
            closure.calibration_probe_inventory_manifest_hash
        ),
        operational_probe_suite_manifest_id=closure.operational_probe_suite_manifest_id,
        operational_probe_suite_manifest_hash=(
            closure.operational_probe_suite_manifest_hash
        ),
        suite_candidate=search.suite_candidates[0],
        kappa_candidate=search.kappa_candidates[0],
    )


def test_registry_runs_each_exact_id_once_with_hash_bound_machine_evidence() -> None:
    report = run_mft_state_gates(_context())

    assert MFT_STATE_IDS == EXPECTED_IDS
    assert report.ordered_test_ids == EXPECTED_IDS
    assert tuple(result.test_id for result in report.results) == EXPECTED_IDS
    assert tuple(result.execution_index for result in report.results) == tuple(range(1, 9))
    assert Counter(result.test_id for result in report.results) == Counter(
        {test_id: 1 for test_id in EXPECTED_IDS}
    )
    assert all(result.status == "pass" for result in report.results)
    assert all(result.reason == "MFT_GATE_PASSED" for result in report.results)
    assert all(result.evidence_hash == mft_state_evidence_hash(result) for result in report.results)
    assert report.provider_calls_issued == 0


def test_pair_match_exposes_explicit_candidate_only_config_and_hash_diff() -> None:
    result = run_mft_state_gates(_context()).results[0]

    assert result.actual.paired_execution_identity_status == "matched"
    assert result.actual.paired_identity_fields == tuple(
        field.name for field in fields(PairingIdentity)
    )
    assert result.actual.config_diff_fields == ("candidate_entry_id",)
    assert result.actual.control_config_hash != result.actual.challenge_config_hash
    assert result.actual.source_state_before_hash == result.actual.source_state_after_hash


def test_exposure_tristate_fail_open_and_nonharm_routes_are_exact() -> None:
    results = {result.test_id: result.actual for result in run_mft_state_gates(_context()).results}

    exposure = results["MFT-FV5-02-EXPOSURE-REQUIRED"]
    assert exposure.candidate_final_context_inclusions == (False,) * 4
    assert exposure.assessment_states == ("not_evaluable",)
    assert exposure.route_targets == ("active",)
    assert exposure.probe_reason_codes == ("CANDIDATE_NOT_EXPOSED",) * 4

    tristate = results["MFT-FV5-03-TRISTATE"]
    assert tristate.assessment_states == (
        "contradicted",
        "not_contradicted",
        "not_evaluable",
    )
    assert tristate.route_targets == ("quarantine", "active", "active")
    assert tristate.routing_reason_codes == (
        "CONTRADICTED",
        "NOT_CONTRADICTED",
        "FAIL_OPEN_NOT_EVALUABLE",
    )

    fail_open = results["MFT-FV5-04-FAIL-OPEN"]
    assert fail_open.probe_reason_codes == (
        "CONTROL_PROVIDER_FAILURE",
        "CONTROL_PARSE_FAILURE",
        "CONTROL_VERIFIER_FAILURE",
        "CHALLENGE_PROVIDER_FAILURE",
        "CHALLENGE_PARSE_FAILURE",
        "CHALLENGE_VERIFIER_FAILURE",
    )
    assert fail_open.scripted_attempt_counts == (2,) * 6
    assert fail_open.route_targets == ("active",) * 6
    assert fail_open.audit_flags == (True,) * 6

    assert results["MFT-FV5-06-SCRIPTED-CORRECT"].route_targets == ("active",)
    irrelevant = results["MFT-FV5-07-SCRIPTED-IRRELEVANT"]
    assert irrelevant.route_targets == ("active",)
    assert irrelevant.probe_reason_codes == (
        "OUTPUT_DIVERGENCE_WITHOUT_VERIFIED_HARM",
    ) * 4


def test_route_metadata_and_challenge_artifacts_cannot_change_science_or_state() -> None:
    results = {result.test_id: result.actual for result in run_mft_state_gates(_context()).results}

    invariant = results["MFT-FV5-05-ROUTE-INVARIANCE"]
    assert len(set(invariant.excluded_metadata_hashes)) == 2
    assert len(set(invariant.policy_input_hashes)) == 1
    assert invariant.assessment_states == ("contradicted", "contradicted")
    assert invariant.route_targets == ("quarantine", "quarantine")

    no_writeback = results["MFT-FV5-08-NO-WRITEBACK"]
    assert no_writeback.challenge_output_artifact_count == 1
    assert no_writeback.challenge_failure_artifact_count == 1
    assert no_writeback.challenge_record_artifact_count == 2
    assert no_writeback.active_memory_write_count == 0
    assert no_writeback.ordinary_trial_write_count == 0
    assert no_writeback.updater_write_count == 0
    assert no_writeback.source_state_before_hash == no_writeback.source_state_after_hash


@pytest.mark.parametrize(
    ("mutation", "test_id", "reason"),
    (
        ("pair_identity", "MFT-FV5-01-PAIR-MATCH", "PAIRED_EXECUTION_IDENTITY_MISMATCH"),
        ("exposure", "MFT-FV5-02-EXPOSURE-REQUIRED", "CANDIDATE_EXPOSURE_ASSERTION_FAILED"),
        ("route", "MFT-FV5-03-TRISTATE", "ROUTING_RECONCILIATION_FAILED"),
        ("source_state", "MFT-FV5-08-NO-WRITEBACK", "SOURCE_DRIFT"),
    ),
)
def test_negative_mutations_fail_for_exact_structural_reason(
    mutation: MftStateMutation, test_id: str, reason: str
) -> None:
    report = run_mft_state_gates(_context(), mutation=mutation)
    failed = tuple(result for result in report.results if result.status == "fail")

    assert tuple((result.test_id, result.reason) for result in failed) == ((test_id, reason),)
    assert all(
        result.status == "pass" for result in report.results if result.test_id != test_id
    )
    actual = failed[0].actual
    match mutation:
        case "pair_identity":
            assert actual.paired_execution_identity_status == "mismatched"
            assert actual.config_diff_fields == (
                "candidate_entry_id",
                "pairing_identity.model_snapshot",
            )
        case "exposure":
            assert all(actual.candidate_final_context_inclusions)
            assert actual.assessment_states == ("contradicted",)
            assert actual.route_targets == ("quarantine",)
        case "route":
            assert actual.route_targets[0] == "active"
            assert failed[0].expected.route_targets[0] == "quarantine"
        case "source_state":
            assert actual.source_state_before_hash != actual.source_state_after_hash


def test_report_is_canonical_json_without_prose_decision_fields(tmp_path: Path) -> None:
    output = tmp_path / "task-12-mft-state.json"
    report = write_mft_state_report(output, _context())
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == report.model_dump(mode="json")
    assert payload["decision_input_kind"] == "machine_structure"
    assert payload["ordered_test_ids"] == list(EXPECTED_IDS)
    assert output.read_text(encoding="utf-8") == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert tuple(MftStateReport.model_fields) == (
        "schema_version",
        "evidence_layer",
        "scientific_result",
        "fixture_only",
        "decision_input_kind",
        "ordered_test_ids",
        "results",
        "provider_calls_issued",
    )
    assert tuple(MftGateResult.model_fields) == (
        "schema_version",
        "test_id",
        "execution_index",
        "inputs",
        "expected",
        "actual",
        "reason",
        "status",
        "evidence_hash",
    )
    assert tuple(MftMachineObservation.model_fields) == (
        "paired_execution_identity_status",
        "paired_identity_fields",
        "config_diff_fields",
        "control_config_hash",
        "challenge_config_hash",
        "candidate_final_context_inclusions",
        "assessment_states",
        "route_targets",
        "audit_flags",
        "probe_reason_codes",
        "routing_reason_codes",
        "scripted_attempt_counts",
        "excluded_metadata_hashes",
        "policy_input_hashes",
        "source_state_before_hash",
        "source_state_after_hash",
        "challenge_output_artifact_count",
        "challenge_failure_artifact_count",
        "challenge_record_artifact_count",
        "active_memory_write_count",
        "ordinary_trial_write_count",
        "updater_write_count",
    )
    forbidden_keys = {
        "candidate_native_content",
        "candidate_role",
        "correctness_label",
        "irrelevance_label",
        "prompt",
        "raw_output",
        "treatment_arm",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert keys(payload).isdisjoint(forbidden_keys)
