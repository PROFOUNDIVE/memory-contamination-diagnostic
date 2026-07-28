from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]
from pydantic import TypeAdapter, ValidationError

from memcontam.experiment.phase12.filter_challenge.audit import ChallengeAuditLabels, PostRouteAuditJoin
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeRoutingDecision
from memcontam.experiment.phase12.filter_challenge.records import FilterChallengeArchive
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


ASSESSMENT_FIELDS = (
    "schema_version", "filter_assessment_id", "evidence_layer", "run_family", "record_kind",
    "filter_policy_version", "policy_family", "decision_rule_id", "failure_mode_id",
    "candidate_entry_id", "candidate_native_kind", "candidate_domain_status",
    "policy_activation_checkpoint_id", "baseline_family", "rag_mode", "source_checkpoint_id",
    "source_active_state_hash", "calibration_probe_inventory_id",
    "calibration_probe_inventory_manifest_hash", "operational_probe_suite_id",
    "operational_probe_suite_manifest_hash", "probe_map_version", "challenge_suite_key",
    "probe_id", "probe_source_span_ids", "replicate_id", "control_trial_id", "control_call_id",
    "control_answer_call_id", "control_parsed_response_source_call_id",
    "control_answer_call_provenance_status", "challenge_trial_id", "challenge_call_id",
    "challenge_answer_call_id", "challenge_parsed_response_source_call_id",
    "challenge_answer_call_provenance_status", "paired_execution_identity_status",
    "control_prompt_hash", "challenge_prompt_hash", "control_raw_output_hash",
    "challenge_raw_output_hash", "control_provider_status", "challenge_provider_status",
    "control_raw_parse_status", "challenge_raw_parse_status",
    "control_canonicalizer_version", "challenge_canonicalizer_version",
    "control_canonicalized_output_hash", "challenge_canonicalized_output_hash",
    "control_canonicalized_parse_status", "challenge_canonicalized_parse_status",
    "control_verifier_status", "challenge_verifier_status", "control_verifier_result",
    "challenge_verifier_result", "control_probe_eligibility_state",
    "candidate_final_context_inclusion", "candidate_final_context_source_ids",
    "noncandidate_displacement_ids", "probe_disposition", "probe_reason_code",
    "assessment_state", "final_routing_decision", "routing_reason_code",
    "retry_count_control", "retry_count_challenge", "baseline_native_aux_call_ids_control",
    "baseline_native_aux_call_ids_challenge", "input_tokens", "output_tokens", "monetary_cost",
    "control_latency_ms", "challenge_latency_ms", "canonicalization_latency_ms",
    "total_latency_ms", "cache_key_control", "archive_path", "raw_record_ranges",
)
AGGREGATE_FIELDS = (
    "schema_version", "candidate_entry_id", "filter_policy_version",
    "calibration_probe_inventory_id", "calibration_probe_inventory_manifest_hash",
    "operational_probe_suite_id", "operational_probe_suite_manifest_hash",
    "decision_rule_id", "coverage_contract_id", "n_nominal_attempted_pairs",
    "n_control_strict_primary_eligible", "n_control_canonicalization_sensitivity_eligible",
    "n_candidate_exposed", "n_strictly_evaluable", "n_witness", "n_no_witness",
    "n_not_evaluable", "n_distinct_evaluable_probes", "n_distinct_witness_probes",
    "witness_probe_ids", "not_evaluable_reason_counts", "aggregation_parameter_tuple",
    "assessment_state", "final_routing_decision", "final_reason_code", "total_answer_calls",
    "total_baseline_native_aux_calls", "total_calls", "total_retries", "total_tokens",
    "total_cost", "total_latency_ms",
)
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase12" / "filter_v5"


def _module():
    return importlib.import_module("memcontam.experiment.phase12.filter_challenge.archive")


def _records():
    return importlib.import_module("memcontam.experiment.phase12.filter_challenge.records")


def _authority(module):
    search = SearchConfig.model_validate(
        yaml.safe_load((FIXTURE_ROOT / "FilterChallengeSearchConfig.yaml").read_text(encoding="utf-8"))
    )
    payload = search.model_dump(mode="json")
    payload["kappa_candidates"] = [
        payload["kappa_candidates"][0]
        | {
            "min_total_evaluable_replicates": 1,
            "min_distinct_evaluable_probes": 1,
            "min_witness_replicates_per_probe": 1,
            "min_distinct_witness_probes": 1,
        }
    ]
    hash_payload = payload.copy()
    del hash_payload["search_config_hash"]
    payload["search_config_hash"] = hashlib.sha256(
        json.dumps(hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    search = SearchConfig.model_validate(payload)
    assert search.search_config_hash == search.stable_hash()
    inventory = ProbeInventoryRegistry.model_validate_json(
        (FIXTURE_ROOT / "probe_inventory_manifest.json").read_text(encoding="utf-8")
    )
    suite = OperationalSuiteRegistry.model_validate_json(
        (FIXTURE_ROOT / "operational_suite_manifest.json").read_text(encoding="utf-8")
    )
    return module.ArchiveRegistryAuthority(search, inventory, suite)


def _canonical(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def _archive(authority) -> FilterChallengeArchive:
    module = _records()
    search = authority.search_config
    inventory = authority.inventory
    suite = authority.suite
    calls = (
        module.ChallengeCallRecord(
            call_id="control-answer-1", filter_assessment_id="assessment-1", call_kind="answer",
            side="control", input_tokens=3, output_tokens=1, monetary_cost=0.1, latency_ms=4,
            retry_count=0,
        ),
        module.ChallengeCallRecord(
            call_id="challenge-answer-1", filter_assessment_id="assessment-1", call_kind="answer",
            side="challenge", input_tokens=4, output_tokens=2, monetary_cost=0.2, latency_ms=5,
            retry_count=0,
        ),
    )
    call_lines = tuple(_canonical(call.model_dump(mode="json")).encode() for call in calls)
    call_bytes = b"".join(call_lines)
    assessment = module.AssessmentRecord(
        filter_assessment_id="assessment-1", filter_policy_version="verifier-paired-challenge-v1",
        policy_family="verifier_backed_paired_challenge", decision_rule_id="rule-1",
        failure_mode_id="fail_open", candidate_entry_id="candidate-1",
        candidate_native_kind="full_history_transcript", candidate_domain_status="challenge_routable_v1",
        policy_activation_checkpoint_id="checkpoint-1", baseline_family="full_history",
        rag_mode="not_applicable", source_checkpoint_id="checkpoint-1",
        source_active_state_hash="source-hash-1", calibration_probe_inventory_id=inventory.registry_id,
        calibration_probe_inventory_manifest_hash=inventory.calibration_probe_inventory_manifest_hash,
        operational_probe_suite_id=search.suite_candidates[0].operational_probe_suite_id,
        operational_probe_suite_manifest_hash=suite.operational_probe_suite_manifest_hash,
        probe_map_version="phase12-filter-probe-map-v1", challenge_suite_key="suite-key-1",
        probe_id=inventory.probe_ids[0], probe_source_span_ids=("span-1",),
        replicate_id="replicate-1", control_trial_id="control-trial-1", control_call_id="control-answer-1",
        control_answer_call_id="control-answer-1", control_parsed_response_source_call_id="control-answer-1",
        control_answer_call_provenance_status="explicit_matched", challenge_trial_id="challenge-trial-1",
        challenge_call_id="challenge-answer-1", challenge_answer_call_id="challenge-answer-1",
        challenge_parsed_response_source_call_id="challenge-answer-1",
        challenge_answer_call_provenance_status="explicit_matched", paired_execution_identity_status="matched",
        control_prompt_hash="control-prompt-hash", challenge_prompt_hash="challenge-prompt-hash",
        control_raw_output_hash="control-raw-hash", challenge_raw_output_hash="challenge-raw-hash",
        control_provider_status="success", challenge_provider_status="success",
        control_raw_parse_status="parsed_raw", challenge_raw_parse_status="parsed_raw",
        control_canonicalizer_version="canon-1", challenge_canonicalizer_version="canon-1",
        control_canonicalized_output_hash="control-canonical-hash",
        challenge_canonicalized_output_hash="challenge-canonical-hash",
        control_canonicalized_parse_status="parsed", challenge_canonicalized_parse_status="parsed",
        control_verifier_status="success", challenge_verifier_status="success",
        control_verifier_result=True, challenge_verifier_result=False,
        control_probe_eligibility_state="strict_primary_eligible",
        candidate_final_context_inclusion=True, candidate_final_context_source_ids=("candidate-1",),
        noncandidate_displacement_ids=(), probe_disposition="witness",
        probe_reason_code="VERIFIER_HARM_WITNESS", assessment_state="contradicted",
        final_routing_decision="quarantine", routing_reason_code="CONTRADICTED",
        retry_count_control=0, retry_count_challenge=0, baseline_native_aux_call_ids_control=(),
        baseline_native_aux_call_ids_challenge=(), input_tokens=7, output_tokens=3, monetary_cost=0.3,
        control_latency_ms=4, challenge_latency_ms=5, canonicalization_latency_ms=1,
        total_latency_ms=10, cache_key_control="cache-key-1", archive_path="assessments.jsonl",
        raw_record_ranges=(
            module.RawRecordRange(path="calls.jsonl", start=0, end=len(call_lines[0])),
            module.RawRecordRange(path="calls.jsonl", start=len(call_lines[0]), end=len(call_bytes)),
        ),
    )
    aggregate = module.CandidateAggregateRecord(
        candidate_entry_id="candidate-1", filter_policy_version="verifier-paired-challenge-v1",
        calibration_probe_inventory_id=inventory.registry_id,
        calibration_probe_inventory_manifest_hash=inventory.calibration_probe_inventory_manifest_hash,
        operational_probe_suite_id=search.suite_candidates[0].operational_probe_suite_id,
        operational_probe_suite_manifest_hash=suite.operational_probe_suite_manifest_hash,
        decision_rule_id="rule-1", coverage_contract_id=search.coverage_contract_candidates[0].coverage_contract_id,
        n_nominal_attempted_pairs=1, n_control_strict_primary_eligible=1,
        n_control_canonicalization_sensitivity_eligible=0,
        n_candidate_exposed=1, n_strictly_evaluable=1, n_witness=1, n_no_witness=0,
        n_not_evaluable=0, n_distinct_evaluable_probes=1, n_distinct_witness_probes=1,
        witness_probe_ids=(inventory.probe_ids[0],), not_evaluable_reason_counts={},
        aggregation_parameter_tuple=(search.kappa_candidates[0].kappa_id, search.coverage_contract_candidates[0].coverage_contract_id), assessment_state="contradicted",
        final_routing_decision="quarantine", final_reason_code="CONTRADICTED", total_answer_calls=2,
        total_baseline_native_aux_calls=0, total_calls=2, total_retries=0, total_tokens=10,
        total_cost=0.3, total_latency_ms=10,
    )
    audit = PostRouteAuditJoin(
        candidate_entry_id="candidate-1",
        routing_decision=TypeAdapter(ChallengeRoutingDecision).validate_python(
            {
                "assessment_state": "contradicted", "route_target": "quarantine", "audit_flag": False,
                "routing_reason_code": "CONTRADICTED",
            }
        ),
        audit_labels=ChallengeAuditLabels(
            candidate_role="false", correctness_label="incorrect", irrelevance_label="not_irrelevant",
            B_star_membership=False, is_injected=True, origin_class="protocol_injected",
            injection_event_id="injection-1", treatment_arm="filter", future_main_outcomes="unobserved",
            future_suffix_outcomes="unobserved",
        ),
    )
    return module.FilterChallengeArchive(
        run=module.FilterChallengeArchiveRun(run_id="filter-v5-build"), assessments=(assessment,),
        candidate_aggregates=(aggregate,), calls=calls, audit_labels=(audit,),
    )


def _reseal(root: Path) -> None:
    manifest = json.loads((root / "public_artifact_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["artifacts"].items():
        path = root / name
        record["count"] = 1 if path.suffix == ".json" else len(path.read_text(encoding="utf-8").splitlines())
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = root / "public_artifact_manifest.json"
    manifest_path.write_text(_canonical(manifest), encoding="utf-8")
    audit_path = root / "audit" / "audit_labels.jsonl"
    (root / "archive_seal.json").write_text(
        _canonical(
            {
                "public_artifact_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "audit_artifacts": {
                    "audit/audit_labels.jsonl": {
                        "count": len(audit_path.read_text(encoding="utf-8").splitlines()),
                        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_archive_module_is_available() -> None:
    assert importlib.util.find_spec("memcontam.experiment.phase12.filter_challenge.archive") is not None


def test_records_pin_literal_closed_field_tuples_and_reject_audit_extras() -> None:
    module = _records()

    archive = _archive(_authority(_module()))

    assert tuple(module.AssessmentRecord.model_fields) == ASSESSMENT_FIELDS
    assert tuple(module.CandidateAggregateRecord.model_fields) == AGGREGATE_FIELDS
    assert archive.assessments[0].schema_version == "filter_challenge_assessment_record_v1"
    assert archive.candidate_aggregates[0].schema_version == "filter_challenge_candidate_aggregate_v1"
    with pytest.raises(ValidationError):
        module.AssessmentRecord.model_validate({**archive.assessments[0].model_dump(), "candidate_role": "false"})


def test_record_hash_uses_task_three_utf8_projection() -> None:
    module = _records()
    record = _archive(_authority(_module())).assessments[0].model_copy(
        update={"cache_key_control": "unicode-한글"}
    )
    expected = hashlib.sha256(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert module.canonical_record_hash(record) == expected


def test_kappa_contradiction_accepts_one_qualifying_witness_probe() -> None:
    authority = _authority(_module())
    archive = _archive(authority)
    search = authority.search_config.model_copy(
        update={
            "kappa_candidates": (
                authority.search_config.kappa_candidates[0].model_copy(
                    update={
                        "min_total_evaluable_replicates": 3,
                        "min_distinct_evaluable_probes": 2,
                        "min_witness_replicates_per_probe": 2,
                        "min_distinct_witness_probes": 1,
                    }
                ),
            )
        }
    )
    first = archive.assessments[0]
    rows = (
        first,
        first.model_copy(update={"filter_assessment_id": "assessment-2"}),
        first.model_copy(
            update={"filter_assessment_id": "assessment-3", "probe_id": authority.inventory.probe_ids[1]}
        ),
    )
    aggregate = archive.candidate_aggregates[0].model_copy(
        update={
            "n_strictly_evaluable": 3,
            "n_distinct_evaluable_probes": 2,
            "n_witness": 3,
            "n_distinct_witness_probes": 2,
            "witness_probe_ids": authority.inventory.probe_ids[:2],
        }
    )
    authority_module = importlib.import_module("memcontam.experiment.phase12.filter_challenge.archive_authority")

    assert authority_module.expected_aggregate_state(aggregate, rows, search) == (
        "contradicted",
        "quarantine",
        "CONTRADICTED",
    )


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("", "ARCHIVE_PATH_EMPTY"), ("/calls.jsonl", "ARCHIVE_PATH_ABSOLUTE"),
        ("C:/calls.jsonl", "ARCHIVE_PATH_DRIVE"), (".", "ARCHIVE_PATH_DOT"),
        ("../calls.jsonl", "ARCHIVE_PATH_PARENT"), ("calls\\raw.jsonl", "ARCHIVE_PATH_BACKSLASH"),
        ("calls//raw.jsonl", "ARCHIVE_PATH_NORMALIZATION"),
    ],
)
def test_records_reject_noncanonical_archive_paths(path: str, code: str) -> None:
    module = _records()
    archive = _archive(_authority(_module()))
    payload = archive.assessments[0].model_dump()
    payload["archive_path"] = path

    with pytest.raises(ValidationError, match=code):
        module.AssessmentRecord.model_validate(payload)


def test_writer_seals_exact_streams_and_keeps_audit_after_route(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    root = tmp_path / "archive"

    module.write_archive(root, _archive(authority), authority)

    assert {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == {
        "run.json", "assessments.jsonl", "candidate_aggregates.jsonl", "calls.jsonl",
        "audit/audit_labels.jsonl", "public_artifact_manifest.json", "archive_seal.json",
    }
    assert module.validate_archive(root, authority).archive_valid is True
    public_bytes = "".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("run.json", "assessments.jsonl", "candidate_aggregates.jsonl", "calls.jsonl")
    )
    assert "candidate_role" not in public_bytes
    assert "future_main_outcomes" not in public_bytes


def test_validator_rejects_resealed_registry_authority_forgery(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    root = tmp_path / "archive"
    module.write_archive(root, _archive(authority), authority)

    assessment = json.loads((root / "assessments.jsonl").read_text(encoding="utf-8"))
    assessment["calibration_probe_inventory_id"] = "forged-inventory"
    (root / "assessments.jsonl").write_text(_canonical(assessment), encoding="utf-8")
    aggregate = json.loads((root / "candidate_aggregates.jsonl").read_text(encoding="utf-8"))
    aggregate["calibration_probe_inventory_id"] = "forged-inventory"
    (root / "candidate_aggregates.jsonl").write_text(_canonical(aggregate), encoding="utf-8")
    _reseal(root)

    assert module.validate_archive(root, authority).reason_code == "REGISTRY_AUTHORITY_MISMATCH"


def test_validator_rejects_missing_ranges_and_duplicate_aggregates(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    root = tmp_path / "ranges"
    module.write_archive(root, _archive(authority), authority)
    assessment = json.loads((root / "assessments.jsonl").read_text(encoding="utf-8"))
    assessment["raw_record_ranges"] = assessment["raw_record_ranges"][:1]
    (root / "assessments.jsonl").write_text(_canonical(assessment), encoding="utf-8")
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "RAW_RECORD_RANGE_INVALID"

    root = tmp_path / "duplicate"
    module.write_archive(root, _archive(authority), authority)
    aggregate = (root / "candidate_aggregates.jsonl").read_text(encoding="utf-8")
    (root / "candidate_aggregates.jsonl").write_text(aggregate * 2, encoding="utf-8")
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "AGGREGATE_IDENTITY_INVALID"


def test_validator_binds_suite_and_aggregation_parameters_to_authority(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    root = tmp_path / "archive"
    module.write_archive(root, _archive(authority), authority)

    suite = authority.search_config.suite_candidates[0].model_copy(
        update={"probe_ids": (authority.inventory.probe_ids[1],)}
    )
    wrong_suite = module.ArchiveRegistryAuthority(
        authority.search_config.model_copy(
            update={"suite_candidates": (suite, authority.search_config.suite_candidates[1])}
        ),
        authority.inventory,
        authority.suite,
    )
    assert module.validate_archive(root, wrong_suite).reason_code == "REGISTRY_AUTHORITY_MISMATCH"

    aggregate = json.loads((root / "candidate_aggregates.jsonl").read_text(encoding="utf-8"))
    aggregate["aggregation_parameter_tuple"][0] = "forged-kappa"
    (root / "candidate_aggregates.jsonl").write_text(_canonical(aggregate), encoding="utf-8")
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "REGISTRY_AUTHORITY_MISMATCH"

def test_validator_rejects_resealed_unknown_registry_probe(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    root = tmp_path / "archive"
    module.write_archive(root, _archive(authority), authority)

    assessment = json.loads((root / "assessments.jsonl").read_text(encoding="utf-8"))
    assessment["probe_id"] = "forged-probe"
    (root / "assessments.jsonl").write_text(_canonical(assessment), encoding="utf-8")
    aggregate = json.loads((root / "candidate_aggregates.jsonl").read_text(encoding="utf-8"))
    aggregate["witness_probe_ids"] = ["forged-probe"]
    (root / "candidate_aggregates.jsonl").write_text(_canonical(aggregate), encoding="utf-8")
    _reseal(root)

    assert module.validate_archive(root, authority).reason_code == "REGISTRY_AUTHORITY_MISMATCH"


def test_validator_rejects_invalid_task_three_closure_and_required_strata(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    root = tmp_path / "archive"
    module.write_archive(root, _archive(authority), authority)

    invalid_closure = module.ArchiveRegistryAuthority(
        authority.search_config.model_copy(update={"calibration_probe_inventory_id": "forged-inventory"}),
        authority.inventory,
        authority.suite,
    )
    assert module.validate_archive(root, invalid_closure).reason_code == "INVENTORY_MANIFEST_REFERENCE_MISMATCH"

    missing_strata = module.ArchiveRegistryAuthority(
        authority.search_config.model_copy(update={"required_strata": ()}),
        authority.inventory,
        authority.suite,
    )
    assert module.validate_archive(root, missing_strata).reason_code == "REGISTRY_AUTHORITY_MISMATCH"


def test_validator_rejects_tamper_range_and_call_provenance_breaks(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    root = tmp_path / "archive"
    module.write_archive(root, _archive(authority), authority)
    (root / "calls.jsonl").write_text("{}\n", encoding="utf-8")
    assert module.validate_archive(root, authority).reason_code == "ARCHIVE_HASH_MISMATCH"

    root = tmp_path / "range"
    module.write_archive(root, _archive(authority), authority)
    assessment = json.loads((root / "assessments.jsonl").read_text(encoding="utf-8"))
    assessment["raw_record_ranges"][0]["end"] = 999
    (root / "assessments.jsonl").write_text(_canonical(assessment), encoding="utf-8")
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "RAW_RECORD_RANGE_INVALID"

    root = tmp_path / "provenance"
    module.write_archive(root, _archive(authority), authority)
    (root / "calls.jsonl").write_text(
        (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8"
    )
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "CALL_RELATION_INVALID"


def test_validator_rejects_forged_ranges_provenance_calls_aggregates_and_audit(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    root = tmp_path / "forged-range"
    module.write_archive(root, _archive(authority), authority)
    assessment = json.loads((root / "assessments.jsonl").read_text(encoding="utf-8"))
    assessment["raw_record_ranges"][0].update(start=1, end=2)
    (root / "assessments.jsonl").write_text(_canonical(assessment), encoding="utf-8")
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "RAW_RECORD_RANGE_INVALID"

    root = tmp_path / "missing-provenance"
    module.write_archive(root, _archive(authority), authority)
    assessment = json.loads((root / "assessments.jsonl").read_text(encoding="utf-8"))
    assessment["control_answer_call_provenance_status"] = "missing"
    assessment["control_parsed_response_source_call_id"] = None
    (root / "assessments.jsonl").write_text(_canonical(assessment), encoding="utf-8")
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "PROVENANCE_DISPOSITION_INVALID"

    root = tmp_path / "aux-answer"
    module.write_archive(root, _archive(authority), authority)
    calls = (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    control = json.loads(calls[0])
    control["call_kind"] = "baseline_native_aux"
    (root / "calls.jsonl").write_text(_canonical(control) + calls[1] + "\n", encoding="utf-8")
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "CALL_RELATION_INVALID"

    root = tmp_path / "aggregate-route"
    module.write_archive(root, _archive(authority), authority)
    aggregate = json.loads((root / "candidate_aggregates.jsonl").read_text(encoding="utf-8"))
    aggregate.update(
        assessment_state="not_contradicted",
        final_routing_decision="active",
        final_reason_code="NOT_CONTRADICTED",
    )
    (root / "candidate_aggregates.jsonl").write_text(_canonical(aggregate), encoding="utf-8")
    audit = json.loads((root / "audit" / "audit_labels.jsonl").read_text(encoding="utf-8"))
    audit["routing_decision"].update(
        assessment_state="not_contradicted",
        route_target="active",
        audit_flag=False,
        routing_reason_code="NOT_CONTRADICTED",
    )
    (root / "audit" / "audit_labels.jsonl").write_text(_canonical(audit), encoding="utf-8")
    _reseal(root)
    assert module.validate_archive(root, authority).reason_code == "AGGREGATE_STATE_INVALID"

    root = tmp_path / "audit-tamper"
    module.write_archive(root, _archive(authority), authority)
    audit = json.loads((root / "audit" / "audit_labels.jsonl").read_text(encoding="utf-8"))
    audit["audit_labels"]["candidate_role"] = "forged"
    (root / "audit" / "audit_labels.jsonl").write_text(_canonical(audit), encoding="utf-8")
    assert module.validate_archive(root, authority).reason_code == "AUDIT_HASH_MISMATCH"


def test_writer_cleans_staging_after_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    authority = _authority(module)

    def crash(*_args: object) -> None:
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(module, "_write_jsonl", crash)

    with pytest.raises(RuntimeError, match="injected write failure"):
        module.write_archive(tmp_path / "archive", _archive(authority), authority)
    assert not tuple(tmp_path.glob(".archive.tmp-*"))


def test_aggregate_rejects_zero_denominator_and_mismatched_relations() -> None:
    module = _records()
    archive = _archive(_authority(_module()))
    aggregate = archive.candidate_aggregates[0].model_dump()
    aggregate["n_nominal_attempted_pairs"] = 0
    with pytest.raises(ValidationError, match="NOMINAL_ATTEMPTS_REQUIRED"):
        module.CandidateAggregateRecord.model_validate(aggregate)

    assessment = archive.assessments[0].model_dump()
    assessment["control_parsed_response_source_call_id"] = "wrong-call"
    with pytest.raises(ValidationError, match="ANSWER_CALL_RELATION_INVALID"):
        module.AssessmentRecord.model_validate(assessment)


def test_archive_bytes_ignore_external_output_root(tmp_path: Path) -> None:
    module = _module()
    authority = _authority(module)
    left = tmp_path / "external-a" / "archive"
    right = tmp_path / "external-b" / "archive"

    module.write_archive(left, _archive(authority), authority)
    module.write_archive(right, _archive(authority), authority)

    assert {
        path.relative_to(left).as_posix(): path.read_bytes()
        for path in left.rglob("*") if path.is_file()
    } == {
        path.relative_to(right).as_posix(): path.read_bytes()
        for path in right.rglob("*") if path.is_file()
    }
