"""Task 14 exact BCT contract and authorization matrix. # noqa: SIZE_OK"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TypeAlias

import pytest
from pydantic import ValidationError

import memcontam.clients.factory as client_factory_module
from memcontam.clients.config import ProviderConfig
from memcontam.experiment.phase12.filter_challenge.bct import (
    BCT_ARCHIVE_LAYOUT,
    BCT_EVIDENCE_FIELD_TUPLE,
    BCT_EVIDENCE_SCHEMA_VERSION,
    BCT_FAMILY_INTERFACES,
    BCT_READINESS_FIELD_TUPLE,
    BCT_READINESS_SCHEMA_VERSION,
    BCT_TEST_IDS,
    EXECUTION_BLOCKING_REASON_CODES,
    SOFTWARE_BLOCKING_REASON_CODES,
    AuxiliaryCallCandidate,
    BCTAuthorizationError,
    BCTContractError,
    BCTEvidence,
    BCTReadiness,
    CallPriceRegistry,
    ExecutionPreflight,
    ExecutionPreflightRequest,
    ExecutionPrerequisites,
    SoftwareInterfaceChecks,
    authorize_client_construction,
    build_cost_preview,
    build_readiness,
    evaluate_execution_preflight,
    evaluate_software_interface_readiness,
    validate_bct_evidence,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import (
    SearchConfig,
    SelectedPolicy,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase12" / "filter_v5"
EXPECTED_EVIDENCE_FIELDS = (
    "schema_version", "test_id", "candidate_class", "fixture_manifest_id",
    "runner_interface_version", "filter_policy_version", "calibration_probe_inventory_id",
    "calibration_probe_inventory_manifest_hash", "operational_probe_suite_id",
    "operational_probe_suite_manifest_hash", "decision_rule_id", "strict_eligibility_status",
    "paired_evaluability_status", "candidate_exposure_status", "witness_status",
    "assessment_state", "routing_decision", "false_quarantine_status", "false_negative_status",
    "route_invariance_status", "activation_domain_status", "behavioral_calls_executed",
    "provider_calls_issued", "cost_preview", "archive_path", "reason_codes",
)
EXPECTED_READINESS_FIELDS = (
    "schema_version", "software_interface_status", "software_interface_reason_code",
    "software_blocking_reason_codes", "execution_status", "overall_reason_code",
    "blocking_reason_codes", "provider_authorization_status", "scientific_inventory_status",
    "canonical_patch_status", "behavioral_calls_executed", "provider_calls_issued",
    "family_statuses", "archive_layout", "cost_preview",
)
EvidencePayloadValue: TypeAlias = str | bool | int | None | list[str]


def _load_yaml(name: str):
    safe_load = getattr(importlib.import_module("yaml"), "safe_load")
    return safe_load((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _search() -> SearchConfig:
    return SearchConfig.model_validate(_load_yaml("FilterChallengeSearchConfig.yaml"))


def _policy() -> SelectedPolicy:
    return SelectedPolicy.model_validate(_load_yaml("FilterChallengeSelectedPolicy.yaml"))


def _prerequisites() -> ExecutionPrerequisites:
    return ExecutionPrerequisites.model_validate_json(
        (FIXTURE_ROOT / "bct_execution_prerequisites.json").read_text(encoding="utf-8")
    )


def _software(**updates: bool):
    checks = SoftwareInterfaceChecks(
        domain_schema_valid=True,
        search_config_valid=True,
        mft_gate_passed=True,
        archive_validation_passed=True,
        answer_call_provenance_engineering_ready=True,
    )
    return evaluate_software_interface_readiness(checks.model_copy(update=updates))


def _evidence_payload(test_id: str) -> dict[str, EvidencePayloadValue]:
    family = next(item for item in BCT_FAMILY_INTERFACES if item.test_id == test_id)
    statuses: dict[str, str | None] = {field: None for field in family.result_fields}
    statuses.update({field: "registered" for field in family.required_result_fields})
    return {
        "schema_version": BCT_EVIDENCE_SCHEMA_VERSION,
        "test_id": family.test_id,
        "candidate_class": family.candidate_class,
        "fixture_manifest_id": family.fixture_manifest_id,
        "runner_interface_version": family.runner_interface_version,
        "filter_policy_version": "verifier-paired-challenge-v1",
        "calibration_probe_inventory_id": "inventory-v1",
        "calibration_probe_inventory_manifest_hash": "a" * 64,
        "operational_probe_suite_id": "suite-v1",
        "operational_probe_suite_manifest_hash": "b" * 64,
        "decision_rule_id": "rule-v1",
        **statuses,
        "behavioral_calls_executed": False,
        "provider_calls_issued": 0,
        "cost_preview": None,
        "archive_path": None,
        "reason_codes": [],
    }


def test_literal_registries_schemas_and_interface_identities_are_exact() -> None:
    assert BCT_TEST_IDS == (
        "BCT-FV5-01-CERTIFIED-FALSE", "BCT-FV5-02-CORRECT",
        "BCT-FV5-03-IRRELEVANT", "BCT-FV5-04-ORDINARY-FALSE",
    )
    assert BCT_EVIDENCE_SCHEMA_VERSION == "filter_challenge_bct_evidence_v1"
    assert BCT_EVIDENCE_FIELD_TUPLE == tuple(BCTEvidence.model_fields) == EXPECTED_EVIDENCE_FIELDS
    assert BCT_READINESS_SCHEMA_VERSION == "filter_challenge_bct_readiness_v1"
    assert BCT_READINESS_FIELD_TUPLE == tuple(BCTReadiness.model_fields) == EXPECTED_READINESS_FIELDS
    assert SOFTWARE_BLOCKING_REASON_CODES == (
        "DOMAIN_SCHEMA_INVALID", "SEARCH_CONFIG_INVALID", "MFT_GATE_FAILED",
        "ARCHIVE_VALIDATION_FAILED", "ANSWER_CALL_PROVENANCE_ENGINEERING_BLOCK",
    )
    assert EXECUTION_BLOCKING_REASON_CODES == (
        "SEARCH_CONFIG_PENDING_FREEZE", "SCIENTIFIC_INVENTORY_PENDING_FREEZE",
        "CANONICAL_PATCHES_PENDING", "PROVIDER_CONFIG_DISABLED",
        "PROVIDER_AUTHORIZATION_ABSENT",
    )
    identities = tuple(
        identity
        for family in BCT_FAMILY_INTERFACES
        for identity in (
            family.fixture_manifest_id,
            family.runner_interface_version,
            family.evidence_interface_version,
        )
    )
    assert tuple(family.test_id for family in BCT_FAMILY_INTERFACES) == BCT_TEST_IDS
    assert len(set(identities)) == 12
    assert tuple(
        (
            family.candidate_class,
            family.fixture_manifest_id,
            family.runner_interface_version,
            family.evidence_interface_version,
        )
        for family in BCT_FAMILY_INTERFACES
    ) == (
        (
            "certified_false", "bct-fv5-01-certified-false-fixture-manifest-v1",
            "bct-fv5-01-certified-false-runner-interface-v1",
            "bct-fv5-01-certified-false-evidence-interface-v1",
        ),
        (
            "correct", "bct-fv5-02-correct-fixture-manifest-v1",
            "bct-fv5-02-correct-runner-interface-v1",
            "bct-fv5-02-correct-evidence-interface-v1",
        ),
        (
            "irrelevant", "bct-fv5-03-irrelevant-fixture-manifest-v1",
            "bct-fv5-03-irrelevant-runner-interface-v1",
            "bct-fv5-03-irrelevant-evidence-interface-v1",
        ),
        (
            "ordinary_route_false", "bct-fv5-04-ordinary-false-fixture-manifest-v1",
            "bct-fv5-04-ordinary-false-runner-interface-v1",
            "bct-fv5-04-ordinary-false-evidence-interface-v1",
        ),
    )


def test_family_evidence_requires_applicable_fields_and_explicit_nulls() -> None:
    for test_id in BCT_TEST_IDS:
        payload = _evidence_payload(test_id)
        evidence = BCTEvidence.model_validate(payload)
        assert validate_bct_evidence(evidence) is evidence
        assert tuple(evidence.model_dump()) == EXPECTED_EVIDENCE_FIELDS
        for field in set(evidence.result_fields) - set(evidence.required_result_fields):
            assert evidence.model_dump()[field] is None

    missing_null = _evidence_payload("BCT-FV5-02-CORRECT")
    del missing_null["witness_status"]
    with pytest.raises(ValidationError):
        BCTEvidence.model_validate(missing_null)
    missing_required = BCTEvidence.model_validate(_evidence_payload("BCT-FV5-02-CORRECT"))
    with pytest.raises(BCTContractError, match="BCT_REQUIRED_RESULT_MISSING"):
        validate_bct_evidence(missing_required.model_copy(update={"false_quarantine_status": None}))
    with pytest.raises(BCTContractError, match="BCT_NONAPPLICABLE_RESULT_PRESENT"):
        validate_bct_evidence(missing_required.model_copy(update={"witness_status": "witness"}))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("false_quarantine_status", None, "BCT_REQUIRED_RESULT_MISSING"),
        ("witness_status", "must-be-null", "BCT_NONAPPLICABLE_RESULT_PRESENT"),
    ),
)
def test_direct_evidence_validation_rejects_family_applicability_bypasses(
    field: str, value: str | None, code: str
) -> None:
    payload = _evidence_payload("BCT-FV5-02-CORRECT")
    payload[field] = value

    with pytest.raises(ValidationError, match=code):
        BCTEvidence.model_validate(payload)


def test_cost_preview_uses_only_registered_cross_product_and_optional_explicit_price() -> None:
    auxiliary = (
        AuxiliaryCallCandidate(
            auxiliary_call_candidate_id="aux-v1",
            control_auxiliary_calls=1,
            challenge_auxiliary_calls=2,
        ),
    )
    priced = build_cost_preview(
        _search(), auxiliary, CallPriceRegistry(price_registry_id="price-v1", price_per_call=0.25)
    )
    symbolic = {
        (item.operational_probe_suite_id, item.replicate_retry_id):
        (item.nominal_provider_calls, item.estimated_cost)
        for item in priced.candidate_estimates
    }
    assert priced.status == "estimated"
    assert symbolic == {
        ("synthetic-build-suite-balanced-v1", "synthetic-build-retry-minimal-v1"): (120, 30.0),
        ("synthetic-build-suite-balanced-v1", "synthetic-build-retry-bounded-v1"): (168, 42.0),
        ("synthetic-build-suite-expanded-v1", "synthetic-build-retry-minimal-v1"): (180, 45.0),
        ("synthetic-build-suite-expanded-v1", "synthetic-build-retry-bounded-v1"): (252, 63.0),
    }
    assert tuple(
        (item.operational_probe_suite_id, item.replicate_retry_id)
        for item in priced.candidate_estimates
    ) == (
        ("synthetic-build-suite-balanced-v1", "synthetic-build-retry-minimal-v1"),
        ("synthetic-build-suite-balanced-v1", "synthetic-build-retry-bounded-v1"),
        ("synthetic-build-suite-expanded-v1", "synthetic-build-retry-minimal-v1"),
        ("synthetic-build-suite-expanded-v1", "synthetic-build-retry-bounded-v1"),
    )
    unpriced = build_cost_preview(_search(), auxiliary, None)
    assert unpriced.status == "not_estimated"
    assert unpriced.price_registry_id is None
    assert all(item.estimated_cost is None for item in unpriced.candidate_estimates)


@pytest.mark.parametrize(
    ("field", "code"),
    tuple(zip(SoftwareInterfaceChecks.model_fields, SOFTWARE_BLOCKING_REASON_CODES, strict=True)),
)
def test_software_readiness_reports_each_failure_independently(field: str, code: str) -> None:
    result = _software(**{field: False})
    assert result.software_interface_status == "not_ready"
    assert result.software_interface_reason_code == code
    assert result.software_blocking_reason_codes == (code,)


def test_blockers_accumulate_in_authority_order_and_build_stays_zero_call() -> None:
    software = _software()
    execution = evaluate_execution_preflight(
        software,
        ExecutionPreflightRequest(
            search_config=_search(), selected_policy=None, stage="build", prerequisites=_prerequisites()
        ),
    )
    preview = build_cost_preview(
        _search(),
        (AuxiliaryCallCandidate(
            auxiliary_call_candidate_id="aux-v1", control_auxiliary_calls=0,
            challenge_auxiliary_calls=0,
        ),),
        None,
    )
    readiness = build_readiness(software, execution, preview)

    assert software.software_interface_status == "ready"
    assert execution.execution_status == "blocked"
    assert execution.overall_reason_code == EXECUTION_BLOCKING_REASON_CODES[0]
    assert execution.blocking_reason_codes == EXECUTION_BLOCKING_REASON_CODES
    assert tuple(readiness.model_dump()) == EXPECTED_READINESS_FIELDS
    assert readiness.behavioral_calls_executed is False
    assert readiness.provider_calls_issued == 0
    assert readiness.provider_authorization_status == "absent"
    assert readiness.scientific_inventory_status == "pending_freeze"
    assert readiness.canonical_patch_status == "pending_before_provider_backed_pilot_b"
    assert tuple(status.status for status in readiness.family_statuses) == ("not_executed",) * 4
    assert readiness.archive_layout == BCT_ARCHIVE_LAYOUT


@pytest.mark.parametrize(
    ("field", "blocked_value"),
    (
        ("search_config_frozen", False),
        ("inventory_frozen", False),
        ("canonical_patch_status", "pending_before_provider_backed_pilot_b"),
        ("provider_config_enabled", False),
        ("runtime_authorization_present", False),
    ),
)
def test_execution_preflight_reports_each_prerequisite_blocker(
    field: str, blocked_value: bool | str
) -> None:
    ready = _prerequisites().model_copy(
        update={
            "search_config_frozen": True,
            "inventory_frozen": True,
            "canonical_patch_status": "applied",
            "provider_config_enabled": True,
            "runtime_authorization_present": True,
            field: blocked_value,
        }
    )
    result = evaluate_execution_preflight(
        _software(),
        ExecutionPreflightRequest(
            search_config=_search(), selected_policy=None, stage="pilot_b", prerequisites=ready
        ),
    )
    assert result.execution_status == "blocked"
    assert len(result.blocking_reason_codes) == 1
    constructed = 0

    def forbidden_constructor() -> str:
        nonlocal constructed
        constructed += 1
        return "client"

    with pytest.raises(BCTAuthorizationError):
        authorize_client_construction(
            _software(), result, forbidden_constructor, stage="pilot_b"
        )
    assert constructed == 0


@pytest.mark.parametrize(
    "update",
    (
        {"provider_authorization_status": "absent"},
        {"scientific_inventory_status": "pending_freeze"},
        {"canonical_patch_status": "pending_before_provider_backed_pilot_b"},
        {"blocking_reason_codes": ("PROVIDER_AUTHORIZATION_ABSENT",)},
        {"overall_reason_code": "PROVIDER_AUTHORIZATION_ABSENT"},
    ),
)
def test_direct_preflight_validation_rejects_incoherent_authorized_states(
    update: dict[str, str | tuple[str, ...]],
) -> None:
    payload = {
        "stage": "main",
        "execution_status": "authorized",
        "overall_reason_code": None,
        "blocking_reason_codes": (),
        "provider_authorization_status": "authorized",
        "scientific_inventory_status": "frozen",
        "canonical_patch_status": "applied",
        **update,
    }

    with pytest.raises(ValidationError, match="EXECUTION_AUTHORIZATION_INCOHERENT"):
        ExecutionPreflight.model_validate(payload)


def test_main_requires_selected_policy_before_authorization_and_constructor_is_unreachable() -> None:
    prerequisites = _prerequisites().model_copy(
        update={
            "search_config_frozen": True,
            "inventory_frozen": True,
            "canonical_patch_status": "applied",
            "provider_config_enabled": True,
            "runtime_authorization_present": True,
        }
    )
    software = _software()
    build = evaluate_execution_preflight(
        software,
        ExecutionPreflightRequest(
            search_config=_search(), selected_policy=None, stage="build", prerequisites=prerequisites
        ),
    )
    pilot_b = evaluate_execution_preflight(
        software,
        ExecutionPreflightRequest(
            search_config=_search(), selected_policy=None, stage="pilot_b", prerequisites=prerequisites
        ),
    )
    main_blocked = evaluate_execution_preflight(
        software,
        ExecutionPreflightRequest(
            search_config=_search(), selected_policy=None, stage="main", prerequisites=prerequisites
        ),
    )
    main_ready = evaluate_execution_preflight(
        software,
        ExecutionPreflightRequest(
            search_config=_search(), selected_policy=_policy(), stage="main", prerequisites=prerequisites
        ),
    )
    constructed = 0

    def forbidden_constructor() -> str:
        nonlocal constructed
        constructed += 1
        return "client"

    assert build.execution_status == pilot_b.execution_status == main_ready.execution_status == "authorized"
    assert main_blocked.overall_reason_code == "SELECTED_POLICY_REQUIRED"
    with pytest.raises(BCTAuthorizationError, match="SELECTED_POLICY_REQUIRED"):
        authorize_client_construction(
            software, main_blocked, forbidden_constructor, stage="main"
        )
    with pytest.raises(BCTAuthorizationError, match="DOMAIN_SCHEMA_INVALID"):
        authorize_client_construction(
            _software(domain_schema_valid=False), main_ready, forbidden_constructor, stage="main"
        )
    assert constructed == 0
    assert json.loads(build_readiness(
        software,
        evaluate_execution_preflight(
            software,
            ExecutionPreflightRequest(
                search_config=_search(), selected_policy=None, stage="build",
                prerequisites=_prerequisites(),
            ),
        ),
        build_cost_preview(
            _search(),
            (AuxiliaryCallCandidate(
                auxiliary_call_candidate_id="aux-v1", control_auxiliary_calls=0,
                challenge_auxiliary_calls=0,
            ),),
            None,
        ),
    ).model_dump_json())["provider_calls_issued"] == 0


def test_build_preflight_cannot_construct_a_main_provider_factory_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prerequisites = ExecutionPrerequisites.model_validate(
        _prerequisites().model_dump()
        | {
            "search_config_frozen": True,
            "inventory_frozen": True,
            "canonical_patch_status": "applied",
            "provider_config_enabled": True,
            "runtime_authorization_present": True,
        }
    )
    software = _software()
    build = evaluate_execution_preflight(
        software,
        ExecutionPreflightRequest(
            search_config=_search(), selected_policy=None, stage="build", prerequisites=prerequisites
        ),
    )
    constructed = 0

    def forbidden_provider(*_args, **_kwargs) -> str:
        nonlocal constructed
        constructed += 1
        return "client"

    monkeypatch.setattr(client_factory_module, "OpenAICompatibleClient", forbidden_provider)

    with pytest.raises(BCTAuthorizationError, match="EXECUTION_STAGE_MISMATCH"):
        authorize_client_construction(
            software,
            build,
            lambda: client_factory_module.build_llm_client(
                ProviderConfig(provider="openai_compatible", live_calls_enabled=True),
                stage="main",
                execution_class="live",
                allow_live_calls=True,
            ),
            stage="main",
        )
    assert constructed == 0
