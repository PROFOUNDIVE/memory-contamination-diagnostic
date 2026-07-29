"""Task 14 exact BCT contract tables and pure readiness boundary. # noqa: SIZE_OK"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, Final, Literal, Self, TypeAlias, TypeVar, assert_never

from pydantic import BeforeValidator, Field, model_validator

from memcontam.experiment.phase12.filter_challenge.registry import validate_stage
from memcontam.experiment.phase12.filter_challenge.registry_common import (
    NonNegativeInt,
    StrictRegistry,
    StringTuple,
    parse_tuple,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import (
    SearchConfig,
    SelectedPolicy,
)

BCT_EVIDENCE_SCHEMA_VERSION: Final = "filter_challenge_bct_evidence_v1"
BCT_READINESS_SCHEMA_VERSION: Final = "filter_challenge_bct_readiness_v1"
BCT_TEST_IDS: Final = (
    "BCT-FV5-01-CERTIFIED-FALSE", "BCT-FV5-02-CORRECT",
    "BCT-FV5-03-IRRELEVANT", "BCT-FV5-04-ORDINARY-FALSE",
)
BCT_EVIDENCE_FIELD_TUPLE: Final = (
    "schema_version", "test_id", "candidate_class", "fixture_manifest_id",
    "runner_interface_version", "filter_policy_version", "calibration_probe_inventory_id",
    "calibration_probe_inventory_manifest_hash", "operational_probe_suite_id",
    "operational_probe_suite_manifest_hash", "decision_rule_id", "strict_eligibility_status",
    "paired_evaluability_status", "candidate_exposure_status", "witness_status",
    "assessment_state", "routing_decision", "false_quarantine_status", "false_negative_status",
    "route_invariance_status", "activation_domain_status", "behavioral_calls_executed",
    "provider_calls_issued", "cost_preview", "archive_path", "reason_codes",
)
BCT_READINESS_FIELD_TUPLE: Final = (
    "schema_version", "software_interface_status", "software_interface_reason_code",
    "software_blocking_reason_codes", "execution_status", "overall_reason_code",
    "blocking_reason_codes", "provider_authorization_status", "scientific_inventory_status",
    "canonical_patch_status", "behavioral_calls_executed", "provider_calls_issued",
    "family_statuses", "archive_layout", "cost_preview",
)
SOFTWARE_BLOCKING_REASON_CODES: Final = (
    "DOMAIN_SCHEMA_INVALID", "SEARCH_CONFIG_INVALID", "MFT_GATE_FAILED",
    "ARCHIVE_VALIDATION_FAILED", "ANSWER_CALL_PROVENANCE_ENGINEERING_BLOCK",
)
EXECUTION_BLOCKING_REASON_CODES: Final = (
    "SEARCH_CONFIG_PENDING_FREEZE", "SCIENTIFIC_INVENTORY_PENDING_FREEZE",
    "CANONICAL_PATCHES_PENDING", "PROVIDER_CONFIG_DISABLED",
    "PROVIDER_AUTHORIZATION_ABSENT",
)
BCT_RESULT_FIELDS: Final = (
    "strict_eligibility_status", "paired_evaluability_status", "candidate_exposure_status",
    "witness_status", "assessment_state", "routing_decision", "false_quarantine_status",
    "false_negative_status", "route_invariance_status", "activation_domain_status",
)
BCT_ARCHIVE_LAYOUT: Final = (
    "run.json", "bct_evidence.jsonl", "calls.jsonl",
    "public_artifact_manifest.json", "archive_seal.json",
)

BCTTestId: TypeAlias = Literal[
    "BCT-FV5-01-CERTIFIED-FALSE", "BCT-FV5-02-CORRECT",
    "BCT-FV5-03-IRRELEVANT", "BCT-FV5-04-ORDINARY-FALSE",
]
CandidateClass: TypeAlias = Literal[
    "certified_false", "correct", "irrelevant", "ordinary_route_false"
]


class BCTContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class BCTAuthorizationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class BCTFamilyInterface(StrictRegistry):
    test_id: BCTTestId
    candidate_class: CandidateClass
    fixture_manifest_id: str
    runner_interface_version: str
    evidence_interface_version: str
    required_result_fields: StringTuple

    @property
    def result_fields(self) -> tuple[str, ...]:
        return BCT_RESULT_FIELDS


BCT_FAMILY_INTERFACES: Final = (
    BCTFamilyInterface(
        test_id=BCT_TEST_IDS[0], candidate_class="certified_false",
        fixture_manifest_id="bct-fv5-01-certified-false-fixture-manifest-v1",
        runner_interface_version="bct-fv5-01-certified-false-runner-interface-v1",
        evidence_interface_version="bct-fv5-01-certified-false-evidence-interface-v1",
        required_result_fields=(
            "strict_eligibility_status", "paired_evaluability_status", "candidate_exposure_status",
            "witness_status", "assessment_state", "routing_decision", "false_negative_status",
        ),
    ),
    BCTFamilyInterface(
        test_id=BCT_TEST_IDS[1], candidate_class="correct",
        fixture_manifest_id="bct-fv5-02-correct-fixture-manifest-v1",
        runner_interface_version="bct-fv5-02-correct-runner-interface-v1",
        evidence_interface_version="bct-fv5-02-correct-evidence-interface-v1",
        required_result_fields=("paired_evaluability_status", "false_quarantine_status"),
    ),
    BCTFamilyInterface(
        test_id=BCT_TEST_IDS[2], candidate_class="irrelevant",
        fixture_manifest_id="bct-fv5-03-irrelevant-fixture-manifest-v1",
        runner_interface_version="bct-fv5-03-irrelevant-runner-interface-v1",
        evidence_interface_version="bct-fv5-03-irrelevant-evidence-interface-v1",
        required_result_fields=("paired_evaluability_status", "false_quarantine_status"),
    ),
    BCTFamilyInterface(
        test_id=BCT_TEST_IDS[3], candidate_class="ordinary_route_false",
        fixture_manifest_id="bct-fv5-04-ordinary-false-fixture-manifest-v1",
        runner_interface_version="bct-fv5-04-ordinary-false-runner-interface-v1",
        evidence_interface_version="bct-fv5-04-ordinary-false-evidence-interface-v1",
        required_result_fields=(
            "strict_eligibility_status", "witness_status", "assessment_state", "routing_decision",
            "route_invariance_status", "activation_domain_status",
        ),
    ),
)


class AuxiliaryCallCandidate(StrictRegistry):
    auxiliary_call_candidate_id: str
    control_auxiliary_calls: NonNegativeInt
    challenge_auxiliary_calls: NonNegativeInt


class CallPriceRegistry(StrictRegistry):
    price_registry_id: str
    price_per_call: Annotated[float, Field(ge=0)]


class CostCandidateEstimate(StrictRegistry):
    operational_probe_suite_id: str
    replicate_retry_id: str
    auxiliary_call_candidate_id: str
    nominal_provider_calls: NonNegativeInt
    estimated_cost: float | None


CostCandidateTuple: TypeAlias = Annotated[
    tuple[CostCandidateEstimate, ...], BeforeValidator(parse_tuple)
]


class CostPreview(StrictRegistry):
    status: Literal["estimated", "not_estimated"]
    price_registry_id: str | None
    candidate_estimates: CostCandidateTuple


class BCTEvidence(StrictRegistry):
    schema_version: Literal["filter_challenge_bct_evidence_v1"] = BCT_EVIDENCE_SCHEMA_VERSION
    test_id: BCTTestId
    candidate_class: CandidateClass
    fixture_manifest_id: str
    runner_interface_version: str
    filter_policy_version: str
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str
    operational_probe_suite_id: str
    operational_probe_suite_manifest_hash: str
    decision_rule_id: str
    strict_eligibility_status: str | None
    paired_evaluability_status: str | None
    candidate_exposure_status: str | None
    witness_status: str | None
    assessment_state: str | None
    routing_decision: str | None
    false_quarantine_status: str | None
    false_negative_status: str | None
    route_invariance_status: str | None
    activation_domain_status: str | None
    behavioral_calls_executed: bool
    provider_calls_issued: NonNegativeInt
    cost_preview: CostPreview | None
    archive_path: str | None
    reason_codes: StringTuple

    @property
    def result_fields(self) -> tuple[str, ...]:
        return BCT_RESULT_FIELDS

    @property
    def required_result_fields(self) -> tuple[str, ...]:
        return _family_for(self.test_id).required_result_fields

    @model_validator(mode="after")
    def _validate_family(self) -> Self:
        _validate_bct_family(self)
        return self


class SoftwareInterfaceChecks(StrictRegistry):
    domain_schema_valid: bool
    search_config_valid: bool
    mft_gate_passed: bool
    archive_validation_passed: bool
    answer_call_provenance_engineering_ready: bool


class SoftwareInterfaceReadiness(StrictRegistry):
    software_interface_status: Literal["ready", "not_ready"]
    software_interface_reason_code: str | None
    software_blocking_reason_codes: StringTuple


class ExecutionPrerequisites(StrictRegistry):
    schema_version: Literal["filter_challenge_bct_execution_prerequisites_v1"]
    prerequisites_id: str
    evidence_layer: Literal["build"]
    scientific_result: Literal[False]
    fixture_only: Literal[True]
    search_config_frozen: bool
    inventory_frozen: bool
    canonical_patch_status: Literal["applied", "pending_before_provider_backed_pilot_b"]
    provider_config_enabled: bool
    runtime_authorization_present: bool


@dataclass(frozen=True, slots=True)
class ExecutionPreflightRequest:
    search_config: SearchConfig
    selected_policy: SelectedPolicy | None
    stage: Literal["build", "pilot_b", "main"]
    prerequisites: ExecutionPrerequisites


class ExecutionPreflight(StrictRegistry):
    stage: Literal["build", "pilot_b", "main"]
    execution_status: Literal["authorized", "blocked", "not_evaluated"]
    overall_reason_code: str | None
    blocking_reason_codes: StringTuple
    provider_authorization_status: Literal["authorized", "absent"]
    scientific_inventory_status: Literal["frozen", "pending_freeze"]
    canonical_patch_status: Literal["applied", "pending_before_provider_backed_pilot_b"]

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        match self.execution_status:
            case "authorized":
                if (
                    self.overall_reason_code is not None
                    or self.blocking_reason_codes
                    or self.provider_authorization_status != "authorized"
                    or self.scientific_inventory_status != "frozen"
                    or self.canonical_patch_status != "applied"
                ):
                    raise BCTAuthorizationError("EXECUTION_AUTHORIZATION_INCOHERENT")
            case "blocked":
                if (
                    self.overall_reason_code is None
                    or not self.blocking_reason_codes
                    or self.blocking_reason_codes[0] != self.overall_reason_code
                ):
                    raise BCTAuthorizationError("EXECUTION_BLOCKED_STATE_INCOHERENT")
            case "not_evaluated":
                if self.overall_reason_code is None or self.blocking_reason_codes:
                    raise BCTAuthorizationError("EXECUTION_NOT_EVALUATED_STATE_INCOHERENT")
            case unreachable:
                assert_never(unreachable)
        return self


class BCTFamilyStatus(StrictRegistry):
    test_id: BCTTestId
    status: Literal["not_executed"]


FamilyStatusTuple: TypeAlias = Annotated[tuple[BCTFamilyStatus, ...], BeforeValidator(parse_tuple)]


class BCTReadiness(StrictRegistry):
    schema_version: Literal["filter_challenge_bct_readiness_v1"] = BCT_READINESS_SCHEMA_VERSION
    software_interface_status: Literal["ready", "not_ready"]
    software_interface_reason_code: str | None
    software_blocking_reason_codes: StringTuple
    execution_status: Literal["authorized", "blocked", "not_evaluated"]
    overall_reason_code: str | None
    blocking_reason_codes: StringTuple
    provider_authorization_status: Literal["authorized", "absent"]
    scientific_inventory_status: Literal["frozen", "pending_freeze"]
    canonical_patch_status: Literal["applied", "pending_before_provider_backed_pilot_b"]
    behavioral_calls_executed: Literal[False]
    provider_calls_issued: Literal[0]
    family_statuses: FamilyStatusTuple
    archive_layout: StringTuple
    cost_preview: CostPreview


def _family_for(test_id: BCTTestId) -> BCTFamilyInterface:
    return next(family for family in BCT_FAMILY_INTERFACES if family.test_id == test_id)


def _validate_bct_family(evidence: BCTEvidence) -> None:
    family = _family_for(evidence.test_id)
    if (
        evidence.candidate_class != family.candidate_class
        or evidence.fixture_manifest_id != family.fixture_manifest_id
        or evidence.runner_interface_version != family.runner_interface_version
    ):
        raise BCTContractError("BCT_INTERFACE_IDENTITY_MISMATCH")
    if any(getattr(evidence, field) is None for field in family.required_result_fields):
        raise BCTContractError("BCT_REQUIRED_RESULT_MISSING")
    if any(
        getattr(evidence, field) is not None
        for field in BCT_RESULT_FIELDS if field not in family.required_result_fields
    ):
        raise BCTContractError("BCT_NONAPPLICABLE_RESULT_PRESENT")


def validate_bct_evidence(evidence: BCTEvidence) -> BCTEvidence:
    _validate_bct_family(evidence)
    return evidence


def build_cost_preview(
    search_config: SearchConfig,
    auxiliary_candidates: tuple[AuxiliaryCallCandidate, ...],
    price_registry: CallPriceRegistry | None,
) -> CostPreview:
    estimates = tuple(
        CostCandidateEstimate(
            operational_probe_suite_id=suite.operational_probe_suite_id,
            replicate_retry_id=retry.replicate_retry_id,
            auxiliary_call_candidate_id=auxiliary.auxiliary_call_candidate_id,
            nominal_provider_calls=(
                len(BCT_TEST_IDS) * len(suite.probe_ids) * suite.replicates_per_probe
                * (2 + retry.control_retry_limit + retry.challenge_retry_limit
                   + auxiliary.control_auxiliary_calls + auxiliary.challenge_auxiliary_calls)
            ),
            estimated_cost=None if price_registry is None else (
                len(BCT_TEST_IDS) * len(suite.probe_ids) * suite.replicates_per_probe
                * (2 + retry.control_retry_limit + retry.challenge_retry_limit
                   + auxiliary.control_auxiliary_calls + auxiliary.challenge_auxiliary_calls)
                * price_registry.price_per_call
            ),
        )
        for suite in search_config.suite_candidates
        for retry in search_config.replicate_retry_candidates
        for auxiliary in auxiliary_candidates
    )
    return CostPreview(
        status="not_estimated" if price_registry is None else "estimated",
        price_registry_id=None if price_registry is None else price_registry.price_registry_id,
        candidate_estimates=estimates,
    )


def evaluate_software_interface_readiness(
    checks: SoftwareInterfaceChecks,
) -> SoftwareInterfaceReadiness:
    blockers = tuple(
        code for passed, code in zip(checks.model_dump().values(), SOFTWARE_BLOCKING_REASON_CODES, strict=True)
        if not passed
    )
    return SoftwareInterfaceReadiness(
        software_interface_status="ready" if not blockers else "not_ready",
        software_interface_reason_code=blockers[0] if blockers else None,
        software_blocking_reason_codes=blockers,
    )


def evaluate_execution_preflight(
    software: SoftwareInterfaceReadiness,
    request: ExecutionPreflightRequest,
) -> ExecutionPreflight:
    prerequisites = request.prerequisites
    provider_status: Literal["authorized", "absent"] = (
        "authorized" if prerequisites.runtime_authorization_present else "absent"
    )
    inventory_status: Literal["frozen", "pending_freeze"] = (
        "frozen" if prerequisites.inventory_frozen else "pending_freeze"
    )
    if software.software_interface_status != "ready":
        return ExecutionPreflight(
            stage=request.stage, execution_status="not_evaluated",
            overall_reason_code=software.software_interface_reason_code,
            blocking_reason_codes=(), provider_authorization_status=provider_status,
            scientific_inventory_status=inventory_status,
            canonical_patch_status=prerequisites.canonical_patch_status,
        )
    stage_gate = validate_stage(
        request.search_config, request.selected_policy, stage=request.stage
    )
    if stage_gate.reason_code is not None:
        return ExecutionPreflight(
            stage=request.stage, execution_status="blocked", overall_reason_code=stage_gate.reason_code,
            blocking_reason_codes=(stage_gate.reason_code,),
            provider_authorization_status=provider_status,
            scientific_inventory_status=inventory_status,
            canonical_patch_status=prerequisites.canonical_patch_status,
        )
    blockers = tuple(
        code for passed, code in (
            (prerequisites.search_config_frozen, EXECUTION_BLOCKING_REASON_CODES[0]),
            (prerequisites.inventory_frozen, EXECUTION_BLOCKING_REASON_CODES[1]),
            (prerequisites.canonical_patch_status == "applied", EXECUTION_BLOCKING_REASON_CODES[2]),
            (prerequisites.provider_config_enabled, EXECUTION_BLOCKING_REASON_CODES[3]),
            (prerequisites.runtime_authorization_present, EXECUTION_BLOCKING_REASON_CODES[4]),
        ) if not passed
    )
    return ExecutionPreflight(
        stage=request.stage, execution_status="blocked" if blockers else "authorized",
        overall_reason_code=blockers[0] if blockers else None,
        blocking_reason_codes=blockers,
        provider_authorization_status=provider_status,
        scientific_inventory_status=inventory_status,
        canonical_patch_status=prerequisites.canonical_patch_status,
    )


def build_readiness(
    software: SoftwareInterfaceReadiness,
    execution: ExecutionPreflight,
    cost_preview: CostPreview,
) -> BCTReadiness:
    software_blocked = software.software_interface_status != "ready"
    return BCTReadiness(
        schema_version=BCT_READINESS_SCHEMA_VERSION,
        software_interface_status=software.software_interface_status,
        software_interface_reason_code=software.software_interface_reason_code,
        software_blocking_reason_codes=software.software_blocking_reason_codes,
        execution_status=execution.execution_status,
        overall_reason_code=(
            software.software_interface_reason_code if software_blocked else execution.overall_reason_code
        ),
        blocking_reason_codes=(
            software.software_blocking_reason_codes if software_blocked else execution.blocking_reason_codes
        ),
        provider_authorization_status=execution.provider_authorization_status,
        scientific_inventory_status=execution.scientific_inventory_status,
        canonical_patch_status=execution.canonical_patch_status,
        behavioral_calls_executed=False,
        provider_calls_issued=0,
        family_statuses=tuple(
            BCTFamilyStatus(test_id=family.test_id, status="not_executed")
            for family in BCT_FAMILY_INTERFACES
        ),
        archive_layout=BCT_ARCHIVE_LAYOUT,
        cost_preview=cost_preview,
    )


_Client = TypeVar("_Client")


def authorize_client_construction(
    software: SoftwareInterfaceReadiness,
    execution: ExecutionPreflight,
    client_factory: Callable[[], _Client],
    *,
    stage: Literal["build", "pilot_b", "main"],
) -> _Client:
    if software.software_interface_status != "ready":
        raise BCTAuthorizationError(software.software_interface_reason_code or "SOFTWARE_NOT_READY")
    if execution.execution_status != "authorized":
        raise BCTAuthorizationError(execution.overall_reason_code or "EXECUTION_NOT_AUTHORIZED")
    if execution.stage != stage:
        raise BCTAuthorizationError("EXECUTION_STAGE_MISMATCH")
    return client_factory()
