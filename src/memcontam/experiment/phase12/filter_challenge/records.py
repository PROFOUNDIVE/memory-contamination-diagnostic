from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from memcontam.experiment.phase12.filter_challenge.audit import PostRouteAuditJoin
from memcontam.experiment.phase12.filter_challenge.registry_common import parse_tuple


ASSESSMENT_SCHEMA_VERSION: Final = "filter_challenge_assessment_record_v1"
AGGREGATE_SCHEMA_VERSION: Final = "filter_challenge_candidate_aggregate_v1"
ASSESSMENT_FIELD_TUPLE: Final = (
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
AGGREGATE_FIELD_TUPLE: Final = (
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
PUBLIC_AUDIT_KEYS: Final = frozenset(
    {
        "candidate_role", "correctness_label", "irrelevance_label", "B_star_membership",
        "is_injected", "origin_class", "injection_event_id", "treatment_arm",
        "future_main_outcomes", "future_suffix_outcomes",
    }
)


class FilterChallengeArchiveError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


StringTuple = Annotated[tuple[str, ...], BeforeValidator(parse_tuple)]


def normalize_archive_path(value: str) -> str:
    if not value:
        raise FilterChallengeArchiveError("ARCHIVE_PATH_EMPTY")
    if "\\" in value:
        raise FilterChallengeArchiveError("ARCHIVE_PATH_BACKSLASH")
    if PureWindowsPath(value).drive:
        raise FilterChallengeArchiveError("ARCHIVE_PATH_DRIVE")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise FilterChallengeArchiveError("ARCHIVE_PATH_ABSOLUTE")
    if value == ".":
        raise FilterChallengeArchiveError("ARCHIVE_PATH_DOT")
    if ".." in path.parts:
        raise FilterChallengeArchiveError("ARCHIVE_PATH_PARENT")
    if str(path) != value:
        raise FilterChallengeArchiveError("ARCHIVE_PATH_NORMALIZATION")
    return value


class RawRecordRange(_StrictRecord):
    path: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return normalize_archive_path(value)

    @model_validator(mode="after")
    def _validate_half_open(self) -> Self:
        if self.end <= self.start:
            raise FilterChallengeArchiveError("RAW_RECORD_RANGE_INVALID")
        return self


class AssessmentRecord(_StrictRecord):
    schema_version: Literal["filter_challenge_assessment_record_v1"] = ASSESSMENT_SCHEMA_VERSION
    filter_assessment_id: str
    evidence_layer: Literal["build"] = "build"
    run_family: Literal["filter_challenge_assessment"] = "filter_challenge_assessment"
    record_kind: Literal["paired_pre_admission_probe"] = "paired_pre_admission_probe"
    filter_policy_version: Literal["verifier-paired-challenge-v1"]
    policy_family: Literal["verifier_backed_paired_challenge"]
    decision_rule_id: str
    failure_mode_id: Literal["fail_open"]
    candidate_entry_id: str
    candidate_native_kind: str
    candidate_domain_status: Literal["challenge_routable_v1", "unsupported"]
    policy_activation_checkpoint_id: str
    baseline_family: Literal["full_history", "rag_frozen", "bot_style", "reflexion_style"]
    rag_mode: Literal["frozen", "not_applicable"]
    source_checkpoint_id: str
    source_active_state_hash: str
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str
    operational_probe_suite_id: str
    operational_probe_suite_manifest_hash: str
    probe_map_version: Literal["phase12-filter-probe-map-v1"]
    challenge_suite_key: str
    probe_id: str
    probe_source_span_ids: StringTuple
    replicate_id: str
    control_trial_id: str
    control_call_id: str
    control_answer_call_id: str
    control_parsed_response_source_call_id: str | None
    control_answer_call_provenance_status: Literal[
        "explicit_matched", "missing", "ambiguous", "historically_reconstructed", "mismatched"
    ]
    challenge_trial_id: str
    challenge_call_id: str
    challenge_answer_call_id: str
    challenge_parsed_response_source_call_id: str | None
    challenge_answer_call_provenance_status: Literal[
        "explicit_matched", "missing", "ambiguous", "historically_reconstructed", "mismatched"
    ]
    paired_execution_identity_status: Literal["matched", "mismatched"]
    control_prompt_hash: str
    challenge_prompt_hash: str
    control_raw_output_hash: str | None
    challenge_raw_output_hash: str | None
    control_provider_status: str
    challenge_provider_status: str
    control_raw_parse_status: str
    challenge_raw_parse_status: str
    control_canonicalizer_version: str | None
    challenge_canonicalizer_version: str | None
    control_canonicalized_output_hash: str | None
    challenge_canonicalized_output_hash: str | None
    control_canonicalized_parse_status: str
    challenge_canonicalized_parse_status: str
    control_verifier_status: str
    challenge_verifier_status: str
    control_verifier_result: bool | None
    challenge_verifier_result: bool | None
    control_probe_eligibility_state: Literal[
        "strict_primary_eligible", "canonicalization_sensitivity_eligible", "ineligible"
    ]
    candidate_final_context_inclusion: bool
    candidate_final_context_source_ids: StringTuple
    noncandidate_displacement_ids: StringTuple
    probe_disposition: Literal["witness", "evaluated_no_witness", "not_evaluable"]
    probe_reason_code: str
    assessment_state: Literal["contradicted", "not_contradicted", "not_evaluable"]
    final_routing_decision: Literal["quarantine", "active"]
    routing_reason_code: Literal["CONTRADICTED", "NOT_CONTRADICTED", "FAIL_OPEN_NOT_EVALUABLE"]
    retry_count_control: int = Field(ge=0)
    retry_count_challenge: int = Field(ge=0)
    baseline_native_aux_call_ids_control: StringTuple
    baseline_native_aux_call_ids_challenge: StringTuple
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    monetary_cost: float = Field(ge=0)
    control_latency_ms: int = Field(ge=0)
    challenge_latency_ms: int = Field(ge=0)
    canonicalization_latency_ms: int = Field(ge=0)
    total_latency_ms: int = Field(ge=0)
    cache_key_control: str
    archive_path: str
    raw_record_ranges: Annotated[tuple[RawRecordRange, ...], BeforeValidator(parse_tuple)]

    @field_validator("archive_path")
    @classmethod
    def _validate_archive_path(cls, value: str) -> str:
        return normalize_archive_path(value)

    @model_validator(mode="after")
    def _validate_record(self) -> Self:
        if self.archive_path != "assessments.jsonl":
            raise FilterChallengeArchiveError("ASSESSMENT_ARCHIVE_PATH_INVALID")
        if not self.raw_record_ranges or self.control_call_id != self.control_answer_call_id:
            raise FilterChallengeArchiveError("ANSWER_CALL_RELATION_INVALID")
        if self.challenge_call_id != self.challenge_answer_call_id:
            raise FilterChallengeArchiveError("ANSWER_CALL_RELATION_INVALID")
        if self.control_answer_call_id == self.challenge_answer_call_id:
            raise FilterChallengeArchiveError("ANSWER_CALL_RELATION_INVALID")
        if self.control_answer_call_provenance_status == "explicit_matched" and (
            self.control_parsed_response_source_call_id != self.control_answer_call_id
        ):
            raise FilterChallengeArchiveError("ANSWER_CALL_RELATION_INVALID")
        if self.challenge_answer_call_provenance_status == "explicit_matched" and (
            self.challenge_parsed_response_source_call_id != self.challenge_answer_call_id
        ):
            raise FilterChallengeArchiveError("ANSWER_CALL_RELATION_INVALID")
        if (
            self.control_answer_call_provenance_status != "explicit_matched"
            or self.challenge_answer_call_provenance_status != "explicit_matched"
        ) and self.probe_disposition != "not_evaluable":
            raise FilterChallengeArchiveError("PROVENANCE_DISPOSITION_INVALID")
        if self.candidate_final_context_inclusion != (
            self.candidate_entry_id in self.candidate_final_context_source_ids
        ):
            raise FilterChallengeArchiveError("CANDIDATE_EXPOSURE_MISMATCH")
        if self.total_latency_ms != (
            self.control_latency_ms + self.challenge_latency_ms + self.canonicalization_latency_ms
        ):
            raise FilterChallengeArchiveError("LATENCY_RECONCILIATION_FAILED")
        route = (self.assessment_state, self.final_routing_decision, self.routing_reason_code)
        if route not in {
            ("contradicted", "quarantine", "CONTRADICTED"),
            ("not_contradicted", "active", "NOT_CONTRADICTED"),
            ("not_evaluable", "active", "FAIL_OPEN_NOT_EVALUABLE"),
        }:
            raise FilterChallengeArchiveError("ROUTING_RECONCILIATION_FAILED")
        if (self.assessment_state == "not_evaluable") != (self.probe_disposition == "not_evaluable"):
            raise FilterChallengeArchiveError("ASSESSMENT_DISPOSITION_MISMATCH")
        return self


class CandidateAggregateRecord(_StrictRecord):
    schema_version: Literal["filter_challenge_candidate_aggregate_v1"] = AGGREGATE_SCHEMA_VERSION
    candidate_entry_id: str
    filter_policy_version: Literal["verifier-paired-challenge-v1"]
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str
    operational_probe_suite_id: str
    operational_probe_suite_manifest_hash: str
    decision_rule_id: str
    coverage_contract_id: str
    n_nominal_attempted_pairs: int = Field(ge=0)
    n_control_strict_primary_eligible: int = Field(ge=0)
    n_control_canonicalization_sensitivity_eligible: int = Field(ge=0)
    n_candidate_exposed: int = Field(ge=0)
    n_strictly_evaluable: int = Field(ge=0)
    n_witness: int = Field(ge=0)
    n_no_witness: int = Field(ge=0)
    n_not_evaluable: int = Field(ge=0)
    n_distinct_evaluable_probes: int = Field(ge=0)
    n_distinct_witness_probes: int = Field(ge=0)
    witness_probe_ids: StringTuple
    not_evaluable_reason_counts: dict[str, int]
    aggregation_parameter_tuple: StringTuple
    assessment_state: Literal["contradicted", "not_contradicted", "not_evaluable"]
    final_routing_decision: Literal["quarantine", "active"]
    final_reason_code: Literal["CONTRADICTED", "NOT_CONTRADICTED", "FAIL_OPEN_NOT_EVALUABLE"]
    total_answer_calls: int = Field(ge=0)
    total_baseline_native_aux_calls: int = Field(ge=0)
    total_calls: int = Field(ge=0)
    total_retries: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_cost: float = Field(ge=0)
    total_latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_aggregate(self) -> Self:
        if self.n_nominal_attempted_pairs == 0:
            raise FilterChallengeArchiveError("NOMINAL_ATTEMPTS_REQUIRED")
        if any(
            count > self.n_nominal_attempted_pairs
            for count in (
                self.n_control_strict_primary_eligible,
                self.n_control_canonicalization_sensitivity_eligible,
                self.n_candidate_exposed,
                self.n_strictly_evaluable,
            )
        ) or self.n_witness > self.n_strictly_evaluable:
            raise FilterChallengeArchiveError("AGGREGATE_COUNT_INVALID")
        if self.n_witness + self.n_no_witness != self.n_strictly_evaluable:
            raise FilterChallengeArchiveError("AGGREGATE_COUNT_INVALID")
        if self.n_strictly_evaluable + self.n_not_evaluable != self.n_nominal_attempted_pairs:
            raise FilterChallengeArchiveError("AGGREGATE_COUNT_INVALID")
        if len(set(self.witness_probe_ids)) != len(self.witness_probe_ids) or len(
            self.witness_probe_ids
        ) != self.n_distinct_witness_probes:
            raise FilterChallengeArchiveError("AGGREGATE_COUNT_INVALID")
        if sum(self.not_evaluable_reason_counts.values()) != self.n_not_evaluable:
            raise FilterChallengeArchiveError("AGGREGATE_COUNT_INVALID")
        if self.total_calls != (
            self.total_answer_calls + self.total_baseline_native_aux_calls + self.total_retries
        ):
            raise FilterChallengeArchiveError("OPERATIONS_RECONCILIATION_FAILED")
        route = (self.assessment_state, self.final_routing_decision, self.final_reason_code)
        if route not in {
            ("contradicted", "quarantine", "CONTRADICTED"),
            ("not_contradicted", "active", "NOT_CONTRADICTED"),
            ("not_evaluable", "active", "FAIL_OPEN_NOT_EVALUABLE"),
        }:
            raise FilterChallengeArchiveError("ROUTING_RECONCILIATION_FAILED")
        return self


class ChallengeCallRecord(_StrictRecord):
    schema_version: Literal["filter_challenge_call_record_v1"] = "filter_challenge_call_record_v1"
    call_id: str
    filter_assessment_id: str
    call_kind: Literal["answer", "baseline_native_aux"]
    side: Literal["control", "challenge"]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    monetary_cost: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    retry_count: int = Field(ge=0)


class FilterChallengeArchiveRun(_StrictRecord):
    schema_version: Literal["filter_challenge_archive_run_v1"] = "filter_challenge_archive_run_v1"
    run_id: str
    evidence_layer: Literal["build"] = "build"
    run_family: Literal["filter_challenge_assessment"] = "filter_challenge_assessment"
    record_kind: Literal["paired_pre_admission_probe"] = "paired_pre_admission_probe"
    status: Literal["completed"] = "completed"


class FilterChallengeArchive(_StrictRecord):
    run: FilterChallengeArchiveRun
    assessments: Annotated[tuple[AssessmentRecord, ...], BeforeValidator(parse_tuple)]
    candidate_aggregates: Annotated[tuple[CandidateAggregateRecord, ...], BeforeValidator(parse_tuple)]
    calls: Annotated[tuple[ChallengeCallRecord, ...], BeforeValidator(parse_tuple)]
    audit_labels: Annotated[tuple[PostRouteAuditJoin, ...], BeforeValidator(parse_tuple)]

    @model_validator(mode="after")
    def _validate_relations(self) -> Self:
        _validate_archive_relations(self)
        return self


def canonical_record_hash(record: BaseModel) -> str:
    payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_archive_relations(archive: FilterChallengeArchive) -> None:
    assessment_ids = {record.filter_assessment_id for record in archive.assessments}
    if not assessment_ids or len(assessment_ids) != len(archive.assessments):
        raise FilterChallengeArchiveError("ASSESSMENT_ID_INVALID")
    aggregate_ids = {record.candidate_entry_id for record in archive.candidate_aggregates}
    if aggregate_ids != {record.candidate_entry_id for record in archive.assessments}:
        raise FilterChallengeArchiveError("AGGREGATE_IDENTITY_INVALID")
    calls = {call.call_id: call for call in archive.calls}
    if len(calls) != len(archive.calls) or {call.filter_assessment_id for call in archive.calls} != assessment_ids:
        raise FilterChallengeArchiveError("CALL_RELATION_INVALID")
    for assessment in archive.assessments:
        expected = {
            assessment.control_answer_call_id,
            assessment.challenge_answer_call_id,
            *assessment.baseline_native_aux_call_ids_control,
            *assessment.baseline_native_aux_call_ids_challenge,
        }
        actual = {call.call_id for call in archive.calls if call.filter_assessment_id == assessment.filter_assessment_id}
        if actual != expected:
            raise FilterChallengeArchiveError("CALL_RELATION_INVALID")
        control = calls[assessment.control_answer_call_id]
        challenge = calls[assessment.challenge_answer_call_id]
        aux_control = {calls[call_id] for call_id in assessment.baseline_native_aux_call_ids_control}
        aux_challenge = {calls[call_id] for call_id in assessment.baseline_native_aux_call_ids_challenge}
        scoped = tuple(call for call in archive.calls if call.filter_assessment_id == assessment.filter_assessment_id)
        if (
            (control.call_kind, control.side) != ("answer", "control")
            or (challenge.call_kind, challenge.side) != ("answer", "challenge")
            or any((call.call_kind, call.side) != ("baseline_native_aux", "control") for call in aux_control)
            or any((call.call_kind, call.side) != ("baseline_native_aux", "challenge") for call in aux_challenge)
            or assessment.input_tokens != sum(call.input_tokens for call in scoped)
            or assessment.output_tokens != sum(call.output_tokens for call in scoped)
            or abs(assessment.monetary_cost - sum(call.monetary_cost for call in scoped)) > 1e-12
            or assessment.control_latency_ms != control.latency_ms
            or assessment.challenge_latency_ms != challenge.latency_ms
            or assessment.retry_count_control != control.retry_count
            or assessment.retry_count_challenge != challenge.retry_count
        ):
            raise FilterChallengeArchiveError("CALL_RELATION_INVALID")
    audit_ids = {join.candidate_entry_id for join in archive.audit_labels}
    if audit_ids != aggregate_ids or len(audit_ids) != len(archive.audit_labels):
        raise FilterChallengeArchiveError("AUDIT_JOIN_INVALID")
    routes = {
        aggregate.candidate_entry_id: {
            "schema_version": "filter_challenge_domain_v1",
            "assessment_state": aggregate.assessment_state,
            "route_target": aggregate.final_routing_decision,
            "audit_flag": aggregate.assessment_state == "not_evaluable",
            "routing_reason_code": aggregate.final_reason_code,
        }
        for aggregate in archive.candidate_aggregates
    }
    if any(join.routing_decision.model_dump(mode="json") != routes[join.candidate_entry_id] for join in archive.audit_labels):
        raise FilterChallengeArchiveError("AUDIT_JOIN_INVALID")


__all__ = (
    "AGGREGATE_FIELD_TUPLE", "AGGREGATE_SCHEMA_VERSION", "ASSESSMENT_FIELD_TUPLE",
    "ASSESSMENT_SCHEMA_VERSION", "PUBLIC_AUDIT_KEYS", "AssessmentRecord", "CandidateAggregateRecord",
    "ChallengeCallRecord", "FilterChallengeArchive", "FilterChallengeArchiveError",
    "FilterChallengeArchiveRun", "RawRecordRange", "canonical_record_hash", "normalize_archive_path",
)
