from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.evaluation.phase13_observability import reconstruct_phase13_trial
from memcontam.evaluation.phase13_observability_models import (
    Phase13ObservabilityError,
    Phase13TrialEvidence,
)
from memcontam.evaluation.phase13_observability_registration import (
    ObservabilityRegistrationPacket,
)
from memcontam.evaluation.phase13_observability_sequence import reconstruct_registered_sequence
from memcontam.readiness.phase13_observability_models import Phase13ObservabilityFixture
from memcontam.readiness.phase13_production_runtime_models import ProductionNoMemTrialEvidence
from memcontam.logging.schema import MethodCall


class ProductionObservabilityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProviderRequestRecord(_FrozenModel):
    api: Literal["OpenAI Responses API"]
    model: Literal["gpt-5.6-luna"]
    service_tier: Literal["default"]
    reasoning_mode: Literal["standard"]
    reasoning_effort: Literal["none"]
    reasoning_context: Literal["current_turn"]
    previous_response_id: None
    store: Literal[False]
    tools: tuple[()] = ()
    timeout_seconds: Literal[180]
    retries_after_initial_attempt: Literal[0]
    semantic_invalid_generic_retry: Literal[False]


class TerminalProviderEvidence(_FrozenModel):
    trigger_class: Literal["provider_call_failure", "input_envelope_violation"]
    failure_code: str | None = None
    attempts_count: int = Field(ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    status: str | None = None
    incomplete_reason: str | None = None
    usage: dict[str, object] | None = None
    token_usage: dict[str, int] | None = None
    cost_usd: float | None = Field(default=None, ge=0)
    response_id: str | None = None


class ProductionTrialRecord(_FrozenModel):
    execution_template_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    scientific_result: bool
    ordered_sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: ProviderRequestRecord
    parsed_answer: str | None
    method_calls: tuple[MethodCall, ...]
    evidence: Phase13TrialEvidence | ProductionNoMemTrialEvidence
    terminal_method_call: MethodCall | None = None
    terminal_provider_evidence: TerminalProviderEvidence | None = None

    @model_validator(mode="after")
    def _runtime_identity(self) -> ProductionTrialRecord:
        event_run_ids = (
            {
                *(event.run_id for event in self.evidence.retrievals),
                *(event.run_id for event in self.evidence.memory_events),
                *((self.evidence.context.run_id,) if self.evidence.context is not None else ()),
            }
            if isinstance(self.evidence, Phase13TrialEvidence)
            else set()
        )
        if event_run_ids and event_run_ids != {self.run_id}:
            raise ProductionObservabilityError("PRODUCTION_RUN_JOIN_MISMATCH")
        failed = self.evidence.trial.execution_status == "failed"
        if failed != (self.terminal_method_call is not None) or failed != (
            self.terminal_provider_evidence is not None
        ):
            raise ProductionObservabilityError("TERMINAL_PROVIDER_EVIDENCE_MISMATCH")
        if self.terminal_method_call is not None and self.terminal_provider_evidence != (
            terminal_provider_evidence(self.terminal_method_call)
        ):
            raise ProductionObservabilityError("TERMINAL_PROVIDER_EVIDENCE_MISMATCH")
        if self.terminal_method_call is not None and self.terminal_method_call not in self.method_calls:
            raise ProductionObservabilityError("TERMINAL_PROVIDER_EVIDENCE_MISMATCH")
        return self


class ProductionObservabilityArchive(_FrozenModel):
    schema_version: Literal[
        "phase13_production_observability_archive_v1",
        "phase13_production_observability_archive_v2",
    ]
    registration_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    u_t_status: Literal["NOT_REGISTERED_FOR_CURRENT_MAIN"]
    records: tuple[ProductionTrialRecord, ...]

    @model_validator(mode="after")
    def _session_isolation(self) -> ProductionObservabilityArchive:
        sessions = tuple(record.session_id for record in self.records)
        if len(sessions) != len(set(sessions)):
            raise ProductionObservabilityError("CROSS_TRIAL_SESSION_REUSE")
        scientific_result = self.schema_version == "phase13_production_observability_archive_v2"
        if any(record.scientific_result is not scientific_result for record in self.records):
            raise ProductionObservabilityError("PRODUCTION_SCIENTIFIC_RESULT_MISMATCH")
        return self


class ProductionObservabilityReport(_FrozenModel):
    status: Literal["PASS"]
    record_count: int = Field(gt=0)
    technical_missing_count: int = Field(ge=0, le=1)
    reconstruction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    u_t_status: Literal["NOT_REGISTERED_FOR_CURRENT_MAIN"]


def conformance_archive(
    fixture: Phase13ObservabilityFixture,
    packet_sha256: str,
) -> ProductionObservabilityArchive:
    ordered_sample_ids_sha256 = hashlib.sha256(
        json.dumps(
            [trial.trial_id for trial in fixture.trials], separators=(",", ":")
        ).encode()
    ).hexdigest()
    records = tuple(
        ProductionTrialRecord(
            execution_template_id=f"conformance:{trial.task}:{trial.baseline}:{trial.trial.execution_key.arm}",
            run_id=(
                trial.context.run_id
                if trial.context is not None
                else f"main-disjoint-conformance-run-{index}"
            ),
            session_id=f"main-disjoint-conformance-session-{index}",
            scientific_result=False,
            ordered_sample_ids_sha256=ordered_sample_ids_sha256,
            request=ProviderRequestRecord(
                api="OpenAI Responses API",
                model="gpt-5.6-luna",
                service_tier="default",
                reasoning_mode="standard",
                reasoning_effort="none",
                reasoning_context="current_turn",
                previous_response_id=None,
                store=False,
                timeout_seconds=180,
                retries_after_initial_attempt=0,
                semantic_invalid_generic_retry=False,
            ),
            parsed_answer=None,
            method_calls=(),
            evidence=trial,
        )
        for index, trial in enumerate(fixture.trials)
    )
    return ProductionObservabilityArchive(
        schema_version="phase13_production_observability_archive_v1",
        registration_packet_sha256=packet_sha256,
        u_t_status="NOT_REGISTERED_FOR_CURRENT_MAIN",
        records=records,
    )


def validate_production_archive(
    archive: ProductionObservabilityArchive,
    packet: ObservabilityRegistrationPacket,
    packet_sha256: str,
) -> ProductionObservabilityReport:
    if archive.registration_packet_sha256 != packet_sha256:
        raise ProductionObservabilityError("PRODUCTION_REGISTRATION_PACKET_MISMATCH")
    evidence = tuple(record.evidence for record in archive.records)
    technical = tuple(
        row for row in evidence if row.trial.execution_status == "failed"
    )
    if technical and (
        len(technical) != 1
        or evidence[-1] is not technical[0]
        or technical[0].verified_outcome is not None
        or technical[0].trial.analysis_inclusion != "excluded"
    ):
        raise ProductionObservabilityError("TERMINAL_TECHNICAL_MISSINGNESS_MISMATCH")
    completed = evidence[: len(evidence) - len(technical)]
    memory_completed = tuple(
        row for row in completed if isinstance(row, Phase13TrialEvidence)
    )
    nomem_completed = tuple(
        row for row in completed if isinstance(row, ProductionNoMemTrialEvidence)
    )
    if memory_completed and nomem_completed:
        raise ProductionObservabilityError("PRODUCTION_EVIDENCE_KIND_MISMATCH")
    registered_target_present = any(
        row.target_set.target_entry_ids for row in memory_completed
    )
    try:
        base = (
            tuple(reconstruct_phase13_trial(row) for row in memory_completed)
            if registered_target_present
            else ()
        )
        reconstructed = (
            reconstruct_registered_sequence(memory_completed, base, packet.recurrence_lookback_h)
            if registered_target_present
            else ()
        )
    except Phase13ObservabilityError as error:
        raise ProductionObservabilityError("PRODUCTION_RECONSTRUCTION_FAILED") from error
    payload = [
        row.model_dump(mode="json")
        for row in reconstructed or memory_completed or nomem_completed
    ]
    payload.extend(row.model_dump(mode="json") for row in technical)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ProductionObservabilityReport(
        status="PASS",
        record_count=len(evidence),
        technical_missing_count=len(technical),
        reconstruction_sha256=digest,
        u_t_status=archive.u_t_status,
    )


def terminal_provider_evidence(call: MethodCall) -> TerminalProviderEvidence:
    if call.error_type is None:
        raise ProductionObservabilityError("TERMINAL_PROVIDER_CALL_INVALID")
    return TerminalProviderEvidence(
        trigger_class=(
            "input_envelope_violation"
            if call.failure_code == "INPUT_ENVELOPE_EXCEEDED"
            else "provider_call_failure"
        ),
        failure_code=call.failure_code,
        attempts_count=call.transport_attempts,
        latency_ms=call.latency_ms,
        status=call.provider_status,
        incomplete_reason=call.provider_incomplete_reason,
        usage=call.provider_usage,
        token_usage=call.token_usage or None,
        cost_usd=call.provider_cost_usd,
        response_id=call.provider_response_id,
    )


__all__ = [
    "ProductionObservabilityArchive",
    "ProductionObservabilityError",
    "ProductionObservabilityReport",
    "ProductionTrialRecord",
    "ProviderRequestRecord",
    "TerminalProviderEvidence",
    "terminal_provider_evidence",
    "conformance_archive",
    "validate_production_archive",
]
