"""Diagnostic identity is neither Main authorization nor provider-free QA."""
from __future__ import annotations

import hashlib
import json
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from memcontam.logging.schema import MethodCall

from .phase13_cost_policy_models import Sha256
from .phase13_durable_call_join import DiagnosticFile
from .phase13_production_runtime_models import ProductionOrdinaryRunIdentity


class LiveDiagnosticError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DiagnosticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()


class LiveDiagnosticScope(DiagnosticModel):
    provenance: Literal["LIVE_DIAGNOSTIC"] = "LIVE_DIAGNOSTIC"
    main_scope: Literal["NOT_MAIN"] = "NOT_MAIN"
    binding_sha256: Sha256
    child_execution_id: str = Field(min_length=1)
    scientific_disposition: Literal["NOT_ADOPTED", "DIAGNOSTIC_ONLY"]
    run_id: str
    input_utf8: str
    input_sha256: Sha256
    state_utf8: str
    state_sha256: Sha256
    source_files: tuple[DiagnosticFile, ...]
    candidate: DiagnosticFile | None
    archive_identity: ProductionOrdinaryRunIdentity


class DiagnosticRequest(DiagnosticModel):
    request_id: str = Field(min_length=1)
    operation: Literal["generation", "count", "prefix", "setup"]
    stage: str = Field(min_length=1)
    native_call_id: str | None = None


class DiagnosticObservation(DiagnosticModel):
    run_id: str = Field(min_length=1)
    child_execution_id: str = Field(min_length=1)
    input_sha256: Sha256
    state_sha256: Sha256
    requests: tuple[DiagnosticRequest, ...] = Field(min_length=1)


class LiveDiagnosticBinding(DiagnosticModel):
    schema_version: Literal["phase13_live_diagnostic_binding_v1"] = "phase13_live_diagnostic_binding_v1"
    provenance: Literal["LIVE_DIAGNOSTIC"] = "LIVE_DIAGNOSTIC"
    main_scope: Literal["NOT_MAIN"] = "NOT_MAIN"
    currency: Literal["USD"] = "USD"
    run_id: str = Field(min_length=1)
    ledger_path: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_files: tuple[DiagnosticFile, ...] = Field(min_length=1)
    registration_id: str = Field(min_length=1)
    registration_sha256: Sha256
    authorization_id: str = Field(min_length=1)
    authorization_sha256: Sha256
    scientific_disposition: Literal["NOT_ADOPTED", "DIAGNOSTIC_ONLY"]
    observations: tuple[DiagnosticObservation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_identities(self) -> LiveDiagnosticBinding:
        runs = [row.run_id for row in self.observations]
        requests = [request.request_id for row in self.observations for request in row.requests]
        calls = [request.native_call_id for row in self.observations for request in row.requests
                 if request.native_call_id is not None]
        sources = [source.path for source in self.source_files]
        if any(len(values) != len(set(values)) for values in (runs, requests, calls, sources)):
            raise LiveDiagnosticError("DIAGNOSTIC_DUPLICATE_IDENTITY")
        return self


class DiagnosticRequestIntent(DiagnosticModel):
    request_id: str
    attempt_id: Sha256
    request_sha256: Sha256
    reservation_id: str = Field(min_length=1)


class DiagnosticRequestReceipt(DiagnosticModel):
    intent: DiagnosticRequestIntent
    currency: Literal["USD"] = "USD"
    receipt_sha256: Sha256
    realized_cost_usd_micros: int | None = Field(gt=0)
    outcome: Literal["completed", "failed", "ambiguous"]


class LiveDiagnosticOutput(DiagnosticModel):
    evidence: JsonValue
    provider_calls: tuple[MethodCall, ...]
    currency: Literal["USD"] = "USD"
    realized_cost_usd_micros: int | None = Field(ge=0)
    disposition: Literal["COMPLETED", "TERMINAL_FAILURE"]


class LiveDiagnosticEvidence(DiagnosticModel):
    schema_version: Literal["phase13_live_diagnostic_evidence_v1"] = "phase13_live_diagnostic_evidence_v1"
    binding_sha256: Sha256
    intent_event_hash: Sha256
    unit_id: Sha256
    sequence: int = Field(ge=0)
    output: LiveDiagnosticOutput


class DiagnosticStatus(TypedDict):
    completed: int
    terminal: int
    pending: int
    in_flight: int
    known_cost_usd_micros: int
    unknown_cost_requests: int
    unresolved_requests: int
