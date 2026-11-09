from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from .phase13_authority_files import read_regular_nofollow
from .phase13_diagnostic_cli_models import RegisteredDiagnostic, RegistrationSubject
from .phase13_diagnostic_proposal import prepare_proposal
from .phase13_diagnostic_proposal_models import (
    PreparationMetadata, PreparedProposal, Prerequisite, ProposalHeader, ProposalProjection,
)
from .phase13_diagnostic_trusted_admission import AdmissionEvidence
from .phase13_live_diagnostic_models import LiveDiagnosticError


class PendingEligibility(ProposalProjection):
    schema_id: Literal["phase13-task11-eligibility-v1"] = Field(alias="schema")
    status: Literal["PENDING_AUTHENTIC_FACTS_AND_NATIVE_BINDINGS"]
    protocol_owner_disposition: JsonValue
    existing_pending_gates: dict[str, JsonValue]


class PendingPrerequisites(LiveDiagnosticError):
    def __init__(self, result: PreparedProposal) -> None:
        self.result = result
        super().__init__("DIAGNOSTIC_PROPOSAL_PREREQUISITES_PENDING")


@dataclass(frozen=True, slots=True)
class IntakePaths:
    scope: Path
    eligibility: Path
    budget: Path
    approval: Path
    preparation: Path | None = None


def pending_gates(value: JsonValue, path: str) -> tuple[Prerequisite, ...]:
    if isinstance(value, dict):
        if "status" in value:
            return (Prerequisite(path=path, code="AUTHENTIC_ELIGIBILITY_EVIDENCE_REQUIRED"),)
        return tuple(row for key, child in value.items() for row in pending_gates(child, f"{path}.{key}"))
    return ()


def candidate_from_intake(paths: IntakePaths) -> RegisteredDiagnostic:
    raw = read_regular_nofollow(paths.scope)
    eligibility = read_regular_nofollow(paths.eligibility)
    if ProposalHeader.model_validate_json(raw).schema_id == "phase13-task11-proposed16-v1":
        metadata = (None if paths.preparation is None else
                    PreparationMetadata.model_validate_json(read_regular_nofollow(paths.preparation)))
        prepared = prepare_proposal(paths.scope, metadata)
        prerequisites = list(prepared.prerequisites)
        if ProposalHeader.model_validate_json(eligibility).schema_id == "phase13-task11-eligibility-v1":
            pending = PendingEligibility.model_validate_json(eligibility)
            prerequisites.extend(pending_gates(pending.existing_pending_gates, "eligibility.existing_pending_gates"))
            prerequisites.append(Prerequisite(path="eligibility.protocol_owner_disposition",
                                              code="AUTHENTIC_SCIENTIFIC_DISPOSITION_REQUIRED"))
            prerequisites.extend(Prerequisite(path=f"admission.{role}", code="SIGNED_EXACT_SUBJECT_ADMISSION_REQUIRED")
                                 for role in ("science", "launch", "credential"))
        if prerequisites:
            raise PendingPrerequisites(prepared.model_copy(update={
                "status": "PENDING_PREREQUISITES", "prerequisites": tuple(prerequisites)}))
        assert prepared.subject is not None
        subject = prepared.subject
    else:
        if ProposalHeader.model_validate_json(raw).schema_id:
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_SCHEMA_UNSUPPORTED")
        if paths.preparation is not None:
            raise LiveDiagnosticError("DIAGNOSTIC_AMBIGUOUS_INTAKE")
        subject = RegistrationSubject.model_validate_json(raw)
    admission = AdmissionEvidence.model_validate_json(eligibility)
    approval_sha256 = hashlib.sha256(read_regular_nofollow(paths.approval)).hexdigest()
    if (subject.context().contract.budget_binding.path != str(paths.budget)
        or subject.binding.authorization_sha256 != approval_sha256
        or not any(pin.path == str(paths.approval) and pin.sha256 == approval_sha256
                   for pin in admission.launch.evidence)):
        raise LiveDiagnosticError("DIAGNOSTIC_INTAKE_BINDING_MISMATCH")
    return RegisteredDiagnostic(subject=subject, admission=admission)
