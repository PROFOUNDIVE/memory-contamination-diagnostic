from __future__ import annotations

import hashlib
import hmac
import json
import stat
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Final, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field

from .phase13_authority_files import read_regular_nofollow
from .phase13_cost_policy_models import Sha256
from .phase13_live_diagnostic_models import DiagnosticModel, LiveDiagnosticError
from .phase13_shared_budget_models import FilePin

TRUST_POLICY_PATH: Final = Path("/etc/memcontam/phase13-diagnostic-trust.json")
Purpose = Literal["OPERATIONAL", "SYNTHETIC_QA"]
Role = Literal["science", "launch", "credential"]


class TrustPolicy(DiagnosticModel):
    purpose: Purpose
    science: Sha256
    launch: Sha256
    credential: Sha256


class SignedStatement(DiagnosticModel):
    role: Role
    purpose: Purpose
    subject_sha256: Sha256
    account_id: str = Field(min_length=1)
    not_before: date
    expires: date
    evidence: tuple[FilePin, ...] = Field(min_length=1)
    credential_fingerprint: Sha256 | None
    signature: Annotated[str, Field(pattern=r"^[0-9a-f]{128}$")]


class AdmissionEvidence(DiagnosticModel):
    science: SignedStatement
    launch: SignedStatement
    credential: SignedStatement


def load_policy() -> TrustPolicy:
    for path in (TRUST_POLICY_PATH, *TRUST_POLICY_PATH.parents):
        metadata = path.lstat()
        if (metadata.st_uid != 0 or metadata.st_mode & 0o022 or stat.S_ISLNK(metadata.st_mode)
            or (path == TRUST_POLICY_PATH and not stat.S_ISREG(metadata.st_mode))):
            raise LiveDiagnosticError("DIAGNOSTIC_TRUST_POLICY_UNSAFE")
    policy = TrustPolicy.model_validate_json(read_regular_nofollow(TRUST_POLICY_PATH))
    if policy.purpose != "OPERATIONAL":
        raise LiveDiagnosticError("DIAGNOSTIC_SYNTHETIC_POLICY_REFUSED")
    return policy


def signing_bytes(statement: SignedStatement) -> bytes:
    return b"phase13-diagnostic-admission-v1\0" + json.dumps(
        statement.model_dump(mode="json", exclude={"signature"}), sort_keys=True, separators=(",", ":")
    ).encode()


def credential_fingerprint(api_key: str) -> str:
    return hashlib.sha256(b"phase13-diagnostic-credential-v1\0" + api_key.encode()).hexdigest()


def authenticate(evidence: AdmissionEvidence, subject_sha256: str, account_id: str) -> TrustPolicy:
    policy = load_policy()
    if len({policy.science, policy.launch, policy.credential}) != 3:
        raise LiveDiagnosticError("DIAGNOSTIC_ROLE_SEPARATION_REQUIRED")
    today = datetime.now(UTC).date()
    for role, statement, public in (("science", evidence.science, policy.science),
                                    ("launch", evidence.launch, policy.launch),
                                    ("credential", evidence.credential, policy.credential)):
        if (statement.role != role or statement.purpose != policy.purpose
            or statement.subject_sha256 != subject_sha256 or statement.account_id != account_id
            or not statement.not_before <= today <= statement.expires
            or (statement.credential_fingerprint is not None) != (role == "credential")):
            raise LiveDiagnosticError("DIAGNOSTIC_SIGNED_ADMISSION_MISMATCH")
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
                bytes.fromhex(statement.signature), signing_bytes(statement))
        except InvalidSignature as error:
            raise LiveDiagnosticError("DIAGNOSTIC_ADMISSION_SIGNATURE_INVALID") from error
        for witness in statement.evidence:
            witness.read()
    return policy


def verify_credential(evidence: AdmissionEvidence, api_key: str) -> None:
    fingerprint = evidence.credential.credential_fingerprint
    if not api_key or fingerprint is None or not hmac.compare_digest(fingerprint, credential_fingerprint(api_key)):
        raise LiveDiagnosticError("DIAGNOSTIC_CREDENTIAL_ACCOUNT_MISMATCH")
