from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from memcontam.experiment.phase12.contracts import (
    FidelityCertificate,
    Phase12IntegrationCertificate,
    canonical_json_hash,
)
from memcontam.readiness.phase12_replay import P12IReplayResult, P12ISubgateEvidence


EXPECTED_REPOSITORY_COMMIT = "830b89c8c169ffa9cdea472887fdae134dbae7cf"
EXPECTED_F1C_CONTRACT_REFS = {
    "cache_setup_doc_blob_sha": "d2d1e7b2d2405e77c1708ae2a6af808a0316d825",
    "live_embedding_test_blob_sha": "366601812c64046a34531c0966dcc9467041e3d5",
    "provider_config_blob_sha": "96657263c4dd0ce72f6aeff13fffa139a058a53f",
    "verifier_blob_sha": "1f590bc5eb934fad62b9cdc3f06fd355e86127de",
}
_F1C_SOURCE_PATHS = {
    "cache_setup_doc_blob_sha": "docs/bge-m3-cache-setup.md",
    "live_embedding_test_blob_sha": "tests/test_live_embedding_policy.py",
    "provider_config_blob_sha": "src/memcontam/clients/config.py",
    "verifier_blob_sha": "scripts/verify_bge_m3_fidelity.py",
}
_GATE_FIELDS = {
    "prefix_checkpoint": "prefix_checkpoint_gate",
    "five_arm_branch": "five_arm_branch_gate",
    "nomem_alias": "nomem_alias_gate",
    "filter_information_boundary": "filter_information_boundary_gate",
    "logging_v3_join": "logging_v3_join_gate",
    "model_behavior_denominator": "model_behavior_denominator_gate",
    "eligibility_recomputation": "eligibility_recomputation_gate",
    "run_archive_reconstruction": "run_archive_reconstruction_gate",
}


class CertificateError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CertificateValidationReport:
    valid: bool
    scientific_admission: bool
    certificate_hash: str
    reason_code: str | None = None


def issue_p12i(
    result: P12IReplayResult, bfv2: FidelityCertificate
) -> Phase12IntegrationCertificate:
    if result.scientific_result is not False:
        raise CertificateError("P12I_SCIENTIFIC_REPLAY_FORBIDDEN")
    if result.overall_status != "pass" or result.reason_code is not None:
        raise CertificateError("P12I_REPLAY_FAILED")
    _validate_bfv2(bfv2)
    artifacts = artifacts_for(result, bfv2)
    subgates = _validated_subgates(result.subgates)
    gate_hashes: dict[str, str] = {}
    for gate in subgates:
        if gate.evidence_hash is None:
            raise CertificateError("P12I_SUBGATE_HASH_MISMATCH")
        gate_hashes[_GATE_FIELDS[gate.gate_id]] = gate.evidence_hash
    run_dir = result.archive_run_dir
    certificate = Phase12IntegrationCertificate(
        certificate_id=f"phase12-integration-{result.replay_id}",
        protocol_version="phase12_integration_v1",
        git_commit=EXPECTED_REPOSITORY_COMMIT,
        resolved_config_hash=artifacts["resolved_config_hash"],
        public_artifact_manifest_hash=artifacts["public_artifact_manifest_hash"],
        bfv2_certificate_id=bfv2.certificate_id,
        bfv2_certificate_hash=artifacts["bfv2_certificate_hash"],
        **gate_hashes,
        overall_status="pass",
        issued_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    validate_p12i(certificate, artifacts)
    if not run_dir.is_dir():
        raise CertificateError("P12I_ARCHIVE_MISSING")
    return certificate


def artifacts_for(
    result: P12IReplayResult, bfv2: FidelityCertificate
) -> dict[str, Any]:
    run_dir = result.archive_run_dir
    resolved_config = _read_json(run_dir / "resolved_config.json", "P12I_CONFIG_MISSING")
    public_manifest = _read_json(
        run_dir / "public_artifact_manifest.json", "P12I_PUBLIC_MANIFEST_MISSING"
    )
    return {
        "repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "resolved_config": resolved_config,
        "resolved_config_hash": canonical_json_hash(resolved_config),
        "public_artifact_manifest": public_manifest,
        "public_artifact_manifest_hash": canonical_json_hash(public_manifest),
        "bfv2": bfv2,
        "bfv2_certificate_hash": canonical_json_hash(bfv2.model_dump(mode="json")),
        "f1c_status": "pass" if bfv2.overall_status == "pass" else "blocked",
        "f1c_contract_refs": dict(EXPECTED_F1C_CONTRACT_REFS),
        "subgates": result.subgates,
    }


def serialize_p12i(
    certificate: Phase12IntegrationCertificate, destination: Path | None = None
) -> str:
    encoded = json.dumps(certificate.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    if destination is not None:
        destination.write_text(encoded, encoding="utf-8")
    return encoded


def load_p12i(source: str | bytes | Path | Mapping[str, Any]) -> Phase12IntegrationCertificate:
    try:
        if isinstance(source, Path):
            payload: Any = json.loads(source.read_text(encoding="utf-8"))
        elif isinstance(source, bytes):
            payload = json.loads(source.decode("utf-8"))
        elif isinstance(source, str):
            payload = json.loads(source)
        else:
            payload = dict(source)
        return Phase12IntegrationCertificate.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise CertificateError("P12I_CERTIFICATE_SCHEMA_INVALID") from error


def validate_p12i(
    certificate: Phase12IntegrationCertificate | Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> CertificateValidationReport:
    cert = load_p12i(certificate) if not isinstance(certificate, Phase12IntegrationCertificate) else certificate
    if cert.protocol_version != "phase12_integration_v1":
        raise CertificateError("P12I_PROTOCOL_VERSION_MISMATCH")
    if cert.overall_status != "pass":
        raise CertificateError("P12I_CERTIFICATE_NOT_PASS")
    if cert.git_commit != EXPECTED_REPOSITORY_COMMIT:
        raise CertificateError("P12I_REPOSITORY_COMMIT_MISMATCH")
    if artifacts.get("repository_commit", EXPECTED_REPOSITORY_COMMIT) != cert.git_commit:
        raise CertificateError("P12I_REPOSITORY_COMMIT_MISMATCH")
    _validate_artifact_hash(artifacts, "resolved_config", "resolved_config_hash", cert.resolved_config_hash)
    _validate_artifact_hash(
        artifacts,
        "public_artifact_manifest",
        "public_artifact_manifest_hash",
        cert.public_artifact_manifest_hash,
    )
    bfv2 = _bfv2_from_artifacts(artifacts)
    _validate_bfv2(bfv2)
    if cert.bfv2_certificate_id != bfv2.certificate_id:
        raise CertificateError("P12I_BFV2_CERTIFICATE_ID_MISMATCH")
    bfv2_hash = canonical_json_hash(bfv2.model_dump(mode="json"))
    if cert.bfv2_certificate_hash != bfv2_hash:
        raise CertificateError("P12I_BFV2_CERTIFICATE_HASH_MISMATCH")
    _validate_f1c(artifacts, bfv2)
    _validate_f1c_contract_refs(artifacts)
    _validate_subgates(artifacts.get("subgates"), cert)
    scientific_admission = bfv2.overall_status == "pass"
    if "scientific_admission" in artifacts and artifacts["scientific_admission"] is not scientific_admission:
        raise CertificateError("P12I_SCIENTIFIC_ADMISSION_MISMATCH")
    return CertificateValidationReport(
        valid=True,
        scientific_admission=scientific_admission,
        certificate_hash=canonical_json_hash(cert.model_dump(mode="json")),
    )


def _validate_bfv2(bfv2: FidelityCertificate) -> None:
    if bfv2.protocol_version != "baseline_fidelity_v2":
        raise CertificateError("P12I_BFV2_PROTOCOL_VERSION_MISMATCH")
    if bfv2.git_commit != EXPECTED_REPOSITORY_COMMIT:
        raise CertificateError("P12I_REPOSITORY_COMMIT_MISMATCH")
    if bfv2.overall_status not in {"pass", "blocked"}:
        raise CertificateError("P12I_F1C_STATUS_INVALID")


def _bfv2_from_artifacts(artifacts: Mapping[str, Any]) -> FidelityCertificate:
    value = artifacts.get("bfv2", artifacts.get("bfv2_certificate"))
    if value is None:
        raise CertificateError("P12I_BFV2_CERTIFICATE_MISSING")
    try:
        return value if isinstance(value, FidelityCertificate) else FidelityCertificate.model_validate(value)
    except (TypeError, ValueError) as error:
        raise CertificateError("P12I_BFV2_CERTIFICATE_INVALID") from error


def _validate_f1c(artifacts: Mapping[str, Any], bfv2: FidelityCertificate) -> None:
    expected_status = "pass" if bfv2.overall_status == "pass" else "blocked"
    status = artifacts.get("f1c_status")
    f1c = artifacts.get("f1c")
    if status is None and isinstance(f1c, Mapping):
        status = f1c.get("status", f1c.get("overall"))
    if status is not None and status != expected_status:
        raise CertificateError("P12I_F1C_STATUS_MISMATCH")
    if isinstance(f1c, Mapping) and "reason" in f1c:
        expected_reason = "missing_cached_bge_m3" if expected_status == "blocked" else None
        if f1c["reason"] != expected_reason:
            raise CertificateError("P12I_F1C_STATUS_MISMATCH")


def _validate_f1c_contract_refs(artifacts: Mapping[str, Any]) -> None:
    refs = artifacts.get("f1c_contract_refs")
    if not isinstance(refs, Mapping) or dict(refs) != EXPECTED_F1C_CONTRACT_REFS:
        raise CertificateError("P12I_F1C_CONTRACT_HASH_MISMATCH")
    root = Path(__file__).resolve().parents[3]
    for key, relative_path in _F1C_SOURCE_PATHS.items():
        path = root / relative_path
        try:
            content = path.read_bytes()
        except OSError as error:
            raise CertificateError("P12I_F1C_ARTIFACT_MISSING") from error
        observed = _git_blob_sha(content)
        if observed != refs[key]:
            raise CertificateError("P12I_F1C_CONTRACT_HASH_MISMATCH")


def _validate_subgates(
    value: Any, certificate: Phase12IntegrationCertificate, *, compare_certificate: bool = True
) -> None:
    if not isinstance(value, (tuple, list)):
        raise CertificateError("P12I_SUBGATE_EVIDENCE_MISSING")
    expected = {field: getattr(certificate, field) for field in _GATE_FIELDS.values()}
    seen: set[str] = set()
    for gate in value:
        if not isinstance(gate, P12ISubgateEvidence):
            try:
                gate_id = gate["gate_id"]
                status = gate["status"]
                reason_code = gate.get("reason_code")
                path = Path(gate["evidence_path"])
                evidence_hash = gate["evidence_hash"]
            except (AttributeError, KeyError, TypeError) as error:
                raise CertificateError("P12I_SUBGATE_EVIDENCE_INVALID") from error
        else:
            gate_id, status, reason_code, path, evidence_hash = (
                gate.gate_id,
                gate.status,
                gate.reason_code,
                gate.evidence_path,
                gate.evidence_hash,
            )
        field = _GATE_FIELDS.get(gate_id)
        if field is None or field in seen or status != "pass" or reason_code is not None:
            raise CertificateError("P12I_REPLAY_FAILED")
        if not path.is_file() or not isinstance(evidence_hash, str):
            raise CertificateError("P12I_SUBGATE_HASH_MISMATCH")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != evidence_hash or (compare_certificate and observed != expected[field]):
            raise CertificateError("P12I_SUBGATE_HASH_MISMATCH")
        seen.add(field)
    if seen != set(expected):
        raise CertificateError("P12I_SUBGATE_EVIDENCE_MISSING")


def _validate_artifact_hash(
    artifacts: Mapping[str, Any], payload_key: str, hash_key: str, expected: str
) -> None:
    supplied_hash = artifacts.get(hash_key)
    if supplied_hash is None:
        payload = artifacts.get(payload_key)
        if payload is None:
            raise CertificateError("P12I_ARTIFACT_HASH_MISSING")
        if isinstance(payload, (str, Path)):
            payload = _read_json(Path(payload), "P12I_ARTIFACT_MISSING")
        supplied_hash = canonical_json_hash(payload)
    if supplied_hash != expected:
        raise CertificateError(f"P12I_{hash_key.upper()}_MISMATCH")
    payload = artifacts.get(payload_key)
    if payload is not None:
        if isinstance(payload, (str, Path)):
            payload = _read_json(Path(payload), "P12I_ARTIFACT_MISSING")
        if canonical_json_hash(payload) != expected:
            raise CertificateError(f"P12I_{hash_key.upper()}_MISMATCH")


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CertificateError(code) from error


def _validated_subgates(subgates: tuple[P12ISubgateEvidence, ...]) -> tuple[P12ISubgateEvidence, ...]:
    dummy = Phase12IntegrationCertificate(
        certificate_id="pending",
        protocol_version="phase12_integration_v1",
        git_commit=EXPECTED_REPOSITORY_COMMIT,
        resolved_config_hash="pending",
        public_artifact_manifest_hash="pending",
        bfv2_certificate_id="pending",
        bfv2_certificate_hash="pending",
        **{field: "pending" for field in _GATE_FIELDS.values()},
        overall_status="pass",
        issued_at="pending",
    )
    _validate_subgates(subgates, dummy, compare_certificate=False)
    return subgates


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


__all__ = [
    "CertificateError",
    "CertificateValidationReport",
    "EXPECTED_F1C_CONTRACT_REFS",
    "EXPECTED_REPOSITORY_COMMIT",
    "artifacts_for",
    "issue_p12i",
    "load_p12i",
    "serialize_p12i",
    "validate_p12i",
]
