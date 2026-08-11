from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memcontam.readiness import phase13_clean_prefix as contract


@dataclass(frozen=True, slots=True)
class VerifiedCalibration:
    config_path: Path
    config_bytes: bytes
    request_path: Path
    request_bytes: bytes
    authorization_path: Path
    authorization_bytes: bytes
    run_id: str
    implementation_commit: str


def verify_authorization(
    *,
    config_path: Path,
    run_id: str,
    request_path: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
    allow_live_calls: bool,
) -> VerifiedCalibration:
    if not request_path.is_file() or not authorization_path.is_file() or not allow_live_calls:
        raise contract.Phase13CalibrationError("CALIBRATION_AUTHORIZATION_REQUIRED")
    config_bytes = config_path.read_bytes()
    request_bytes = request_path.read_bytes()
    authorization_bytes = authorization_path.read_bytes()
    if _sha256(authorization_bytes) != expected_authorization_sha256:
        raise contract.Phase13CalibrationError("CALIBRATION_AUTHORIZATION_HASH_MISMATCH")
    config = contract.load_clean_prefix_config_bytes(config_bytes)
    request = _read_json(request_bytes)
    authorization = _read_json(authorization_bytes)
    implementation_commit = contract._git("rev-parse", "HEAD")
    if not contract._execution_worktree_clean():
        raise contract.Phase13CalibrationError("CALIBRATION_WORKTREE_DIRTY")
    expected_contract = contract._request_contract(config)
    if any(request.get(key) != value for key, value in expected_contract.items()):
        raise contract.Phase13CalibrationError("CALIBRATION_REQUEST_MISMATCH")
    expected_manifest = str(contract.resolve_output_root(config) / run_id / "artifact_manifest.json")
    if (
        request.get("run_id") != run_id
        or request.get("config")
        != {"path": str(config_path), "sha256": _sha256(config_bytes)}
        or request.get("implementation_commit") != implementation_commit
        or request.get("tracked_worktree_clean") is not True
        or request.get("output_manifest_location") != expected_manifest
        or authorization.get("schema_version") != "phase13_clean_prefix_authorization_v1"
        or authorization.get("run_id") != run_id
        or authorization.get("request_sha256") != _sha256(request_bytes)
        or authorization.get("implementation_commit") != request.get("implementation_commit")
        or authorization.get("config_sha256") != request.get("config", {}).get("sha256")
        or authorization.get("schedule_sha256") != request.get("schedule_sha256")
        or authorization.get("provider_decoding_sha256")
        != request.get("provider_decoding_sha256")
        or authorization.get("maximum_semantic_calls")
        != expected_contract["budget"]["maximum_semantic_calls"]
        or authorization.get("maximum_transport_attempts")
        != expected_contract["budget"]["maximum_transport_attempts"]
        or authorization.get("hard_ceiling_microusd")
        != expected_contract["budget"]["hard_ceiling_microusd"]
    ):
        raise contract.Phase13CalibrationError("CALIBRATION_AUTHORIZATION_MISMATCH")
    return VerifiedCalibration(
        config_path,
        config_bytes,
        request_path,
        request_bytes,
        authorization_path,
        authorization_bytes,
        run_id,
        implementation_commit,
    )


def assert_execution_state(verified: VerifiedCalibration) -> None:
    if (
        contract._git("rev-parse", "HEAD") != verified.implementation_commit
        or not contract._execution_worktree_clean()
        or _sha256(verified.config_path.read_bytes()) != _sha256(verified.config_bytes)
        or _sha256(verified.request_path.read_bytes()) != _sha256(verified.request_bytes)
        or _sha256(verified.authorization_path.read_bytes())
        != _sha256(verified.authorization_bytes)
    ):
        raise contract.Phase13CalibrationError("CALIBRATION_EXECUTION_STATE_CHANGED")


def _read_json(raw: bytes) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise contract.Phase13CalibrationError("CALIBRATION_AUTHORIZATION_MISMATCH")
    return payload


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
