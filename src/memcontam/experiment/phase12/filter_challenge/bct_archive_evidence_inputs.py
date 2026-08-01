from __future__ import annotations

from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.bct_archive_models import LedgerError
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EvidenceBuildError,
    json_value_from_bytes,
    read_regular_nofollow,
    sha256_regular_nofollow,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


_AUTHORITY_MANIFEST: Final = Path("docs/evidence/phase12-filter-v5-bct-v1/authority_transition_manifest.json")
_AUTHORITY_MANIFEST_SHA256: Final = "8ec54eba36214371e5b6e513392a4b6392d27f4839ebd23158eddcd08706c499"
_CALIBRATION_CONFIG: Final = Path("configs/phase12/filter_v5_bct_calibration.yaml")
_CALIBRATION_CONFIG_SHA256: Final = "76c710a8fcb6b77b8c759dacffa4610ae0335e05f1729d6c8bb58fab27213548"
_DATA_ROOT: Final = Path("data/phase12/filter_v5_bct_v1")
_FREEZE_A: Final = _DATA_ROOT / "freeze_a.json"
_METHODS_LOCK: Final = Path("docs/phase12-filter-v5-bct-methods-lock.md")
_METHODS_LOCK_SHA256: Final = "3c13f53695ded5503080a4ff552e3e9f4c97ec7b5eafe1a3104b6411546740e8"
_SCREENING_REQUEST: Final = _DATA_ROOT / "screening_authorization_request.json"
_SOURCE_UNIVERSE: Final = _DATA_ROOT / "source_universe_v1.json"
_FREEZE_BOUND_REPORTS: Final = (
    "freeze_a",
    "screening",
    "freeze_b_search_config",
    "bct_execution",
    "archive_validation",
    "claim_scope",
)


def validate_current_readiness_inputs(reports: dict[str, dict[str, JsonValue]]) -> None:
    root = Path.cwd()
    _require_digest(root / _AUTHORITY_MANIFEST, _AUTHORITY_MANIFEST_SHA256)
    _require_digest(root / _CALIBRATION_CONFIG, _CALIBRATION_CONFIG_SHA256)
    _require_digest(root / _METHODS_LOCK, _METHODS_LOCK_SHA256)
    source_digest = _digest(root / _SOURCE_UNIVERSE, "EVIDENCE_SOURCE_UNIVERSE_INVALID")
    freeze_digest = _digest(root / _FREEZE_A, "EVIDENCE_FROZEN_INPUT_INVALID")
    request_digest = _digest(root / _SCREENING_REQUEST, "EVIDENCE_FROZEN_INPUT_INVALID")
    _require_report_digest(reports["freeze_a"], "source_universe", source_digest)
    _validate_source_universe(root)
    _validate_freeze(root, freeze_digest)
    _validate_screening_request(root, freeze_digest)
    for report_id in _FREEZE_BOUND_REPORTS:
        _require_report_digest(reports[report_id], "freeze_a", freeze_digest)
    for report_id in _FREEZE_BOUND_REPORTS[1:]:
        _require_report_digest(reports[report_id], "authorization_request", request_digest)


def _require_digest(path: Path, expected: str) -> None:
    if _digest(path, "EVIDENCE_FROZEN_INPUT_INVALID") != expected:
        raise LedgerError("EVIDENCE_FROZEN_INPUT_INVALID")


def _require_report_digest(payload: dict[str, JsonValue], name: str, expected: str) -> None:
    inputs = payload.get("input_digests")
    if not isinstance(inputs, dict) or inputs.get(name) != expected:
        raise LedgerError("EVIDENCE_FROZEN_INPUT_INVALID")


def _validate_source_universe(root: Path) -> None:
    source = _json(root / _SOURCE_UNIVERSE, "EVIDENCE_SOURCE_UNIVERSE_INVALID")
    files = source.get("source_files")
    if source.get("schema_version") != "phase12_fv5_source_universe_v1" or not isinstance(files, dict):
        raise LedgerError("EVIDENCE_SOURCE_UNIVERSE_INVALID")
    for relative_path, expected in files.items():
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise LedgerError("EVIDENCE_SOURCE_UNIVERSE_INVALID")
        path = Path(relative_path)
        if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
            raise LedgerError("EVIDENCE_SOURCE_UNIVERSE_INVALID")
        if _digest(root / path, "EVIDENCE_SOURCE_UNIVERSE_INVALID") != expected:
            raise LedgerError("EVIDENCE_SOURCE_UNIVERSE_INVALID")


def _validate_freeze(root: Path, freeze_digest: str) -> None:
    freeze = _json(root / _FREEZE_A, "EVIDENCE_FROZEN_INPUT_INVALID")
    manifests = freeze.get("manifest_sha256")
    if (
        freeze.get("schema_version") != "phase12_fv5_freeze_a_v1"
        or not isinstance(manifests, dict)
    ):
        raise LedgerError("EVIDENCE_FROZEN_INPUT_INVALID")
    for name, expected in manifests.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise LedgerError("EVIDENCE_FROZEN_INPUT_INVALID")
        path = Path(name)
        if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
            raise LedgerError("EVIDENCE_FROZEN_INPUT_INVALID")
        if _digest(root / _DATA_ROOT / path, "EVIDENCE_FROZEN_INPUT_INVALID") != expected:
            raise LedgerError("EVIDENCE_FROZEN_INPUT_INVALID")
    if freeze.get("provider_calls_issued") != 0:
        raise LedgerError("EVIDENCE_FROZEN_INPUT_INVALID")


def _validate_screening_request(root: Path, freeze_digest: str) -> None:
    request = _json(root / _SCREENING_REQUEST, "EVIDENCE_FROZEN_INPUT_INVALID")
    if (
        request.get("schema_version") != "phase12_fv5_authorization_request_v1"
        or request.get("stage") != "screening"
        or request.get("freeze_sha256") != freeze_digest
        or request.get("provider_calls_issued") != 0
    ):
        raise LedgerError("EVIDENCE_FROZEN_INPUT_INVALID")


def _json(path: Path, code: str) -> dict[str, JsonValue]:
    try:
        value = json_value_from_bytes(read_regular_nofollow(path, code), code)
    except EvidenceBuildError as error:
        raise LedgerError(code) from error
    if not isinstance(value, dict):
        raise LedgerError(code)
    return value


def _digest(path: Path, code: str) -> str:
    try:
        return sha256_regular_nofollow(path, code)
    except EvidenceBuildError as error:
        raise LedgerError(code) from error
