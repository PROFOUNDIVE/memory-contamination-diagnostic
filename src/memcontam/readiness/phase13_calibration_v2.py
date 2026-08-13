from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memcontam.readiness.phase13_analysis_contract import (
    Phase13AnalysisError,
    parse_analysis_registry,
)
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_calibration_v2_registry import (
    CalibrationV2Error,
    validate_calibration_v2_registry,
)
from memcontam.readiness.phase13_execution_contract import (
    Phase13ExecutionError,
    parse_execution_registry,
)
from memcontam.readiness.phase13_terminal import (
    DeterministicAuthoritySyncComplete,
    MainExecutionForbidden,
    Terminal,
)


ROOT = Path(__file__).resolve().parents[3]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CalibrationV2ConfigError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PartitionRef(StrictModel):
    path: Literal["data/phase13/calibration_v2/seed_partition_registry_v1.json"]
    file_sha256: Literal["a31b731244f5c56b4aafa5ed83bbe720c8623563cfbd800e7a478a0025aff4ba"]


class ExecutionRef(StrictModel):
    path: Literal["data/phase13/authority/execution_registry_v1.json"]
    file_sha256: Literal["7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970"]
    registry_hash: Literal["acb769e1e1adbc3eb69e4302322c8eac81829dc836611519caea2ba960900c38"]


class AnalysisRef(StrictModel):
    path: Literal["data/phase13/authority/analysis_registry_v1.json"]
    file_sha256: Literal["b58e6aec8acc040fb934e9b25842eb68c702d098a08b41ba0eab9502a198a0f3"]
    registry_hash: Literal["82960a8f65d316c53bcf55da3e215f0c4b62781643c21155307b40aa9adf4eee"]


class HistoricalCompatibilityRef(StrictModel):
    path: Literal["data/phase13/authority/historical_compatibility_v1.json"]
    file_sha256: Literal["446e5634d7be2bd049ffd3af733262e72a076d22ec24a0e9c11d7259b60264d4"]
    registry_kind: Literal["phase13_historical_compatibility_v1"]
    historical_run_id: Literal["phase13-pre-main-calibration-15usd-rerun1"]
    sealed_archive_availability: Literal["external_reference_unavailable"]


class Authorities(StrictModel):
    calibration_partition: PartitionRef
    execution: ExecutionRef
    analysis: AnalysisRef
    historical_compatibility: HistoricalCompatibilityRef


class ScientificContract(StrictModel):
    H_primary: Literal[5]
    H_run: Literal[10]


class OwnershipContract(StrictModel):
    prefix_owner_id: Literal["phase13-clean-prefix-owner-v1"]
    execution_owner_id: Literal["phase13-h10-execution-owner-v1"]
    offline_owner_id: Literal["phase13-offline-compute-owner-v1"]


class OutputContract(StrictModel):
    artifact_root: Literal["runs/phase13-calibration-v2"]


class CalibrationV2Config(StrictModel):
    config_kind: Literal["phase13_pre_main_calibration_v2"]
    config_id: Literal["phase13-pre-main-calibration-v2"]
    authority: Authorities
    scientific: ScientificContract
    ownership: OwnershipContract
    output: OutputContract
    main_execution: Literal["forbidden"]


def _read(path: Path) -> bytes:
    try:
        return read_regular_nofollow(path)
    except AuthorityFileError as error:
        raise CalibrationV2ConfigError(str(error)) from error


def load_calibration_v2_config(path: Path) -> CalibrationV2Config:
    raw = _read(path)
    try:
        payload = yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise CalibrationV2ConfigError("INVALID_CALIBRATION_V2_CONFIG")
        return CalibrationV2Config.model_validate(payload)
    except ValidationError as error:
        locations = {str(item) for issue in error.errors() for item in issue["loc"]}
        if "H" in locations:
            raise CalibrationV2ConfigError("BARE_H_PROHIBITED") from error
        if "artifact_root" in locations:
            raise CalibrationV2ConfigError("CONFIG_IDENTITY_MISMATCH") from error
        if "authority" in locations:
            raise CalibrationV2ConfigError("AUTHORITY_HASH_MISMATCH") from error
        raise CalibrationV2ConfigError("INVALID_CALIBRATION_V2_CONFIG") from error


def validate_calibration_v2(path: Path) -> Terminal:
    config = load_calibration_v2_config(path)
    partition_raw = _read(ROOT / config.authority.calibration_partition.path)
    execution_raw = _read(ROOT / config.authority.execution.path)
    analysis_raw = _read(ROOT / config.authority.analysis.path)
    historical_raw = _read(ROOT / config.authority.historical_compatibility.path)
    references = (
        (partition_raw, config.authority.calibration_partition.file_sha256),
        (execution_raw, config.authority.execution.file_sha256),
        (analysis_raw, config.authority.analysis.file_sha256),
        (historical_raw, config.authority.historical_compatibility.file_sha256),
    )
    if any(hashlib.sha256(raw).hexdigest() != digest for raw, digest in references):
        raise CalibrationV2ConfigError("AUTHORITY_HASH_MISMATCH")
    try:
        historical = json.loads(historical_raw)
        validate_calibration_v2_registry(ROOT / "data/phase13/calibration_v2", ROOT)
        execution = parse_execution_registry(execution_raw, ROOT)
        analysis = parse_analysis_registry(analysis_raw, ROOT)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationV2ConfigError("HISTORICAL_COMPATIBILITY_INVALID") from error
    except (CalibrationV2Error, Phase13ExecutionError, Phase13AnalysisError) as error:
        raise CalibrationV2ConfigError(error.code) from error
    historical_ref = config.authority.historical_compatibility
    if (
        execution.registry_hash != config.authority.execution.registry_hash
        or analysis.registry_hash != config.authority.analysis.registry_hash
        or execution.timing.H_run != config.scientific.H_run
        or analysis.inference.H_primary != config.scientific.H_primary
        or execution.prefix_owner_id != config.ownership.prefix_owner_id
        or execution.execution_owner_id != config.ownership.execution_owner_id
        or analysis.offline_compute.owner_id != config.ownership.offline_owner_id
        or not isinstance(historical, dict)
        or historical.get("registry_kind") != historical_ref.registry_kind
        or historical.get("historical_execution", {}).get("run_id")
        != historical_ref.historical_run_id
        or historical.get("historical_execution", {}).get("sealed_archive", {}).get(
            "availability"
        )
        != historical_ref.sealed_archive_availability
    ):
        raise CalibrationV2ConfigError("AUTHORITY_SEMANTICS_MISMATCH")
    return DeterministicAuthoritySyncComplete()


def prepare_calibration_v2(path: Path) -> Terminal:
    return validate_calibration_v2(path)


def main_forbidden() -> Terminal:
    return MainExecutionForbidden()
