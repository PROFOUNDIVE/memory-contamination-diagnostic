from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from memcontam.readiness.phase13_authority import Phase13AuthorityError, parse_phase13_authority_freeze
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_calibration_v2 import ROOT, validate_calibration_v2
from memcontam.readiness.phase13_execution_contract import load_execution_registry
from memcontam.readiness.retrieval_smoke import RetrievalSmokeError, resolve_bge_cache_path


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
PLAN_PATH: Final = Path(
    "/home/hyunwoo/git/memory-contamination-diagnostic/.omo/plans/phase13-canonical-authority-sync-calibration-v2.md"
)
EXPECTED_FILES: Final = {
    "freeze": "data/phase13/authority/phase13_authority_freeze_v1.json",
    "config": "configs/phase13/pre_main_calibration_v2.yaml",
    "partition": "data/phase13/calibration_v2/seed_partition_registry_v1.json",
    "execution": "data/phase13/authority/execution_registry_v1.json",
    "analysis": "data/phase13/authority/analysis_registry_v1.json",
    "structural": "data/phase13/authority/structural_checkpoint_registry_v1.json",
}
EXPECTED_FREEZE_SHA256: Final = "c56de79385e9eee0e00fdc02aae9deea5bc84789af31c0f2e8d9b84d8a6ff449"
IMPLEMENTATION_FILES: Final = {
    "cli": "src/memcontam/readiness/phase13_cli.py",
    "provider_runtime": "src/memcontam/readiness/phase13_provider_runtime.py",
    "trajectory_runtime": "src/memcontam/readiness/phase13_calibration_v2_runtime.py",
}
ImplementationRole = Literal["cli", "provider_runtime", "trajectory_runtime"]


class CalibrationV2AuthorizationError(ValueError):
    def __init__(self) -> None:
        self.code = "CALIBRATION_V2_EXTERNAL_BLOCK"
        super().__init__(self.code)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FileBinding(StrictModel):
    path: str
    sha256: Sha256


class RegistryBindings(StrictModel):
    partition: FileBinding
    execution: FileBinding
    analysis: FileBinding
    structural: FileBinding


class ExecutionIdentities(StrictModel):
    provider_id: Literal["openai-responses-v1"]
    model_snapshot_id: Literal["gpt-4o-2024-11-20"]
    decoding_contract_id: Literal["phase13-decoding-zero-v1"]
    prompt_contract_id: Literal["baseline-fidelity-v2-prompts"]
    tool_contract_id: Literal["text-only-equal-availability-v1"]
    session_contract_id: Literal["paired-isolated-session-v1"]
    failure_contract_id: Literal["baseline-fidelity-v2-failure-taxonomy"]
    resource_contract_id: Literal["phase13-resource-envelope-v1"]


class OwnerBindings(StrictModel):
    prefix: Literal["phase13-clean-prefix-owner-v1"]
    execution: Literal["phase13-h10-execution-owner-v1"]
    offline: Literal["phase13-offline-compute-owner-v1"]


class OperatorCapacity(StrictModel):
    maximum_semantic_calls: Literal[14327]
    maximum_transport_attempts: Literal[57308]
    maximum_input_tokens: Literal[234733568]
    maximum_output_tokens: Literal[117366784]
    maximum_cost_microusd: PositiveInt
    per_request_timeout_seconds: PositiveInt
    maximum_latency_milliseconds: PositiveInt
    maximum_storage_bytes: PositiveInt
    maximum_wall_clock_seconds: PositiveInt
    provider_requests_per_minute: PositiveInt
    provider_concurrency: PositiveInt


class CalibrationV2Request(StrictModel):
    schema_version: Literal["phase13_calibration_v2_request_v1"]
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    freeze: FileBinding
    config: FileBinding
    registries: RegistryBindings
    stream_registry_id: Literal["phase13-calibration-v2-rotations-v1"]
    suffix_registry_id: Literal["phase13-calibration-v2-suffix-v1"]
    analysis_window_registry_id: Literal["phase13-analysis-window-registry-v1"]
    primary_analysis_window_id: Literal["accuracy-h5-primary"]
    identities: ExecutionIdentities
    owners: OwnerBindings
    capacity: OperatorCapacity
    output_root: str
    plan: FileBinding
    implementation_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    implementation: dict[ImplementationRole, FileBinding]
    credential_env_name: Literal["OPENAI_API_KEY"]
    cache_env_name: Literal["MEMCONTAM_BGE_CACHE_DIR"]
    runtime_python: str
    tracked_worktree_clean: Literal[True]

    @model_validator(mode="after")
    def _complete_implementation(self) -> CalibrationV2Request:
        if set(self.implementation) != set(IMPLEMENTATION_FILES):
            raise CalibrationV2AuthorizationError()
        return self


class CalibrationV2Authorization(StrictModel):
    schema_version: Literal["phase13_calibration_v2_authorization_v1"]
    authorization_id: Annotated[str, Field(min_length=1)]
    issued_at: datetime
    expires_at: datetime
    request_sha256: Sha256
    bindings: CalibrationV2Request


class VerifiedCalibrationV2Authorization(StrictModel):
    request: CalibrationV2Request
    authorization: CalibrationV2Authorization
    request_bytes: bytes
    authorization_bytes: bytes


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> bytes:
    try:
        return read_regular_nofollow(path)
    except AuthorityFileError as error:
        raise CalibrationV2AuthorizationError() from error


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_worktree_clean(root: Path) -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout


def _cache_ready(environment: Mapping[str, str]) -> bool:
    try:
        resolve_bge_cache_path(environment)
    except RetrievalSmokeError:
        return False
    return True


def _verify_file(binding: FileBinding, expected: Path) -> bytes:
    if Path(binding.path) != expected:
        raise CalibrationV2AuthorizationError()
    raw = _read(expected)
    if _sha(raw) != binding.sha256:
        raise CalibrationV2AuthorizationError()
    return raw


def verify_calibration_v2_authorization(
    *, config_path: Path, request_path: Path, authorization_path: Path,
    expected_authorization_sha256: str, allow_live_calls: bool,
    environment: Mapping[str, str] | None = None, now: datetime | None = None,
) -> VerifiedCalibrationV2Authorization:
    env = os.environ if environment is None else environment
    if not allow_live_calls or not expected_authorization_sha256:
        raise CalibrationV2AuthorizationError()
    request_raw, authorization_raw = _read(request_path), _read(authorization_path)
    if _sha(authorization_raw) != expected_authorization_sha256:
        raise CalibrationV2AuthorizationError()
    try:
        request = CalibrationV2Request.model_validate_json(request_raw)
        permit = CalibrationV2Authorization.model_validate_json(authorization_raw)
    except (ValidationError, CalibrationV2AuthorizationError) as error:
        raise CalibrationV2AuthorizationError() from error
    current = now or datetime.now(UTC)
    if permit.request_sha256 != _sha(request_raw) or permit.bindings != request:
        raise CalibrationV2AuthorizationError()
    if current.tzinfo is None or permit.issued_at.tzinfo is None or permit.expires_at.tzinfo is None:
        raise CalibrationV2AuthorizationError()
    if permit.issued_at > current or permit.expires_at <= current or permit.expires_at <= permit.issued_at:
        raise CalibrationV2AuthorizationError()
    config_expected = ROOT / EXPECTED_FILES["config"]
    try:
        if config_path.resolve(strict=False) != config_expected or request.config.path != str(config_path):
            raise CalibrationV2AuthorizationError()
    except OSError as error:
        raise CalibrationV2AuthorizationError() from error
    _verify_file(request.config, config_expected)
    freeze_raw = _verify_file(request.freeze, ROOT / EXPECTED_FILES["freeze"])
    if request.freeze.sha256 != EXPECTED_FREEZE_SHA256:
        raise CalibrationV2AuthorizationError()
    try:
        parse_phase13_authority_freeze(freeze_raw)
    except Phase13AuthorityError as error:
        raise CalibrationV2AuthorizationError() from error
    for role in ("partition", "execution", "analysis", "structural"):
        _verify_file(getattr(request.registries, role), ROOT / EXPECTED_FILES[role])
    _verify_file(request.plan, PLAN_PATH)
    for role in ("cli", "provider_runtime", "trajectory_runtime"):
        relative = IMPLEMENTATION_FILES[role]
        _verify_file(request.implementation[role], ROOT / relative)
    try:
        if request.implementation_commit != _git_head(ROOT) or not _tracked_worktree_clean(ROOT):
            raise CalibrationV2AuthorizationError()
    except (OSError, subprocess.SubprocessError) as error:
        raise CalibrationV2AuthorizationError() from error
    if request.runtime_python != sys.version.split()[0]:
        raise CalibrationV2AuthorizationError()
    try:
        if not env.get(request.credential_env_name) or not _cache_ready(env):
            raise CalibrationV2AuthorizationError()
    except OSError as error:
        raise CalibrationV2AuthorizationError() from error
    if request.output_root != str(ROOT / "runs/phase13-calibration-v2" / request.run_id):
        raise CalibrationV2AuthorizationError()
    try:
        validate_calibration_v2(config_path)
        execution = load_execution_registry(ROOT / EXPECTED_FILES["execution"], ROOT)
    except (OSError, ValueError) as error:
        raise CalibrationV2AuthorizationError() from error
    if any(
        getattr(execution.identities, field) != value
        for field, value in request.identities.model_dump().items()
    ):
        raise CalibrationV2AuthorizationError()
    return VerifiedCalibrationV2Authorization(
        request=request, authorization=permit,
        request_bytes=request_raw, authorization_bytes=authorization_raw,
    )


__all__ = (
    "CalibrationV2AuthorizationError", "VerifiedCalibrationV2Authorization",
    "verify_calibration_v2_authorization",
)
