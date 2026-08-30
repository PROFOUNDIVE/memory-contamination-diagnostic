from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_main_checkpoint import (
    CommonCheckpointRegistry,
    validate_main_checkpoint_package,
)


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
IMPLEMENTATION_PATHS: Final = {
    "live_cli": "src/memcontam/readiness/phase13_readiness0_cli.py",
    "live_orchestrator": "src/memcontam/readiness/phase13_readiness0_live.py",
    "production_executor": "src/memcontam/readiness/phase13_readiness0_live_runtime.py",
    "call_budget": "src/memcontam/readiness/phase13_readiness0_budget.py",
    "f1c_runtime": "src/memcontam/readiness/phase13_readiness0_f1c.py",
    "f1c_report": "src/memcontam/readiness/phase13_readiness0_f1c_report.py",
    "f1c_contract": "src/memcontam/readiness/phase13_readiness0_f1c_contract.py",
    "f1c_builder": "src/memcontam/readiness/phase13_readiness0_f1c_build.py",
    "f1c_validator": "src/memcontam/readiness/phase13_readiness0_f1c_validate.py",
    "evidence_models": "src/memcontam/readiness/phase13_readiness0_evidence_models.py",
    "evidence_builder": "src/memcontam/readiness/phase13_readiness0_case_evidence.py",
    "evidence_validator": "src/memcontam/readiness/phase13_readiness0_evidence_validate.py",
    "current_status": "src/memcontam/readiness/phase13_readiness0_current_status.py",
    "ordinary_runtime": "src/memcontam/experiment/phase13_ordinary_runtime.py",
    "rag_adapter": "src/memcontam/baselines/retrieval_rag_phase12.py",
    "bot_retrieval": "src/memcontam/baselines/bot_read.py",
    "dc_rs_adapter": "src/memcontam/baselines/dynamic_cheatsheet_phase12.py",
    "legacy_rag_runtime": "src/memcontam/readiness/phase13_legacy_rag_runtime.py",
    "provider_adapter": "src/memcontam/clients/openai_responses.py",
    "recording_adapter": "src/memcontam/clients/recording.py",
    "logging_schema": "src/memcontam/logging/schema.py",
    "logging_schema_v3": "src/memcontam/logging/schema_v3.py",
    "cost_policy": "src/memcontam/readiness/phase13_cost_policy.py",
    "cost_policy_models": "src/memcontam/readiness/phase13_cost_policy_models.py",
    "stage_envelopes": "data/phase13/main/cost_envelope_v2/stage_envelope_registry_v1.json",
    "failure_contract": "data/phase13/main/cost_envelope_v2/retry_failure_contract_v1.json",
    "activated_cost_policy": "data/phase13/main/cost_envelope_v2/activated_policy_v1.json",
}


class Readiness0PackageError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactIdentity(_FrozenModel):
    path: str = Field(min_length=1)
    sha256: Sha256


class LiveImplementationManifest(_FrozenModel):
    schema_version: Literal["phase13_readiness0_live_implementation_manifest_v1"]
    status: Literal["PASS"]
    artifacts: dict[str, ArtifactIdentity]
    manifest_hash: Sha256


class WindowRow(_FrozenModel):
    window_id: str = Field(pattern=r"^core_prefix_(0[1-9]|[1-4][0-9]|50)$")
    start: Literal[1]
    end: int = Field(ge=1, le=50)
    role: Literal["confirmatory_primary", "prespecified_sensitivity", "descriptive"]


class TaskSeedContext(_FrozenModel):
    task: str = Field(min_length=1)
    seed: int = Field(ge=0, le=9)
    checkpoint_registry_sha256: Sha256
    suffix_sample_ids_sha256: Sha256
    suffix_length: Literal[50]


class WindowProof(_FrozenModel):
    schema_version: Literal["phase13_readiness0_window_proof_v1"]
    status: Literal["PASS"]
    H_run: Literal[50]
    H_primary: Literal[50]
    primary_analysis_window_id: Literal["core_prefix_50"]
    provider_dispatch_suffix_positions: tuple[Literal[1], ...]
    resolved_context_count: Literal[2500]
    contexts: tuple[TaskSeedContext, ...]
    windows: tuple[WindowRow, ...]
    proof_hash: Sha256


def build_implementation_manifest(repository_root: Path) -> LiveImplementationManifest:
    artifacts = {
        name: ArtifactIdentity(path=path, sha256=_sha256(repository_root / path))
        for name, path in IMPLEMENTATION_PATHS.items()
    }
    manifest = LiveImplementationManifest(
        schema_version="phase13_readiness0_live_implementation_manifest_v1",
        status="PASS",
        artifacts=artifacts,
        manifest_hash="0" * 64,
    )
    return manifest.model_copy(update={"manifest_hash": _model_hash(manifest, "manifest_hash")})


def validate_implementation_manifest(
    raw: bytes, repository_root: Path
) -> LiveImplementationManifest:
    try:
        manifest = LiveImplementationManifest.model_validate_json(raw)
    except ValidationError as error:
        raise Readiness0PackageError("READINESS0_IMPLEMENTATION_MANIFEST_INVALID") from error
    if manifest != build_implementation_manifest(repository_root):
        raise Readiness0PackageError("READINESS0_IMPLEMENTATION_MANIFEST_INVALID")
    return manifest


def build_window_proof(root: Path, repository_root: Path) -> WindowProof:
    report = validate_main_checkpoint_package(root, repository_root)
    registry = CommonCheckpointRegistry.model_validate_json(
        read_regular_nofollow(root / "main_a_common_checkpoint_registry_v1.json")
    )
    if any(
        len(seed.suffix_sample_ids) != 50
        for task_row in registry.tasks.values()
        for seed in task_row.seeds
    ):
        raise Readiness0PackageError("READINESS0_WINDOW_PROOF_INVALID")
    contexts = tuple(
        TaskSeedContext(
            task=task,
            seed=seed.seed,
            checkpoint_registry_sha256=report.registry_sha256,
            suffix_sample_ids_sha256=seed.suffix_sample_ids_sha256,
            suffix_length=50,
        )
        for task, task_row in registry.tasks.items()
        for seed in task_row.seeds
    )
    windows = tuple(
        WindowRow(
            window_id=f"core_prefix_{end:02d}",
            start=1,
            end=end,
            role=(
                "confirmatory_primary"
                if end == 50
                else "prespecified_sensitivity"
                if end in {5, 10, 20}
                else "descriptive"
            ),
        )
        for end in range(1, 51)
    )
    proof = WindowProof(
        schema_version="phase13_readiness0_window_proof_v1",
        status="PASS",
        H_run=50,
        H_primary=50,
        primary_analysis_window_id="core_prefix_50",
        provider_dispatch_suffix_positions=(1,),
        resolved_context_count=2500,
        contexts=contexts,
        windows=windows,
        proof_hash="0" * 64,
    )
    return proof.model_copy(update={"proof_hash": _model_hash(proof, "proof_hash")})


def validate_window_proof(raw: bytes, root: Path, repository_root: Path) -> WindowProof:
    try:
        proof = WindowProof.model_validate_json(raw)
    except ValidationError as error:
        raise Readiness0PackageError("READINESS0_WINDOW_PROOF_INVALID") from error
    if proof != build_window_proof(root, repository_root):
        raise Readiness0PackageError("READINESS0_WINDOW_PROOF_INVALID")
    return proof


def _model_hash(model: _FrozenModel, field: str) -> str:
    payload: dict[str, JsonValue] = model.model_dump(mode="json", exclude={field})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(read_regular_nofollow(path)).hexdigest()


__all__ = [
    "LiveImplementationManifest",
    "Readiness0PackageError",
    "WindowProof",
    "build_implementation_manifest",
    "build_window_proof",
    "validate_implementation_manifest",
    "validate_window_proof",
]
