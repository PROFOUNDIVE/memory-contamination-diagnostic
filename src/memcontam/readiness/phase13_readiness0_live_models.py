from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from memcontam.readiness.phase13_readiness0_evidence_models import (
    ProviderCallEvidence,
    RuntimeJoinEvidence,
)


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CaseId = Literal[
    "nomem_mmlu_engineering_seed0_suffix1",
    "nomem_mmlu_physics_seed0_suffix1",
    "fh_bounded_game24_clean_seed0_suffix1",
    "rag_frozen_game24_clean_seed0_suffix1",
    "bot_style_game24_clean_seed0_suffix1",
    "reflexion_game24_clean_seed0_suffix1",
    "dc_rs_game24_clean_seed0_suffix1",
]
Baseline = Literal["nomem", "fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactBinding(_FrozenModel):
    path: str = Field(min_length=1)
    sha256: Sha256


class LiveRequest(_FrozenModel):
    schema_version: Literal["phase13_readiness0_live_request_v1"]
    status: Literal["PRE_LIVE_AUTHORIZED"]
    scientific_result: Literal[False]
    main_result: Literal[False]
    measured_main_a_trajectory_count: Literal[0]
    case_ids: tuple[CaseId, ...]
    maximum_provider_calls: Literal[12]
    implementation_manifest: ArtifactBinding
    window_proof: ArtifactBinding
    f1c_registry: ArtifactBinding
    core_manifest: ArtifactBinding
    legacy_rag_manifest: ArtifactBinding
    checkpoint_registry: ArtifactBinding
    observability_packet: ArtifactBinding
    credentials_source: Literal["CURRENT_PROCESS_ENVIRONMENT_ONLY"]
    request_hash: Sha256


class LiveAuthorization(_FrozenModel):
    schema_version: Literal["phase13_readiness0_live_authorization_v1"]
    scope: Literal["MINIMUM_PRODUCTION_FACING_READINESS0_LIVE_PILOT"]
    request_sha256: Sha256
    allow_live_calls: Literal[True]
    maximum_provider_calls: Literal[12]
    authorizes_provider_backed_scientific_calibration: Literal[False]
    authorizes_mr_p5: Literal[False]
    authorizes_mr_p6: Literal[False]
    authorizes_main_a: Literal[False]
    answer_correctness_acceptance_criterion: Literal[False]


class F1CRegistry(_FrozenModel):
    schema_version: Literal["phase13_readiness0_f1c_registry_v1"]
    status: Literal["PASS"]
    cache_environment_variable: Literal["MEMCONTAM_BGE_CACHE_DIR"]
    local_files_only: Literal[True]
    model_id: Literal["BAAI/bge-m3"]
    revision: Literal["5617a9f61b028005a4858fdac845db406aefb181"]
    normalize_embeddings: Literal[True]
    vector_dimension: Literal[1024]
    report: ArtifactBinding
    runtime_hash: Sha256
    legacy_rag_manifest: ArtifactBinding
    ready_legacy_cells: tuple[
        Literal["game24", "math_equation_balancer", "word_sorting"], ...
    ]
    f1c_hash: Sha256


class F1CRuntimeMetadata(_FrozenModel):
    provider_identity: Literal[
        "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181"
    ]
    vector_dimension: Literal[1024]
    normalize_embeddings: Literal[True]
    python: str
    sentence_transformers: str
    torch: str
    device: str
    dtype: str
    local_files_only: Literal[True]
    network_attempts: Literal[0]
    runtime_hash: Sha256


@dataclass(frozen=True, slots=True)
class Readiness0Case:
    case_id: CaseId
    task: Literal["game24", "mmlu_pro_engineering", "mmlu_pro_physics"]
    baseline: Baseline
    stages: tuple[str, ...]
    suffix_position: Literal[1] = 1


class CaseEvidence(_FrozenModel):
    case_id: CaseId
    status: Literal["succeeded", "failed"]
    stages: tuple[str, ...]
    provider_calls: int = Field(ge=0)
    calls: tuple[ProviderCallEvidence, ...]
    answer_call_id: str | None
    runtime: RuntimeJoinEvidence
    reflexion_route_policy_id: Literal["readiness0_reflexion_fail_then_pass_v1"] | None
    routing_verifier_results: tuple[bool, ...]
    actual_verifier_results: tuple[bool, ...]
    scientific_result: Literal[False]
    main_result: Literal[False]


class EvidenceManifest(_FrozenModel):
    schema_version: Literal["phase13_readiness0_live_evidence_manifest_v1"]
    status: Literal["PASS", "FAILED", "PARTIAL"]
    request_sha256: Sha256
    authorization_sha256: Sha256
    f1c_registry_sha256: Sha256
    f1c_runtime: F1CRuntimeMetadata
    cases: ArtifactBinding
    case_count: int = Field(ge=0, le=7)
    provider_call_count: int = Field(ge=0, le=12)
    scientific_result: Literal[False]
    main_result: Literal[False]
    measured_main_a_trajectory_count: Literal[0]
    terminal_case_id: CaseId | None
    terminal_stage: str | None
    failure_code: str | None
    manifest_hash: Sha256


@dataclass(frozen=True, slots=True)
class PilotResult:
    status: Literal["PASS", "FAILED"]
    provider_calls_issued: int
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class EvidenceClosureReport:
    case_count: int
    provider_call_count: int
    scientific_result: bool
    main_result: bool


@dataclass(frozen=True, slots=True)
class VerifiedReadiness0:
    request: LiveRequest
    authorization: LiveAuthorization
    f1c: F1CRegistry
    request_sha256: str
    authorization_sha256: str
    f1c_sha256: str
    output_dir: Path
    f1c_runtime: F1CRuntimeMetadata


class CaseExecutor(Protocol):
    def __call__(self, case: Readiness0Case) -> CaseEvidence: ...


__all__ = [
    "ArtifactBinding", "CaseEvidence", "CaseExecutor", "EvidenceClosureReport",
    "EvidenceManifest", "F1CRegistry", "F1CRuntimeMetadata", "LiveAuthorization", "LiveRequest", "PilotResult",
    "ProviderCallEvidence", "Readiness0Case", "RuntimeJoinEvidence", "VerifiedReadiness0",
]
