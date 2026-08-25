from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


TaskName = Literal["game24", "math_equation_balancer", "word_sorting"]
FeasibleTaskName = TaskName
BranchName = Literal["clean", "correct", "irrelevant", "contam"]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MaterializationStatus = Literal[
    "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE", "TEST_ONLY_NOT_READY"
]
TASKS = ("game24", "math_equation_balancer", "word_sorting")
FEASIBLE_TASKS = TASKS
MATERIALIZED_TASKS: tuple[FeasibleTaskName, ...] = TASKS
BRANCHES = ("clean", "correct", "irrelevant", "contam")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactReference(_FrozenModel):
    path: str
    sha256: Sha256
    row_count: Annotated[int, Field(ge=0)] | None = None


class GeneratorIdentity(_FrozenModel):
    generator_id: str
    implementation: ArtifactReference
    authority: ArtifactReference
    candidate_domain: str
    candidate_key_schema: str
    candidate_ordering_contract: Literal["sha256_raw_bytes_then_canonical_json_bytes_v1"]


class BuildCandidate(_FrozenModel):
    candidate_id: Sha256
    canonical_signature: Sha256
    candidate_bytes: str
    response: str


class CandidateAuditRecord(_FrozenModel):
    candidate_id: Sha256
    semantic_validator_status: Literal["PASS"]
    leakage_audit_status: Literal["PASS"]
    leakage_reason_code: Literal["NO_REGISTERED_COLLISION"]


class RepeatabilityReport(_FrozenModel):
    schema_version: Literal["phase13_legacy_rag_repeatability_v1"]
    status: Literal["PASS"]
    compared_artifact_hashes: dict[str, Sha256]
    first_materialization_sha256: Sha256
    repeat_materialization_sha256: Sha256


class BuildRegistry(_FrozenModel):
    schema_version: Literal["phase13_legacy_rag_build_registry_v1"]
    build_registry_id: str
    task_id: FeasibleTaskName
    build_source_contract_id: Literal["legacy_rag_build_source_contract_v2"]
    canonical_byte_contract_id: Literal["legacy_rag_canonical_bytes_v1"]
    generator: GeneratorIdentity
    calibration_registry_path: str
    calibration_registry_id: str
    calibration_registry_sha256: Sha256
    calibration_signature_hashes: tuple[Sha256, ...]
    calibration_selection_law: str
    build_partition_law: str
    historical_pilot_status: Literal["HISTORICAL_EVIDENCE_ONLY"] | None = None
    leakage_calibration_artifact: ArtifactReference | None = None
    structural_threshold_artifact: ArtifactReference | None = None
    opaque_exclusion_registry_sha256: Sha256
    eligible_candidate_count: Literal[64]
    candidates: tuple[BuildCandidate, ...]
    candidate_audits: tuple[CandidateAuditRecord, ...]
    selected_worked_example_ids: tuple[Sha256, ...]
    partition_disjointness: Literal["PASS"]


class CorpusDocument(_FrozenModel):
    schema_version: Literal["phase13_legacy_rag_clean_document_v1"]
    document_id: str
    task_id: FeasibleTaskName
    semantic_stratum: Literal["A", "B", "C", "D"]
    source_registry_id: str
    construction_pool_id: str
    authoring_or_extraction_rule_id: str
    canonical_byte_contract_id: Literal["legacy_rag_canonical_bytes_v1"]
    review_status: Literal["PASS"]
    leakage_audit_status: Literal["PASS"]
    content_hash: Sha256
    text: str
    build_instance_id: Sha256 | None = None
    build_generator_id: str | None = None
    build_registry_id: str | None = None
    build_registry_sha256: Sha256 | None = None
    generator_implementation_sha256: Sha256 | None = None
    canonical_response_constructor_id: str | None = None
    semantic_validator_status: Literal["PASS"] | None = None


class SerializedDocument(_FrozenModel):
    id: str
    text: str


class SerializedBranchCorpus(_FrozenModel):
    branch: BranchName
    serialization_id: str
    clean_base_hash: Sha256
    documents: tuple[SerializedDocument, ...]
    active_document_ids: tuple[str, ...]


class CorpusBundle(_FrozenModel):
    schema_version: Literal["phase13_legacy_rag_corpus_bundle_v1"]
    task_id: FeasibleTaskName
    clean_documents: tuple[CorpusDocument, ...]
    semantic_registry_sha256: Sha256
    clean_corpus_sha256: Sha256
    triplet_registry: ArtifactReference
    triplet_id: str
    triplet_artifact_hash: Sha256
    branches: dict[BranchName, SerializedBranchCorpus]


class SerializedBranchIndex(_FrozenModel):
    branch: BranchName
    corpus_serialization_id: str
    corpus_content_hash: Sha256
    index_serialization_id: str
    index_artifact_hash: Sha256
    embedding_contract: dict[str, str | int | bool]
    documents: tuple[SerializedDocument, ...]
    vectors: dict[str, tuple[float, ...]]


class EmbeddingRuntimeIdentity(_FrozenModel):
    model_id: Literal["BAAI/bge-m3"]
    revision: Literal["5617a9f61b028005a4858fdac845db406aefb181"]
    embedding_library_version: str
    vector_dimension: Annotated[int, Field(gt=0)]
    normalize_embeddings: Literal[True]


class RetrievalDiagnostic(_FrozenModel):
    status: Literal["PASS"]
    query_document_id: str
    returned_document_ids: tuple[str, str, str]
    scores: tuple[float, float, float]


class IndexBundle(_FrozenModel):
    schema_version: Literal["phase13_legacy_rag_serialized_indices_v1"]
    task_id: FeasibleTaskName
    top_k: Literal[3]
    similarity: Literal["cosine"]
    reranker: None
    score_threshold: None
    tie_break: Literal["document_id_lexical"]
    update_mode: Literal["frozen_read_only"]
    corpus_scope: Literal["same_task_only"]
    embedding_runtime: EmbeddingRuntimeIdentity
    retrieval_diagnostic: RetrievalDiagnostic
    branches: dict[BranchName, SerializedBranchIndex]


class TaskStatus(_FrozenModel):
    status: MaterializationStatus


class PackageStatus(_FrozenModel):
    schema_version: Literal["phase13_legacy_rag_package_status_v1"]
    package_status: MaterializationStatus
    tasks: dict[TaskName, TaskStatus]


class PackageManifest(_FrozenModel):
    schema_version: Literal["phase13_legacy_rag_manifest_v1"]
    package_status: MaterializationStatus
    materialization_profile: Literal["production_bge_m3", "test_only"]
    artifact_hashes: dict[str, Sha256]


class LegacyRagMaterializationReport(_FrozenModel):
    package_status: MaterializationStatus
    tasks: dict[TaskName, TaskStatus]


__all__ = [
    "BRANCHES", "FEASIBLE_TASKS", "TASKS", "ArtifactReference", "BranchName",
    "BuildCandidate", "BuildRegistry", "CandidateAuditRecord", "CorpusBundle", "CorpusDocument",
    "EmbeddingRuntimeIdentity", "FeasibleTaskName", "GeneratorIdentity", "IndexBundle",
    "LegacyRagMaterializationReport", "PackageManifest", "PackageStatus",
    "MATERIALIZED_TASKS", "RepeatabilityReport", "RetrievalDiagnostic", "SerializedBranchCorpus",
    "SerializedBranchIndex", "SerializedDocument", "TaskName", "TaskStatus",
]
