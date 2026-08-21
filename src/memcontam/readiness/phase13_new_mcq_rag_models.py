from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

TASKS = ("mmlu_pro_engineering", "mmlu_pro_physics")
BRANCHES = ("clean", "correct", "irrelevant", "contam")
BranchName = Literal["clean", "correct", "irrelevant", "contam"]
EXPECTED_CLASSES = (
    "complete_source_eligibility_registry",
    "accepted_document_registry",
    "verified_embedding_runtime_artifact",
    "serialized_branch_index_artifacts",
    "complete_leakage_evidence",
    "retained_h2_intervention_registry",
    "accepted_h2_selection_record",
)


class FrozenArtifactError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AcceptedDocument(_FrozenModel):
    schema_version: Literal["new_mcq_rag_clean_doc_v1"]
    document_id: str
    task_id: str
    semantic_stratum: str
    document_ordinal: int
    text: str
    source_registry_ids: tuple[str, ...]
    authoring_template_id: Literal["new_mcq_procedural_atomic_v1"]
    review_status: Literal["accepted"]
    content_hash: str


class InterventionDocument(_FrozenModel):
    document_id: str
    task_id: str
    role: Literal["correct", "irrelevant", "contam"]
    semantic_id: str
    text: str
    source_registry_ids: tuple[Literal["phase13_protocol_revised_v8"], ...]
    content_hash: str


class TaskInterventions(_FrozenModel):
    selected_candidate_id: Literal["MCQ-H2-DETAIL-LENGTH-v1"]
    candidate_family_status: Literal["ACCEPTED_H2"]
    documents: dict[Literal["correct", "irrelevant", "contam"], InterventionDocument]


class InterventionRegistry(_FrozenModel):
    schema_version: Literal["phase13_new_mcq_rag_intervention_registry_v1"]
    protocol_authority_sha256: Literal[
        "022879f559b145e30e645b6ccbd139e9927899d370f1956d27a0562580acf85f"
    ]
    experiment_authority_sha256: Literal[
        "4b1db4e55e68ec8e00fe022b9bea1685bebb340138df0e39fddc7823aafdc374"
    ]
    authority_selection_sha256: str
    authority_stack: tuple[
        Literal[
            "phase13_theory_revised_v1",
            "phase13_baseline_revised_v5",
            "phase13_protocol_revised_v8",
            "phase13_experiment_revised_v8",
        ],
        ...,
    ]
    tasks: dict[str, TaskInterventions]


class AuthoritySelection(_FrozenModel):
    schema_version: Literal["phase13_new_mcq_rag_authority_selection_v1"]
    protocol_authority_sha256: Literal[
        "022879f559b145e30e645b6ccbd139e9927899d370f1956d27a0562580acf85f"
    ]
    selection_law: Literal["accepted_continuation_state_2026_08_21"]
    task_selections: dict[str, Literal["MCQ-H2-DETAIL-LENGTH-v1"]]
    gpqa_extension_status: Literal["H2_BLINDED_PLAUSIBILITY_GATE_FAILED_EXTENSION_ONLY"]


class SerializedBranchIndex(_FrozenModel):
    branch: BranchName
    corpus_serialization_id: str
    corpus_content_hash: str
    index_serialization_id: str
    index_artifact_hash: str
    embedding_contract: dict[str, str | int | bool]
    documents: tuple[dict[str, str], ...]
    vectors: dict[str, tuple[float, ...]]


class SerializedIndexBundle(_FrozenModel):
    schema_version: Literal["new_mcq_rag_serialized_branch_indices_v1"]
    task_id: str
    top_k: Literal[3]
    branches: dict[BranchName, SerializedBranchIndex]


__all__ = [
    "BRANCHES",
    "EXPECTED_CLASSES",
    "TASKS",
    "AcceptedDocument",
    "AuthoritySelection",
    "BranchName",
    "FrozenArtifactError",
    "InterventionRegistry",
    "SerializedBranchIndex",
    "SerializedIndexBundle",
    "TaskInterventions",
]
