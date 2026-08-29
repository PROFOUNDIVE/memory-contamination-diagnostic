from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Task = Literal[
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
]
Arm = Literal["clean", "correct", "irrelevant", "contam"]
RetrievalBaseline = Literal["rag_frozen", "bot_style", "dc_rs"]


class F1CRowModelError(ValueError):
    pass


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class F1CRuntimeProof(_FrozenModel):
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


class F1CRetrievalRow(_FrozenModel):
    row_id: str = Field(min_length=1)
    task: Task
    baseline: RetrievalBaseline
    arm: Arm
    sample_id: str = Field(min_length=1)
    query_sha256: Sha256
    query_source: str = Field(min_length=1)
    query_source_sha256: Sha256
    state_identity_sha256: Sha256 | None
    corpus_identity_sha256: Sha256 | None
    index_identity_sha256: Sha256
    candidate_ids: tuple[str, ...]
    scores: tuple[float, ...]
    ranks: tuple[int, ...]
    tie_policy: Literal["score_desc_id_lexical"]
    selected_ids: tuple[str, ...]
    threshold: float | None
    top_k: int = Field(ge=1, le=3)
    source_span_ids: tuple[str, ...]
    source_span_join_sha256: Sha256

    @model_validator(mode="after")
    def _aligned(self) -> F1CRetrievalRow:
        if (
            not self.candidate_ids
            or len(self.candidate_ids) != len(self.scores)
            or len(self.scores) != len(self.ranks)
            or self.ranks != tuple(range(1, len(self.ranks) + 1))
            or self.selected_ids != self.source_span_ids
        ):
            raise F1CRowModelError("READINESS0_F1C_ROW_INVALID")
        return self


class F1CReport(_FrozenModel):
    schema_version: Literal["phase13_readiness0_f1c_report_v1"]
    status: Literal["PASS"]
    runtime: F1CRuntimeProof
    row_scope: Literal["ACTIVE_CURRENT_MAIN_RETRIEVAL_ARM_CELLS"]
    row_count: Literal[52]
    rows: tuple[F1CRetrievalRow, ...]
    report_hash: Sha256


__all__ = ["Arm", "F1CReport", "F1CRetrievalRow", "F1CRuntimeProof", "Task"]
