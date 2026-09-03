from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.logging.schema_v3 import NoMemTrialLog


class ProductionRuntimeJoinError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ProductionOrdinaryRunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    execution_template_id: str = Field(min_length=1)
    trajectory_seed: int = Field(ge=0, le=9)
    concrete_seed_id: str = Field(pattern=r"^[0-9]$")
    ordered_sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registration_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_result: bool
    analysis_window_id: Literal["core_prefix_50"] = "core_prefix_50"
    source_package_manifest_sha256: str | None = None
    checkpoint_registry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _conformance_seed(self) -> ProductionOrdinaryRunIdentity:
        if self.concrete_seed_id != str(self.trajectory_seed):
            raise ProductionRuntimeJoinError("PRODUCTION_CONCRETE_SEED_MISMATCH")
        return self


class ProductionNoMemTrialEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["phase13_nomem_trial_evidence_v1"] = (
        "phase13_nomem_trial_evidence_v1"
    )
    evidence_scope: Literal["production_runtime"]
    task: Literal[
        "game24",
        "math_equation_balancer",
        "word_sorting",
        "mmlu_pro_engineering",
        "mmlu_pro_physics",
    ]
    baseline: Literal["nomem"]
    trajectory_seed: int = Field(ge=0)
    concrete_seed_id: str = Field(min_length=1)
    analysis_window_id: Literal["core_prefix_50"]
    trial_id: str = Field(min_length=1)
    order_key: int = Field(ge=0)
    trial: NoMemTrialLog
    verified_outcome: Literal[0, 1] | None


__all__ = [
    "ProductionNoMemTrialEvidence",
    "ProductionOrdinaryRunIdentity",
    "ProductionRuntimeJoinError",
]
