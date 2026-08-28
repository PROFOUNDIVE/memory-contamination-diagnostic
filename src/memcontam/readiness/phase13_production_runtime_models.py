from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    scientific_result: Literal[False]
    analysis_window_id: Literal["core_prefix_50"] = "core_prefix_50"
    source_package_manifest_sha256: str | None = None
    checkpoint_registry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _conformance_seed(self) -> ProductionOrdinaryRunIdentity:
        if self.concrete_seed_id != str(self.trajectory_seed):
            raise ProductionRuntimeJoinError("PRODUCTION_CONCRETE_SEED_MISMATCH")
        return self


__all__ = ["ProductionOrdinaryRunIdentity", "ProductionRuntimeJoinError"]
