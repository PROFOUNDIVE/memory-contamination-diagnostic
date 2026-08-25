from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.evaluation.phase13_observability import (
    Phase13Aggregate,
    Phase13AggregateTrial,
    Phase13TrialAnalysis,
    Phase13TrialEvidence,
)
from memcontam.logging.schema import TargetContaminationSetSpec


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, Field(min_length=1)]


class Phase13ObservabilityModelError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactIdentity(_FrozenModel):
    path: str = Field(min_length=1)
    sha256: Sha256


class TargetSetRegistry(_FrozenModel):
    schema_version: Literal["phase13_target_set_registry_v1"]
    registry_id: Literal["phase13_recorded_exact_lineage_targets_v1"]
    scope: Literal["synthetic_contract_fixture_only"]
    source_package_manifest: ArtifactIdentity
    target_sets: tuple[TargetContaminationSetSpec, ...]


class Phase13ObservabilityFixture(_FrozenModel):
    schema_version: Literal["phase13_observability_fixture_v1"]
    fixture_id: Literal["phase13_main_disjoint_observability_fixture_v1"]
    provider_backed_calls: Literal[0]
    concrete_seed_ids: tuple[NonEmptyString, ...] = Field(min_length=10, max_length=10)
    trials: tuple[Phase13TrialEvidence, ...]
    aggregate_templates: tuple[Phase13AggregateTrial, ...]

    @model_validator(mode="after")
    def _unique_seed_ids(self) -> Phase13ObservabilityFixture:
        if len(set(self.concrete_seed_ids)) != 10:
            raise Phase13ObservabilityModelError("DUPLICATE_CONCRETE_SEED_ID")
        if any(
            row.trajectory_seed >= 10
            or row.concrete_seed_id != self.concrete_seed_ids[row.trajectory_seed]
            for row in self.trials
        ) or any(
            row.trajectory_seed >= 10
            or row.concrete_seed_id != self.concrete_seed_ids[row.trajectory_seed]
            for row in self.aggregate_templates
        ):
            raise Phase13ObservabilityModelError("FIXTURE_SEED_REGISTRY_MISMATCH")
        return self


class Phase13Reconstruction(_FrozenModel):
    trials: tuple[Phase13TrialAnalysis, ...]
    aggregate: Phase13Aggregate


class Phase13ObservabilityManifest(_FrozenModel):
    schema_version: Literal["phase13_observability_manifest_v1"]
    manifest_id: Literal["phase13_observability_measurement_identity_v1"]
    evidence_scope: Literal["synthetic_contract_fixture_only"]
    track2_5_status: Literal["BLOCKED"]
    artifacts: dict[str, ArtifactIdentity]
    implementations: dict[str, ArtifactIdentity]
    expected_reconstruction_sha256: Sha256
    failure_classifier_registry_status: Literal["NOT_REGISTERED_BY_AUTHORITY"]
    u_t_status: Literal["NOT_REGISTERED_FOR_CURRENT_MAIN"]
    blockers: tuple[
        Literal["FAILURE_CLASSIFIER_REGISTRY_NOT_REGISTERED"],
        Literal["RECURRENCE_LOOKBACK_NOT_REGISTERED"],
        Literal["EXPOSURE_CONDITIONING_RULE_NOT_REGISTERED"],
        Literal["POST_EVICTION_TIMING_NOT_REGISTERED"],
        Literal["RETENTION_DURATION_ENDPOINT_NOT_REGISTERED"],
        Literal["PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"],
        Literal["CONCRETE_MAIN_SEED_REGISTRY_NOT_FROZEN"],
        Literal["LEVEL2_FH_INTERACTIONS_NOT_MATERIALIZED"],
    ]
    mr_p4_prerequisite_status: Literal["BLOCKED"]
    mr_p5_handoff_status: Literal["NOT_AVAILABLE"]
    main_a_measured_scientific_execution_count: Literal[0]


class Phase13ObservabilityReport(_FrozenModel):
    manifest_id: str
    manifest_sha256: Sha256
    evidence_scope: Literal["synthetic_contract_fixture_only"]
    track2_5_status: Literal["BLOCKED"]
    reconstruction_sha256: Sha256
    repeat_reconstruction_sha256: Sha256
    reconstructed_trial_count: int = Field(ge=0)
    target_set_registry_id: str
    failure_classifier_registry_status: str
    u_t_status: str
    blockers: tuple[str, ...]
    mr_p4_prerequisite_status: Literal["BLOCKED"]
    mr_p5_handoff_status: Literal["NOT_AVAILABLE"]
    main_a_measured_scientific_execution_count: Literal[0]


__all__ = [
    "ArtifactIdentity",
    "Phase13ObservabilityFixture",
    "Phase13ObservabilityManifest",
    "Phase13ObservabilityModelError",
    "Phase13ObservabilityReport",
    "Phase13Reconstruction",
    "Sha256",
    "TargetSetRegistry",
]
