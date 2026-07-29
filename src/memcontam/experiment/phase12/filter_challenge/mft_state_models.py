from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.experiment.phase12.filter_challenge.registry_search import (
    KappaCandidate,
    SuiteCandidate,
)

MFT_STATE_SCHEMA_VERSION: Final = "filter_challenge_mft_state_v1"
MftStateId: TypeAlias = Literal[
    "MFT-FV5-01-PAIR-MATCH",
    "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE",
    "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE",
    "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT",
    "MFT-FV5-08-NO-WRITEBACK",
]
MFT_STATE_IDS: Final[tuple[MftStateId, ...]] = (
    "MFT-FV5-01-PAIR-MATCH",
    "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE",
    "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE",
    "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT",
    "MFT-FV5-08-NO-WRITEBACK",
)
MftStateMutation: TypeAlias = Literal[
    "none", "pair_identity", "exposure", "route", "source_state"
]
JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | tuple["JsonValue", ...]
    | dict[str, "JsonValue"]
)


class StrictMftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MftStateContext(StrictMftModel):
    evidence_layer: Literal["build"] = "build"
    scientific_result: Literal[False] = False
    fixture_only: Literal[True] = True
    search_config_id: str
    search_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    operational_probe_suite_manifest_id: str
    operational_probe_suite_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_candidate: SuiteCandidate
    kappa_candidate: KappaCandidate

    @model_validator(mode="after")
    def _validate_synthetic_contract(self) -> MftStateContext:
        identities = (
            self.search_config_id,
            self.calibration_probe_inventory_id,
            self.operational_probe_suite_manifest_id,
            self.suite_candidate.operational_probe_suite_id,
            self.kappa_candidate.kappa_id,
            *self.suite_candidate.probe_ids,
        )
        if any("synthetic-build" not in identity for identity in identities):
            raise ValueError("MFT_SYNTHETIC_FIXTURE_REQUIRED")
        kappa, suite = self.kappa_candidate, self.suite_candidate
        if (
            kappa.min_distinct_witness_probes > kappa.min_distinct_evaluable_probes
            or kappa.min_distinct_evaluable_probes > len(suite.probe_ids)
            or kappa.min_witness_replicates_per_probe > suite.replicates_per_probe
            or kappa.min_total_evaluable_replicates
            > len(suite.probe_ids) * suite.replicates_per_probe
        ):
            raise ValueError("KAPPA_INCOHERENT")
        return self


class MftGateInputs(StrictMftModel):
    registry_context: MftStateContext
    candidate_entry_id: str
    source_checkpoint_id: str
    source_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_ids: tuple[str, ...]
    provider_calls_issued: Literal[0] = 0


class MftMachineObservation(StrictMftModel):
    paired_execution_identity_status: Literal["matched", "mismatched"] | None = None
    paired_identity_fields: tuple[str, ...] = ()
    config_diff_fields: tuple[str, ...] = ()
    control_config_hash: str | None = None
    challenge_config_hash: str | None = None
    candidate_final_context_inclusions: tuple[bool, ...] = ()
    assessment_states: tuple[str, ...] = ()
    route_targets: tuple[str, ...] = ()
    audit_flags: tuple[bool, ...] = ()
    probe_reason_codes: tuple[str, ...] = ()
    routing_reason_codes: tuple[str, ...] = ()
    scripted_attempt_counts: tuple[int, ...] = ()
    excluded_metadata_hashes: tuple[str, ...] = ()
    policy_input_hashes: tuple[str, ...] = ()
    source_state_before_hash: str | None = None
    source_state_after_hash: str | None = None
    challenge_output_artifact_count: int = 0
    challenge_failure_artifact_count: int = 0
    challenge_record_artifact_count: int = 0
    active_memory_write_count: int = 0
    ordinary_trial_write_count: int = 0
    updater_write_count: int = 0


class MftGateResult(StrictMftModel):
    schema_version: Literal["filter_challenge_mft_state_v1"] = MFT_STATE_SCHEMA_VERSION
    test_id: MftStateId
    execution_index: int = Field(ge=1, le=8)
    inputs: MftGateInputs
    expected: MftMachineObservation
    actual: MftMachineObservation
    reason: str
    status: Literal["pass", "fail"]
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class MftStateReport(StrictMftModel):
    schema_version: Literal["filter_challenge_mft_state_v1"] = MFT_STATE_SCHEMA_VERSION
    evidence_layer: Literal["build"] = "build"
    scientific_result: Literal[False] = False
    fixture_only: Literal[True] = True
    decision_input_kind: Literal["machine_structure"] = "machine_structure"
    ordered_test_ids: tuple[MftStateId, ...]
    results: tuple[MftGateResult, ...]
    provider_calls_issued: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_registry(self) -> MftStateReport:
        if self.ordered_test_ids != MFT_STATE_IDS or tuple(
            result.test_id for result in self.results
        ) != MFT_STATE_IDS:
            raise ValueError("MFT_STATE_REGISTRY_MISMATCH")
        if tuple(result.execution_index for result in self.results) != tuple(range(1, 9)):
            raise ValueError("MFT_STATE_EXECUTION_COUNT_MISMATCH")
        return self


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    scenarios: tuple[str, ...]
    expected: MftMachineObservation
    actual: MftMachineObservation
    failure_reason: str


def canonical_hash(value: JsonValue) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mft_state_evidence_hash(result: MftGateResult) -> str:
    return canonical_hash(result.model_dump(mode="json", exclude={"evidence_hash"}))
