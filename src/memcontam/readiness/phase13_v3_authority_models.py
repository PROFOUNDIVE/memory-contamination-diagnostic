from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memcontam.readiness.phase13_cost_policy_models import Capacity, Sha256


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


IdentityComponent = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*-v3$", max_length=200)]
RETIRED_IDENTITIES: Final = frozenset({
    "phase13-main-a-corrected-20260905-v3",
    "phase13-main-a-corrected-execution-freeze-v3",
    "phase13-main-a-corrected-authorized-execution-v3",
    "phase13-main-a-corrected-cost-proof-v3",
})


class IdentityError(ValueError):
    def __init__(self) -> None:
        super().__init__("MAIN_CORRECTED_RUN_ID_MISMATCH")


class V3Identity(FrozenModel):
    run_id: IdentityComponent
    package_id: IdentityComponent
    authorization_id: IdentityComponent
    cost_proof_id: IdentityComponent
    ledger_filename: Literal["main_run_ledger_v3.sqlite3"] = "main_run_ledger_v3.sqlite3"
    conformance_contract_id: Literal["phase13-main-provider-free-conformance-v3"] = "phase13-main-provider-free-conformance-v3"

    @field_validator("run_id", "package_id", "authorization_id", "cost_proof_id")
    @classmethod
    def reject_retired(cls, value: str) -> str:
        if value in RETIRED_IDENTITIES:
            raise IdentityError()
        return value


V3Schema = Literal[
    "phase13_main_authority_snapshot_v3", "phase13_mr_p4_local_closure_manifest_v3",
    "phase13_main_activated_cost_policy_v3", "phase13_main_base_cost_inputs_v3",
    "phase13_main_complete_cost_inputs_v3", "phase13_main_cost_witness_v3",
    "phase13_main_cost_proof_v3", "phase13_main_provider_free_conformance_v3",
    "phase13_main_dispatch_evidence_v3", "phase13_main_reconciliation_v3",
    "phase13_main_run_ledger_v3", "phase13_main_execution_freeze_v3",
    "phase13_main_live_contract_v3", "phase13_main_authorization_v3",
]
Role = Literal[
    "theory", "baseline", "contamination_protocol", "narrow_addendum",
    "experiment", "corrective_scientific", "router", "provenance_only",
]
ROUTED_DOCUMENTS: Final[tuple[tuple[Role, str], ...]] = (
    ("theory", "Phase 13 \u2014 THEORETICAL ARTIFACT revised-v1.md"),
    ("baseline", "Phase 13-Compatible Baseline Memory and Filter Design revised-v5.md"),
    ("contamination_protocol", "Phase 13-Compatible Contamination Construction Intervention Timing and Sensitivity Protocol revised-v8.md"),
    ("narrow_addendum", "2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum_revised-v3.md"),
    ("experiment", "Phase 13-Compatible Pilot Main and Exploratory Experiment Design revised-v12.md"),
    ("corrective_scientific", "2026-09-03_Phase13_MainA_Corrective_Scientific_Decision_Authority.md"),
    ("router", "AGENTS.md"),
)
PROVENANCE_FILENAME: Final = "2026-09-05_Phase13_Input_Envelope_Authority_Revision_Manifest.md"


class DocumentBinding(FrozenModel):
    filename: str = Field(min_length=1)
    role: Role
    size: int = Field(ge=0)
    sha256: Sha256


class AuthorityStage(FrozenModel):
    stage_id: str
    maximum_output_tokens: int = Field(gt=0)
    maximum_input_tokens: int = Field(gt=0)


class V3Registry(FrozenModel):
    registry_id: Literal["CORE_EXECUTION_ENVELOPE_REGISTRY_V3"]
    sha256: Literal["f97e30aa81d71a76a3023792314de606073d9d9215cc612927e69050688269ee"]
    transport_contract_id: Literal["CORE_TRANSPORT_ATTEMPT_CONTRACT_V2"]
    per_request_timeout_seconds: Literal[180]
    max_transport_attempts: Literal[1]
    transport_retries: Literal[0]
    T: Literal[1]
    stages: tuple[AuthorityStage, ...]


class TerminalContract(FrozenModel):
    contract_id: Literal["CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"]
    sha256: Literal["9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"]
    triggers: tuple[str, ...]
    propagation: tuple[str, ...]


class AuthoritySnapshotV3(FrozenModel):
    schema_version: Literal["phase13_main_authority_snapshot_v3"] = "phase13_main_authority_snapshot_v3"
    identity: V3Identity
    documents: tuple[DocumentBinding, ...]
    provenance: DocumentBinding
    registry: V3Registry
    terminal: TerminalContract
    capacity: Capacity
    predecessor_registry_id: Literal["CORE_EXECUTION_ENVELOPE_REGISTRY_V2"] = "CORE_EXECUTION_ENVELOPE_REGISTRY_V2"
    predecessor_rag_input_tokens: Literal[290] = 290
    repository_344_status: Literal["STALE_IMPLEMENTATION_HISTORY"] = "STALE_IMPLEMENTATION_HISTORY"
