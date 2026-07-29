from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.experiment.phase12.filter_challenge.mft import MFT_IDS


TASK17_COMMAND_IDS: Final = (
    "validate-search-config", "validate-selected-policy", "mft", "build-archive",
    "validate-archive", "cost-preview", "bct-readiness",
)
VALIDATION_GATE_IDS: Final = ("ruff", "mypy", "diff-check")
_SHA256 = r"^[0-9a-f]{64}$"
_GIT_SHA = r"^[0-9a-f]{40}$"


class Task17CommandRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command_id: str
    cwd: str
    exit_code: Literal[0]
    normalized_argv: tuple[str, ...]
    stderr_sha256: str = Field(pattern=_SHA256)
    stdout_sha256: str = Field(pattern=_SHA256)


class ValidationGateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: str
    status: Literal["pass"]


class Task17ValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_freeze_id: Literal["phase12-filter-v5-build-freeze-v1"]
    archive_implementation_commit: str
    archive_search_config_hash: str = Field(pattern=_SHA256)
    bct_execution_status: Literal["blocked"]
    bct_software_interface_status: Literal["ready"]
    command_records: tuple[Task17CommandRecord, ...]
    information_boundary_status: Literal["pass"]
    initial_head: str = Field(pattern=_GIT_SHA)
    implementation_commit: str
    mft_ids: tuple[str, ...]
    provider_calls_issued: Literal[0]
    quality_findings: tuple[()] = ()
    quality_worktree_clean: Literal[True]
    reviewed_plan_sha256: str = Field(pattern=_SHA256)
    route_invariance_status: Literal["pass"]
    validation_gates: tuple[ValidationGateRecord, ...]
    answer_call_provenance_status: Literal["pass"]

    @model_validator(mode="after")
    def validate_contract(self) -> Task17ValidationSummary:
        if tuple(record.command_id for record in self.command_records) != TASK17_COMMAND_IDS:
            raise ValueError("command_records must use the exact Task 17 order")
        if tuple(gate.gate_id for gate in self.validation_gates) != VALIDATION_GATE_IDS:
            raise ValueError("validation_gates must use the exact Task 17 order")
        if self.mft_ids != MFT_IDS:
            raise ValueError("mft_ids must use the exact MFT order")
        return self
