from __future__ import annotations

from memcontam.experiment.phase12.filter_challenge.mft import MFT_IDS
from memcontam.experiment.phase12.filter_challenge.validation_summary import (
    TASK17_COMMAND_IDS,
    VALIDATION_GATE_IDS,
    Task17CommandRecord,
    Task17ValidationSummary,
    ValidationGateRecord,
)


def complete_validation_summary(
    plan_sha256: str,
    implementation_commit: str,
    command_records: tuple[Task17CommandRecord, ...] | None = None,
) -> Task17ValidationSummary:
    digest = "0" * 64
    return Task17ValidationSummary(
        archive_freeze_id="phase12-filter-v5-build-freeze-v1",
        archive_implementation_commit=implementation_commit,
        archive_search_config_hash=digest,
        bct_execution_status="blocked",
        bct_software_interface_status="ready",
        command_records=command_records or tuple(
            Task17CommandRecord(
                command_id=command_id,
                cwd="<repository>",
                exit_code=0,
                normalized_argv=("phase12", "filter-v5", command_id),
                stderr_sha256=digest,
                stdout_sha256=digest,
            )
            for command_id in TASK17_COMMAND_IDS
        ),
        information_boundary_status="pass",
        implementation_commit=implementation_commit,
        mft_ids=MFT_IDS,
        provider_calls_issued=0,
        quality_worktree_clean=True,
        reviewed_plan_sha256=plan_sha256,
        route_invariance_status="pass",
        validation_gates=tuple(
            ValidationGateRecord(gate_id=gate_id, status="pass")
            for gate_id in VALIDATION_GATE_IDS
        ),
        answer_call_provenance_status="pass",
    )
