from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.experiment.phase13_ordinary_runtime import (
    ProspectiveOrdinaryResult,
    ProspectiveOrdinaryRun,
)
from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_production_runtime_evidence import (
    build_production_trial_evidence,
    trial_id,
)
from memcontam.readiness.phase13_production_runtime_models import (
    ProductionOrdinaryRunIdentity,
    ProductionRuntimeJoinError,
)

if TYPE_CHECKING:
    from memcontam.readiness.phase13_production_observability import (
        ProductionObservabilityArchive,
    )


def production_archive_from_ordinary(
    run: ProspectiveOrdinaryRun,
    result: ProspectiveOrdinaryResult,
    identity: ProductionOrdinaryRunIdentity,
) -> ProductionObservabilityArchive:
    from memcontam.readiness.phase13_production_observability import (
        ProductionObservabilityArchive,
        ProductionTrialRecord,
        ProviderRequestRecord,
        terminal_provider_evidence,
    )

    if run.model != "gpt-5.6-luna":
        raise ProductionRuntimeJoinError("PRODUCTION_PROVIDER_MODEL_REQUIRED")
    if run.baseline != "nomem" and run.branch is None:
        raise ProductionRuntimeJoinError("PRODUCTION_CHECKPOINT_REQUIRED")
    if run.baseline == "nomem" and run.branch is not None:
        raise ProductionRuntimeJoinError("PRODUCTION_NOMEM_CHECKPOINT_FORBIDDEN")
    if identity.scientific_result is not True:
        raise ProductionRuntimeJoinError("PRODUCTION_SCIENTIFIC_RESULT_REQUIRED")
    if run.trajectory_seed is None or run.trajectory_seed != identity.trajectory_seed:
        raise ProductionRuntimeJoinError("PRODUCTION_TRAJECTORY_SEED_MISMATCH")
    ordered_sample_ids_sha256 = hashlib.sha256(
        json.dumps(result.sample_ids, separators=(",", ":")).encode()
    ).hexdigest()
    if identity.ordered_sample_ids_sha256 != ordered_sample_ids_sha256:
        raise ProductionRuntimeJoinError("PRODUCTION_SAMPLE_ORDER_MISMATCH")
    if (
        result.task_name != run.task_name
        or result.baseline != run.baseline
        or result.arm != run.arm
        or not result.trials
        or len(result.trials) > len(result.sample_ids)
        or (
            len(result.trials) < len(result.sample_ids)
            and result.trials[-1].outcome.status != "failed"
        )
    ):
        raise ProductionRuntimeJoinError("PRODUCTION_RESULT_IDENTITY_MISMATCH")
    checkpoint_index = (
        None
        if run.branch is None
        else run.branch.checkpoint.state.native_state.get("checkpoint_index")
    )
    if run.branch is not None and (type(checkpoint_index) is not int or checkpoint_index < 0):
        raise ProductionRuntimeJoinError("PRODUCTION_CHECKPOINT_INDEX_REQUIRED")
    request = ProviderRequestRecord(
        api="OpenAI Responses API",
        model="gpt-5.6-luna",
        service_tier="default",
        reasoning_mode="standard",
        reasoning_effort="none",
        reasoning_context="current_turn",
        previous_response_id=None,
        store=False,
        timeout_seconds=180,
        retries_after_initial_attempt=0,
        semantic_invalid_generic_retry=False,
    )
    records = tuple(
        ProductionTrialRecord(
            execution_template_id=identity.execution_template_id,
            run_id=run.run_id,
            session_id=f"{trial_id(run, sample_id, suffix_order)}:session",
            scientific_result=identity.scientific_result,
            ordered_sample_ids_sha256=identity.ordered_sample_ids_sha256,
            request=request,
            parsed_answer=trial.outcome.parsed_answer,
            method_calls=tuple(
                call for call in trial.outcome.method_calls if isinstance(call, MethodCall)
            ),
            terminal_method_call=(
                None
                if trial.outcome.status == "succeeded"
                else _terminal_method_call(trial.outcome)
            ),
            terminal_provider_evidence=(
                None
                if trial.outcome.status == "succeeded"
                else terminal_provider_evidence(_terminal_method_call(trial.outcome))
            ),
            evidence=build_production_trial_evidence(
                run,
                trial,
                identity,
                sample_id,
                suffix_order,
                checkpoint_index,
            ),
        )
        for suffix_order, (sample_id, trial) in enumerate(
            zip(result.sample_ids, result.trials, strict=False), start=1
        )
    )
    return ProductionObservabilityArchive(
        schema_version="phase13_production_observability_archive_v2",
        registration_packet_sha256=identity.registration_packet_sha256,
        u_t_status="NOT_REGISTERED_FOR_CURRENT_MAIN",
        records=records,
    )


def _terminal_method_call(outcome: BaselineExecutionOutcome) -> MethodCall:
    calls = tuple(call for call in outcome.method_calls if isinstance(call, MethodCall))
    matching = tuple(call for call in calls if call.call_id == outcome.answer_call_id)
    call = matching[-1] if matching else (calls[-1] if calls else None)
    if call is None or call.error_type is None:
        raise ProductionRuntimeJoinError("PRODUCTION_TERMINAL_CALL_REQUIRED")
    return call


__all__ = [
    "ProductionOrdinaryRunIdentity",
    "ProductionRuntimeJoinError",
    "production_archive_from_ordinary",
]
