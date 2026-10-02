from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from memcontam.baselines.bot_solve import parse_bot_solve_result
from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import FAILURE_TAXONOMY, BaselineExecutionOutcome
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.evaluation.phase13_observability_registration import (
    AUTHORITY_HASHES,
    ObservabilityRegistrationPacket,
    registered_failure_class,
    registered_verifier_result,
)
from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
from memcontam.experiment.phase13_ordinary_runtime import (
    ProspectiveOrdinaryResult,
    ProspectiveOrdinaryRun,
    _ordered_tasks,
)
from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_production_runtime_evidence import (
    build_production_trial_evidence,
    trial_id,
)
from memcontam.readiness.phase13_production_runtime_models import (
    ProductionNoMemTrialEvidence,
    ProductionOrdinaryRunIdentity,
    ProductionRuntimeJoinError,
)
from memcontam.tasks.base import TaskInstance

if TYPE_CHECKING:
    from .phase13_production_observability import (
        ProductionObservabilityArchive,
    )


def production_archive_from_ordinary(
    run: ProspectiveOrdinaryRun,
    result: ProspectiveOrdinaryResult,
    identity: ProductionOrdinaryRunIdentity,
) -> ProductionObservabilityArchive:
    return _archive_from_ordinary(run, result, identity, partial=False)


def production_archive_from_completed_prefix(
    run: ProspectiveOrdinaryRun,
    result: ProspectiveOrdinaryResult,
    identity: ProductionOrdinaryRunIdentity,
) -> ProductionObservabilityArchive:
    return _archive_from_ordinary(run, result, identity, partial=True)


def _archive_from_ordinary(
    run: ProspectiveOrdinaryRun, result: ProspectiveOrdinaryResult,
    identity: ProductionOrdinaryRunIdentity, *, partial: bool,
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
        or (not result.trials and not partial)
        or len(result.trials) > len(result.sample_ids)
        or (partial and (result.terminal_failure is None or any(
            trial.outcome.status != "succeeded" for trial in result.trials
        )))
        or (not partial and
            len(result.trials) < len(result.sample_ids)
            and (not result.trials or result.trials[-1].outcome.status != "failed")
        )
    ):
        raise ProductionRuntimeJoinError("PRODUCTION_RESULT_IDENTITY_MISMATCH")
    checkpoint_index = (
        None
        if run.branch is None
        else run.branch.checkpoint.checkpoint_index
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
    queries = {task.sample_id: task for task in _ordered_tasks(run)}
    if set(result.sample_ids) - set(queries):
        raise ProductionRuntimeJoinError("PRODUCTION_SAMPLE_ORDER_MISMATCH")
    records = tuple(
        ProductionTrialRecord(
            execution_template_id=identity.execution_template_id,
            run_id=run.run_id,
            session_id=f"{trial_id(run, sample_id, suffix_order)}:session",
            scientific_result=identity.scientific_result,
            ordered_sample_ids_sha256=identity.ordered_sample_ids_sha256,
            request=request,
            parsed_answer=trial.outcome.parsed_answer,
            task_instance=queries[sample_id],
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
                else terminal_provider_evidence(_terminal_method_call(trial.outcome),
                                                trial.outcome.failure_disposition)
            ),
            terminal_failure_code=(None if trial.outcome.status == "succeeded"
                                   else trial.outcome.failure_disposition),
            evidence=_classified_evidence(run, trial, identity, queries[sample_id], sample_id, suffix_order, checkpoint_index),
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


def _classified_evidence(
    run: ProspectiveOrdinaryRun, trial: RuntimeTrialResult, identity: ProductionOrdinaryRunIdentity,
    task: TaskInstance, sample_id: str, suffix_order: int, checkpoint_index: int | None,
) -> Phase13TrialEvidence | ProductionNoMemTrialEvidence:
    evidence = build_production_trial_evidence(run, trial, identity, sample_id, suffix_order, checkpoint_index)
    if evidence.verified_outcome is None or trial.outcome.parsed_answer is None:
        return evidence
    failure = registered_failure_class(task, trial.outcome.parsed_answer, evidence.verified_outcome)
    return evidence.model_copy(update={"trial": evidence.trial.model_copy(update={"failure_class": failure})})


def validate_classifier_joins(
    archive: ProductionObservabilityArchive, packet: ObservabilityRegistrationPacket,
    frozen_tasks: tuple[TaskInstance, ...] | None,
) -> None:
    from .phase13_production_observability import ProductionObservabilityError

    if packet.authority_hashes != AUTHORITY_HASHES or archive.schema_version != "phase13_production_observability_archive_v2":
        return
    if frozen_tasks is None:
        raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
    source = {task.sample_id: task for task in frozen_tasks}
    if len(source) != len(frozen_tasks) or any(
        record.task_instance is None or not record.evidence.trial_id.endswith(f":{record.task_instance.sample_id}")
        or record.task_instance.task_name != record.evidence.task
        or record.task_instance != source.get(record.task_instance.sample_id) for record in archive.records
    ):
        raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
    for record in archive.records:
        if (isinstance(record.evidence, ProductionNoMemTrialEvidence)
            and (len(record.method_calls) != 1 or record.method_calls[0].stage != "no_memory_generate")):
            raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
        if record.parsed_answer is None:
            if record.evidence.trial.execution_status == "completed" or record.evidence.verified_outcome is not None:
                raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
            continue
        answer_call_id = (record.evidence.target_set.answer_call_id
                          if isinstance(record.evidence, Phase13TrialEvidence) else None)
        answer_calls = tuple(call for call in record.method_calls if call.call_id == answer_call_id) if answer_call_id else record.method_calls[-1:]
        if len(answer_calls) != 1 or answer_calls[0].raw_response is None:
            raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
        if isinstance(record.evidence, Phase13TrialEvidence) and record.evidence.baseline == "dc_rs":
            from memcontam.baselines.dynamic_cheatsheet_phase12 import curate_pre_generation

            nodes = {node.entry_id: node for node in record.evidence.lineage}
            synthesis_calls = tuple(call for call in record.method_calls if call.stage == "dc_rs_synthesize")
            unavailable_source = bool(synthesis_calls) and curate_pre_generation(
                synthesis_calls[-1].raw_response or "", fallback_strategy="",
                retrieved_archive_ids=(), strict_whole_response=True,
            ).lineage_status == "unavailable"
            if any(
                (unavailable_source or span.lineage_status in {"unavailable", "approximate"})
                and (span.entry_id not in nodes or nodes[span.entry_id].lineage_status == "exact")
                for span in answer_calls[0].source_spans
            ):
                raise ProductionObservabilityError("PRODUCTION_LINEAGE_STATUS_MISMATCH")
        try:
            raw_answer = answer_calls[0].raw_response
            if record.evidence.baseline == "bot_style":
                raw_answer = parse_bot_solve_result(raw_answer).final_answer
            parsed = parse_final_answer(raw_answer)
        except ValueError as error:
            raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH") from error
        if parsed != record.parsed_answer:
            raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
        if record.evidence.trial.execution_status == "failed":
            expected_failure = next((kind for disposition, (kind, _reason) in FAILURE_TAXONOMY.items()
                                     if disposition == record.terminal_failure_code), None)
            if (record.evidence.verified_outcome is not None or expected_failure is None
                or record.evidence.trial.failure_class != expected_failure
                or record.terminal_provider_evidence is None
                or record.terminal_provider_evidence.trigger_class != "post_response_semantic_failure"):
                raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
            continue
        if record.task_instance is not None and (
            record.evidence.verified_outcome != int(registered_verifier_result(record.task_instance, parsed).is_correct)
        ):
            raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
        if (record.task_instance is not None and record.evidence.verified_outcome is not None
            and record.evidence.trial.failure_class != registered_failure_class(
                record.task_instance, parsed, record.evidence.verified_outcome
            )):
            raise ProductionObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")


def _terminal_method_call(outcome: BaselineExecutionOutcome) -> MethodCall:
    calls = tuple(call for call in outcome.method_calls if isinstance(call, MethodCall))
    call = calls[-1] if calls else None
    if call is None or (call.error_type is None and call.raw_response is None):
        raise ProductionRuntimeJoinError("PRODUCTION_TERMINAL_CALL_REQUIRED")
    return call


__all__ = [
    "ProductionOrdinaryRunIdentity",
    "ProductionRuntimeJoinError",
    "production_archive_from_ordinary",
]
