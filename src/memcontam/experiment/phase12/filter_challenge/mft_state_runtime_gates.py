from __future__ import annotations

from dataclasses import fields
from typing import Literal

import memcontam.experiment.phase12.filter_challenge.executor as executor
import memcontam.experiment.phase12.filter_challenge.executor_source as executor_source
import memcontam.experiment.phase12.filter_challenge.native_execution as native_execution
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    PairExecutorError,
    PairingIdentity,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import (
    GateEvaluation,
    MftMachineObservation,
    MftStateMutation,
    canonical_hash,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_runtime import (
    ScriptedCallFailure,
    arm_config,
    config_diff,
    runtime_case,
)


def pair_gate(mutation: MftStateMutation) -> GateEvaluation:
    case = runtime_case(mutate_identity=mutation == "pair_identity")
    before = executor_source.source_snapshot(case.request.execution)
    try:
        audit = executor.execute_isolated_pair(case.request)
    except PairExecutorError as error:
        if mutation != "pair_identity" or str(error) != "PAIRED_EXECUTION_IDENTITY_MISMATCH":
            raise
        status: Literal["matched", "mismatched"] = "mismatched"
    else:
        status = audit.paired_execution_identity.paired_execution_identity_status
    after = executor_source.source_snapshot(case.request.execution)
    control_config = arm_config(case.expected_identity, candidate_entry_id=None)
    challenge_config = arm_config(
        case.request.identity, candidate_entry_id=case.request.candidate.candidate_entry_id
    )
    diff = config_diff(control_config, challenge_config)
    expected_challenge = arm_config(
        case.expected_identity, candidate_entry_id=case.request.candidate.candidate_entry_id
    )
    identity_fields = tuple(field.name for field in fields(PairingIdentity))
    expected = MftMachineObservation(
        paired_execution_identity_status="matched",
        paired_identity_fields=identity_fields,
        config_diff_fields=("candidate_entry_id",),
        control_config_hash=canonical_hash(control_config),
        challenge_config_hash=canonical_hash(expected_challenge),
        source_state_before_hash=before.canonical_sha256,
        source_state_after_hash=before.canonical_sha256,
    )
    actual = MftMachineObservation(
        paired_execution_identity_status=status,
        paired_identity_fields=identity_fields,
        config_diff_fields=diff,
        control_config_hash=canonical_hash(control_config),
        challenge_config_hash=canonical_hash(challenge_config),
        source_state_before_hash=before.canonical_sha256,
        source_state_after_hash=after.canonical_sha256,
    )
    return GateEvaluation(
        ("candidate-only-native-diff",), expected, actual, "PAIRED_EXECUTION_IDENTITY_MISMATCH"
    )


def no_writeback_gate(mutation: MftStateMutation) -> GateEvaluation:
    native_case = runtime_case(include_failure=True)
    before = executor_source.source_snapshot(native_case.request.execution)
    native = native_execution.execute_native_pair(
        native_case.request.execution,
        native_case.request.candidate,
        None,
        native_case.request.execution_order,
    )
    after_native = executor_source.source_snapshot(native_case.request.execution)
    try:
        native_case.challenge_client.chat([], "replay", {})
    except ScriptedCallFailure:
        pass

    executor_case = runtime_case()
    audit = executor.execute_isolated_pair(executor_case.request)
    after_executor = executor_source.source_snapshot(executor_case.request.execution)
    source_after = after_executor.canonical_sha256
    if after_native != before:
        source_after = after_native.canonical_sha256
    if mutation == "source_state":
        source_after = canonical_hash({"mutation": "source-state"})
    records = native_case.challenge_client.records
    assessment_penalty = 0 if executor_case.sink.assessments == [audit] else 1
    updater_writes = sum(
        int(arm.updater_enabled or arm.memory_write_event_id is not None)
        for arm in (native.control, native.challenge)
    )
    expected = MftMachineObservation(
        source_state_before_hash=before.canonical_sha256,
        source_state_after_hash=before.canonical_sha256,
        challenge_output_artifact_count=1,
        challenge_failure_artifact_count=1,
        challenge_record_artifact_count=2,
    )
    actual = MftMachineObservation(
        source_state_before_hash=before.canonical_sha256,
        source_state_after_hash=source_after,
        challenge_output_artifact_count=records.count("output"),
        challenge_failure_artifact_count=records.count("failure"),
        challenge_record_artifact_count=len(records) + assessment_penalty,
        active_memory_write_count=int(
            after_native != before or after_executor.canonical_sha256 != before.canonical_sha256
        ),
        ordinary_trial_write_count=len(executor_case.sink.trials) + len(executor_case.sink.calls),
        updater_write_count=updater_writes,
    )
    return GateEvaluation(
        ("challenge-artifact-1", "challenge-artifact-2"), expected, actual, "SOURCE_DRIFT"
    )
