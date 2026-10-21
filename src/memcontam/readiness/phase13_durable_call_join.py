"""Diagnostic-only input preimages and native archive joins; no writer or authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from memcontam.baselines.bot_solve import parse_bot_solve_result
from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import FAILURE_TAXONOMY
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.logging.schema import MethodCall
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_cost_policy_models import Sha256
from memcontam.readiness.phase13_production_observability import validate_production_archive
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import render_model_visible_task

if TYPE_CHECKING:
    from .phase13_main_live_evidence import MemoryUnitEvidence
    from .phase13_main_production import ProductionObject


class DiagnosticJoinError(ValueError):
    def __init__(self, code: str = "DIAGNOSTIC_SCOPE_JOIN_INVALID") -> None:
        super().__init__(code)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DiagnosticFile(_Frozen):
    path: str
    sha256: Sha256


class DiagnosticBranch(_Frozen):
    arm: str
    prefix_identity: str
    source_identity: str
    checkpoint_index: int
    root_count: int
    injected_root_id: str | None
    candidate_triplet_id: str | None
    native_render_id: str | None


class DiagnosticInput(_Frozen):
    tasks: tuple[TaskInstance, ...]
    task_name: str
    baseline: Literal["bot_style", "dc_rs"]
    arm: Literal["clean", "correct", "irrelevant", "contam"]
    model: Literal["gpt-5.6-luna"]
    decoding: dict[str, JsonValue]
    baseline_configs: dict[str, JsonValue]
    trajectory_seed: int
    branch: DiagnosticBranch


class DiagnosticScope(_Frozen):
    provenance: Literal["provider_free_qa"] = "provider_free_qa"
    real_spend_cap_krw: Literal[0] = 0
    run_id: str
    input_utf8: str
    input_sha256: Sha256
    state_utf8: str
    state_sha256: Sha256
    source_files: tuple[DiagnosticFile, ...]
    candidate: DiagnosticFile | None
    archive_identity: ProductionOrdinaryRunIdentity

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()


def validate_diagnostic_join(unit: ProductionObject, evidence: MemoryUnitEvidence,
                             calls: tuple[MethodCall, ...], *, partial: bool = False) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_source_paths

    runtime = evidence.runtime_evidence
    scope = runtime.diagnostic_scope
    archive = runtime.production_observability_archive
    if (type(unit.diagnostic_trial_count) is not int or unit.diagnostic_trial_count != 1
        or scope is None or scope.sha256() != unit.diagnostic_scope_sha256
        or unit.kind != "MEMORY_BEARING" or archive is None or len(archive.records) != 1
        or runtime.production_identity.scientific_result
        or scope.archive_identity != runtime.production_identity):
        raise DiagnosticJoinError()
    inputs = DiagnosticInput.model_validate_json(scope.input_utf8)
    if (len(inputs.tasks) != 1 or inputs.branch.arm != inputs.arm
        or hashlib.sha256(scope.input_utf8.encode()).hexdigest() != scope.input_sha256
        or hashlib.sha256(scope.state_utf8.encode()).hexdigest() != scope.state_sha256
        or unit.unit_id != hashlib.sha256((scope.run_id + scope.sha256()).encode()).hexdigest()
        or unit.prefix_unit_id != hashlib.sha256(inputs.branch.prefix_identity.encode()).hexdigest()
        or unit.checkpoint_registry_sha256 != scope.state_sha256
        or unit.execution_template_id != f"{inputs.task_name}:{inputs.baseline}:{inputs.arm}"
        or (inputs.task_name, inputs.baseline, inputs.arm, inputs.trajectory_seed) != (
            unit.task, unit.memory_baseline, unit.arm, unit.seed)):
        raise DiagnosticJoinError()
    required = snapshot_source_paths(inputs.baseline) | {Path(__file__).resolve()}
    paths = tuple(Path(pin.path) for pin in scope.source_files)
    if len(paths) != len(set(paths)) or not required <= set(paths):
        raise DiagnosticJoinError()
    for pin in (*scope.source_files, *((scope.candidate,) if scope.candidate is not None else ())):
        path = Path(pin.path)
        if path != path.resolve() or hashlib.sha256(read_regular_nofollow(path)).hexdigest() != pin.sha256:
            raise DiagnosticJoinError()
    before = NativeState.from_mapping(json.loads(scope.state_utf8))
    checkpoint = serialize_checkpoint(before, checkpoint_index=inputs.branch.checkpoint_index)
    if runtime.diagnostic_post_state_utf8 is None:
        raise DiagnosticJoinError()
    after = NativeState.from_mapping(json.loads(runtime.diagnostic_post_state_utf8))
    serialize_checkpoint(after)
    if (before.baseline != inputs.baseline or after.baseline != inputs.baseline
        or checkpoint.canonical_sha256 != scope.state_sha256
        or (evidence.consumed_checkpoint_id, evidence.consumed_checkpoint_identity_sha256,
            evidence.consumed_checkpoint_canonical_sha256) != (
                checkpoint.identity.checkpoint_id, checkpoint.identity.sha256, scope.state_sha256)):
        raise DiagnosticJoinError()
    row = archive.records[0]
    trial = row.evidence
    task = inputs.tasks[0]
    expected_id = f"{scope.run_id}{'' if inputs.arm == 'clean' else ':' + inputs.arm}:trial:1:{task.sample_id}"
    sample_hash = hashlib.sha256(json.dumps([task.sample_id], separators=(",", ":")).encode()).hexdigest()
    if (not isinstance(trial, Phase13TrialEvidence) or row.scientific_result
        or row.task_instance != task or row.run_id != scope.run_id
        or row.session_id != f"{expected_id}:session" or trial.trial_id != expected_id
        or row.execution_template_id != unit.execution_template_id
        or row.ordered_sample_ids_sha256 != sample_hash or sample_hash != unit.ordered_sample_ids_sha256
        or row.request != runtime.request or row.method_calls != calls
        or (trial.task, trial.baseline, trial.trajectory_seed, trial.concrete_seed_id) != (
            unit.task, unit.memory_baseline, unit.seed, str(unit.seed))
        or trial.analysis_window_id != scope.archive_identity.analysis_window_id
        or trial.order_key != inputs.branch.checkpoint_index + 1
        or trial.trial.absolute_trial_index != trial.order_key or trial.trial.event_time != 0
        or trial.trial.checkpoint_id != checkpoint.identity.checkpoint_id
        or trial.trial.checkpoint_index != inputs.branch.checkpoint_index
        or trial.trial.prefix_run_id != inputs.branch.prefix_identity
        or trial.trial.branch_id != unit.arm or trial.trial.execution_key.arm != unit.arm
        or trial.trial.candidate_triplet_id_or_none != inputs.branch.candidate_triplet_id
        or trial.trial.native_render_id_or_none != inputs.branch.native_render_id
        or trial.target_set.source_package_manifest_sha256 != scope.archive_identity.source_package_manifest_sha256
        or trial.target_set.target_entry_ids != ((inputs.branch.injected_root_id,)
            if inputs.arm == "contam" and inputs.branch.injected_root_id is not None else ())
        or trial.memory_before_ids != tuple(entry.entry_id for entry in before.entries if isinstance(entry, NativeEntry))
        or trial.memory_after_ids != tuple(entry.entry_id for entry in after.entries if isinstance(entry, NativeEntry))
        or (trial.trial.execution_status == "failed") != partial):
        raise DiagnosticJoinError()
    stages = {"bot_style": ("bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill"),
              "dc_rs": ("dc_rs_synthesize", "dc_rs_generate")}[inputs.baseline]
    call_prefix = (f"{scope.run_id}:{task.task_name}:{task.sample_id}:bot_style:{inputs.arm}:{inputs.model}"
                   if inputs.baseline == "bot_style" else expected_id)
    if (not calls or len(calls) > len(stages) or tuple(call.stage for call in calls) != stages[:len(calls)]
        or (not partial and len(calls) != len(stages))
        or tuple(call.call_id for call in calls) != tuple(f"{call_prefix}:call:{index}" for index in range(1, len(calls) + 1))):
        raise DiagnosticJoinError()
    answer_stage = stages[1]
    answers = tuple(call for call in calls if call.stage == answer_stage)
    expected_answer_id = (answers[0].call_id if answers else None)
    if partial and inputs.baseline == "dc_rs":
        expected_answer_id = None
    if trial.target_set.answer_call_id != expected_answer_id:
        raise DiagnosticJoinError()
    if answers:
        answer = answers[0]
        visible = render_model_visible_task(task)
        marker = f"\n\n{visible}\n\n" if inputs.baseline == "bot_style" else f"Current task:\n{visible}\n\nCheatsheet:\n"
        if not any(marker in message["content"] and all(
            span.message_index != index or span.end <= message["content"].index(marker)
            or span.start >= message["content"].index(marker) + len(marker) for span in answer.source_spans
        ) for index, message in enumerate(answer.messages)):
            raise DiagnosticJoinError()
        raw_answer = answer.raw_response
        if raw_answer is None and (not partial or answer.error_type is None or row.parsed_answer is not None):
            raise DiagnosticJoinError()
        if raw_answer is not None:
            if inputs.baseline == "bot_style":
                raw_answer = parse_bot_solve_result(raw_answer).final_answer
            if parse_final_answer(raw_answer) != row.parsed_answer:
                raise DiagnosticJoinError()
        for span in answer.source_spans:
            content = answer.messages[span.message_index]["content"][span.start:span.end]
            if hashlib.sha256(content.encode()).hexdigest() != span.rendered_hash:
                raise DiagnosticJoinError()
    packet_path = Path(__file__).resolve().parents[3] / "data/phase13/observability/registration_packet_v2.json"
    packet_raw = read_regular_nofollow(packet_path)
    if hashlib.sha256(packet_raw).hexdigest() != unit.registration_packet_sha256:
        raise DiagnosticJoinError()
    packet = ObservabilityRegistrationPacket.model_validate_json(packet_raw)
    validate_production_archive(archive, packet, hashlib.sha256(packet_raw).hexdigest(),
                                frozen_tasks=None if partial else inputs.tasks)
    if partial:
        terminal = calls[-1]
        if (row.terminal_method_call != terminal or terminal.error_type is None
            or terminal.model != runtime.request.model
            or terminal.provider_status not in {None, "failed"}
            or terminal.provider_response_status not in {None, "failed"}
            or row.terminal_failure_code != "provider_call_failed"
            or trial.trial.failure_class != FAILURE_TAXONOMY["provider_call_failed"][0]
            or row.terminal_provider_evidence is None
            or row.terminal_provider_evidence.trigger_class != "provider_call_failure"
            or terminal.raw_response is not None or terminal.transport_attempts != 1 or terminal.retry_count != 0
            or terminal.provider_cost_usd is not None or terminal.token_usage or terminal.provider_usage is not None
            or terminal.authoritative_provider_cost_usd is not None
            or terminal.derived_cost_usd is not None or terminal.provider_cost_source is not None
            or trial.verified_outcome is not None or before != after
            or runtime.verifier_result is not None):
            raise DiagnosticJoinError()
    elif runtime.verifier_result != bool(trial.verified_outcome):
        raise DiagnosticJoinError()
