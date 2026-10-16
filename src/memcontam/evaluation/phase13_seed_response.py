from __future__ import annotations

import hashlib
import inspect
import json
import hmac
import secrets
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Literal
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.evaluation.phase13_observability_registration import AUTHORITY_HASHES, ObservabilityRegistrationPacket
from memcontam.experiment.phase13_ordinary_runtime import REPOSITORY_ROOT, ProspectiveOrdinaryResult, ProspectiveOrdinaryRun, _context, _ordered_tasks, execute_prospective_ordinary
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.game24_runner import RuntimeWriterCallbacks
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
from memcontam.contamination.phase12.models import CandidateRegistry
from memcontam.contamination.phase12.registry import load_current_candidate_registry, validate_current_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, Phase12Checkpoint, deserialize_checkpoint, serialize_checkpoint
from memcontam.readiness.phase13_capacity_realization import validated_common_capacity_tokens
from memcontam.readiness.phase13_main_new_mcq_runtime import (
    build_new_mcq_live_branches, load_new_mcq_runtime_registry,
)
from memcontam.readiness.phase13_main_live_runtime_support import core_task_name
from memcontam.readiness.phase13_route_capacity import capacity_contract_error
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityArchive,
    validate_production_archive,
)
from memcontam.readiness.phase13_production_runtime_join import production_archive_from_ordinary
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
from memcontam.readiness.phase13_v3_terminal_models import parse_event
from memcontam.readiness.phase13_v3_request import RequestKeyV3
from memcontam.readiness.phase13_v3_request import STAGES
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_v3_cost_models import CostUnit
from memcontam.readiness.phase13_v3_cost_models import ActivatedPolicyV3
from memcontam.readiness.phase13_v3_retry import schedule_unit
from memcontam.readiness.phase13_v3_entrypoint import SelectedExecutionV3
from memcontam.readiness.phase13_main_live_evidence import _BASELINE_STAGES
from memcontam.tasks.dispatch import render_model_visible_task
from memcontam.tasks.base import TaskInstance
from memcontam.clients.base import LLMClient, LLMResponse


class SeedResponseError(ValueError):
    pass


_PRODUCTION_ORIGIN_KEY = secrets.token_bytes(32)


def _production_origin_signature(parts: tuple[bytes, ...]) -> bytes:
    return hmac.digest(_PRODUCTION_ORIGIN_KEY, b"".join(
        len(part).to_bytes(8, "big") + part for part in parts), "sha256")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class StructuralSupportEvidence(_Frozen):
    task: str
    baseline: str
    trajectory_seed: int = Field(ge=0, le=9)
    concrete_seed_id: str
    registered_pair: tuple[str, str] | None = None
    supported: bool
    source_package_manifest_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_id: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_index: int | None
    ordered_sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_template_id: str
    registration_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_sha256: str | None = None
    branch_checkpoint_sha256: tuple[str, str, str, str]
    candidate_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suffix_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    handler_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_properties_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conformance_trace: tuple[tuple[str, int, str, str, str, str], ...] = ()
    replay_witness: tuple[tuple[str, str, str, str, str, int | None], ...] = ()
    unsupported_reason: str | None = None
    qualification_sha256: str | None = None
    subject_contract_sha256: str | None = None


@dataclass(frozen=True)
class StructuralSupportInputs:
    prefix: Phase12Checkpoint
    context: Game24RuntimeContext
    candidates: CandidateRegistry
    run: ProspectiveOrdinaryRun
    identity: ProductionOrdinaryRunIdentity
    fixture_task_bytes: tuple[bytes, ...] = ()
    qualification_observer: Callable[[str, int, RuntimeTrialResult], None] | None = None


@dataclass(frozen=True, slots=True)
class StructuralSupportValidationContext:
    """Client-free fixture qualification bindings.

    Expected identities must not be populated from the submitted proof.
    This value does not authenticate production qualification origin.
    """
    qualified_evidence_bytes: bytes
    qualified_evidence_sha256: str
    identity: ProductionOrdinaryRunIdentity
    package_sha256: str | None
    prefix: Phase12Checkpoint
    fixture_contract_sha256: str
    branch_checkpoint_sha256: tuple[str, str, str, str]
    candidate_registry_sha256: str
    production_origin: _ProductionStructuralOrigin | None = None


@dataclass(frozen=True, slots=True)
class _ProductionStructuralOrigin:
    """Constructor-owned authenticated bytes, never caller-supplied expectations.

    The qualification is a bounded native capability contract. Its fixture traces
    are deliberately not copied into the production subject's future trajectory.
    """
    proof_bytes: bytes
    prefix_bytes: bytes
    identity_bytes: bytes
    package_sha256: str
    qualification_bytes: bytes
    subject_bytes: bytes
    constructor_seal: bytes = b""


def structural_support_context(selected: SelectedExecutionV3, unit: ProductionObject,
                               ledger: TerminalLedgerV3, *, scientific_result: bool = True) -> StructuralSupportValidationContext | None:
    from memcontam.readiness.phase13_structural_qualification import production_structural_context
    return production_structural_context(selected, unit, ledger, scientific_result=scientific_result)


def validate_qualified_structural_support(proof: StructuralSupportEvidence,
                                         context: StructuralSupportValidationContext) -> None:
    if hashlib.sha256(context.qualified_evidence_bytes).hexdigest() != context.qualified_evidence_sha256:
        raise SeedResponseError("STRUCTURAL_QUALIFICATION_BINDING_MISMATCH")
    qualified = StructuralSupportEvidence.model_validate_json(context.qualified_evidence_bytes)
    identity = context.identity
    if context.production_origin is not None:
        origin = context.production_origin
        expected_seal = _production_origin_signature((origin.proof_bytes, origin.prefix_bytes,
            origin.identity_bytes, origin.package_sha256.encode(), origin.qualification_bytes, origin.subject_bytes))
        if not hmac.compare_digest(origin.constructor_seal, expected_seal):
            raise SeedResponseError("STRUCTURAL_QUALIFICATION_ORIGIN_REQUIRED")
        from memcontam.readiness.phase13_structural_qualification import validate_production_context
        validate_production_context(proof, context, origin)
        return
    if identity.scientific_result:
        raise SeedResponseError("STRUCTURAL_QUALIFICATION_ORIGIN_REQUIRED")
    deserialize_checkpoint(context.prefix)
    evidence_hash = hashlib.sha256(json.dumps(
        qualified.model_dump(mode="json", exclude={"evidence_sha256"}),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (proof != qualified
        or qualified.evidence_sha256 != evidence_hash
        or (context.package_sha256 is None and identity.source_package_manifest_sha256 is None)
        or proof.baseline != context.prefix.state.baseline
        or proof.source_package_manifest_sha256 != identity.source_package_manifest_sha256
        or proof.package_sha256 != context.package_sha256
        or proof.registration_packet_sha256 != identity.registration_packet_sha256
        or proof.execution_template_id != identity.execution_template_id
        or proof.trajectory_seed != identity.trajectory_seed
        or proof.concrete_seed_id != identity.concrete_seed_id
        or proof.ordered_sample_ids_sha256 != identity.ordered_sample_ids_sha256
        or proof.checkpoint_id != context.prefix.identity.checkpoint_id
        or proof.checkpoint_sha256 != context.prefix.canonical_sha256
        or proof.checkpoint_index != context.prefix.checkpoint_index
        or proof.fixture_contract_sha256 != context.fixture_contract_sha256
        or proof.branch_checkpoint_sha256 != context.branch_checkpoint_sha256
        or proof.candidate_registry_sha256 != context.candidate_registry_sha256
        or proof.handler_contract_sha256 != _handler_contract(proof.baseline)
        or proof.native_properties_sha256 != _native_properties_contract()):
        raise SeedResponseError("STRUCTURAL_QUALIFIED_CONTEXT_MISMATCH")
    if context.package_sha256 is not None:
        raise SeedResponseError("STRUCTURAL_QUALIFICATION_ORIGIN_REQUIRED")
    trace = proof.conformance_trace
    slots = tuple((arm, index) for arm in ("clean", "correct", "irrelevant", "contam")
                  for index in range(1, 51))
    if (proof.supported != (proof.unsupported_reason is None)
        or len(trace) > len(slots)
        or (proof.supported and len(trace) != len(slots))
        or hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest()
           != proof.suffix_contract_sha256
        or any(row[:2] != slot for row, slot in zip(trace, slots))
        or any(row[0] == previous[0] and row[4] != previous[5]
               for previous, row in zip(trace, trace[1:]))):
        raise SeedResponseError("STRUCTURAL_QUALIFIED_TRACE_MISMATCH")


def _native_digest(state: NativeState, index: int | None) -> str:
    return serialize_checkpoint(state, checkpoint_index=index).canonical_sha256


def _validate_native_transition(before: NativeState, after: NativeState,
                                stages: tuple[str, ...]) -> None:
    if before.baseline != after.baseline:
        raise SeedResponseError("STRUCTURAL_NATIVE_BASELINE_CHANGED")
    baseline = before.baseline
    if baseline == "rag_frozen":
        if before != after or before.native_state.get("read_only") is not True:
            raise SeedResponseError("STRUCTURAL_RAG_IMMUTABILITY_VIOLATION")
    elif baseline == "bot_style":
        if (before.native_state.get("active_capacity") is not None
            or after.native_state.get("active_capacity") is not None
            or after.entries[:len(before.entries)] != before.entries
            or len(after.entries) not in (len(before.entries), len(before.entries) + 1)):
            raise SeedResponseError("STRUCTURAL_BOT_INSERTION_VIOLATION")
    elif baseline == "fh_bounded":
        retained = after.entries[:-1]
        if (not after.entries or len(retained) > len(before.entries)
            or (retained and retained != before.entries[-len(retained):])
            or after.entries[-1] in before.entries):
            raise SeedResponseError("STRUCTURAL_FH_FIFO_VIOLATION")
    elif baseline == "reflexion_style":
        reflections = stages.count("reflexion_reflect")
        if reflections == 0:
            if before != after:
                raise SeedResponseError("STRUCTURAL_REFLEXION_SUCCESS_WRITE")
        else:
            old_ids = {entry.entry_id for entry in before.entries if isinstance(entry, NativeEntry)}
            retained = tuple(entry for entry in after.entries
                             if isinstance(entry, NativeEntry) and entry.entry_id in old_ids)
            inserted = len(after.entries) - len(retained)
            if (not 1 <= inserted <= reflections
                or (retained and retained != before.entries[-len(retained):])
                or after.entries[:len(retained)] != retained):
                raise SeedResponseError("STRUCTURAL_REFLEXION_RETENTION_VIOLATION")
    elif baseline == "dc_rs":
        before_archive = before.native_state.get("archive")
        after_archive = after.native_state.get("archive")
        if (not isinstance(before_archive, list) or not isinstance(after_archive, list)
            or len(after_archive) != len(before_archive) + 1
            or after_archive[:-1] != before_archive):
            raise SeedResponseError("STRUCTURAL_DC_ARCHIVE_APPEND_VIOLATION")
    else:
        raise SeedResponseError("STRUCTURAL_NATIVE_PROPERTIES_UNAVAILABLE")


def _native_properties_contract() -> str:
    return hashlib.sha256(inspect.getsource(_validate_native_transition).encode()).hexdigest()


def _handler_contract(baseline: str) -> str:
    entry = PHASE13_CORE_BASELINE_REGISTRY[baseline]
    handlers = (entry.initial_state, entry.restore_state, entry.execute_trial,
                entry.serialize_state)
    files = sorted({str(inspect.getsourcefile(handler)) for handler in handlers})
    from memcontam.baselines import (bot_phase12, full_history_phase12,
                                     reflexion_phase12, retrieval_rag_phase12)
    from memcontam.experiment import phase13_dc_rs_runtime
    from memcontam.memory import checkpoint_v3
    modules = (bot_phase12, full_history_phase12, reflexion_phase12,
               retrieval_rag_phase12, phase13_dc_rs_runtime, checkpoint_v3)
    files.extend(str(inspect.getsourcefile(module)) for module in modules)
    return hashlib.sha256(json.dumps(
        [(name, hashlib.sha256(Path(name).read_bytes()).hexdigest())
         for name in sorted(set(files))], separators=(",", ":")).encode()).hexdigest()


def _request_digest(messages: list[dict[str, str]], model: str, config: dict) -> str:
    return hashlib.sha256(json.dumps([messages, model, config], sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, default=str).encode()).hexdigest()


class _RecordingClient:
    def __init__(self, client: LLMClient, arm: str,
                 witness: list[tuple[str, str, str, str, str, int | None]]) -> None:
        self.client = client
        self.arm = arm
        self.witness = witness

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        try:
            response = self.client.chat(messages, model, config)
        except Exception as error:
            raise SeedResponseError("STRUCTURAL_CLIENT_FAILURE_UNAVAILABLE") from error
        self.witness.append((self.arm, _request_digest(messages, model, config),
            response.content, json.dumps(response.raw, sort_keys=True, default=str),
            json.dumps(response.token_usage, sort_keys=True), response.latency_ms))
        return response


class _ReplayClient:
    def __init__(self, arm: str,
                 witness: tuple[tuple[str, str, str, str, str, int | None], ...]) -> None:
        self.arm = arm
        self.witness = witness
        self.position = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        if self.position >= len(self.witness):
            raise SeedResponseError("STRUCTURAL_REPLAY_EXHAUSTED")
        arm, request, content, raw, usage, latency = self.witness[self.position]
        if arm != self.arm or request != _request_digest(messages, model, config):
            raise SeedResponseError("STRUCTURAL_REPLAY_REQUEST_MISMATCH")
        self.position += 1
        return LLMResponse(content, json.loads(raw), json.loads(usage), latency)


def certify_structural_support(inputs: StructuralSupportInputs) -> StructuralSupportEvidence:
    return _structural_support(inputs, execute=True)


def _structural_support(inputs: StructuralSupportInputs, *, execute: bool,
                         trace: tuple[tuple[str, int, str, str, str, str], ...] = (),
                         failure_reason: str | None = None,
                         replay_witness: tuple[tuple[str, str, str, str, str, int | None], ...] = (),
                         replay: bool = False) -> StructuralSupportEvidence:
    prefix, context, run, identity = inputs.prefix, inputs.context, inputs.run, inputs.identity
    tasks = _ordered_tasks(run)
    if (run.baseline == "nomem" or run.baseline not in CORE_MAIN_REGISTRY.memory_baselines
        or (run.task_name, run.baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells
        or not tasks or context.task.task_name != run.task_name
        or context.task.sample_id != tasks[0].sample_id
        or context.model != run.model or context.identities.run_id != run.run_id
        or prefix.state.baseline != run.baseline or run.branch is None
        or len(tasks) != CORE_MAIN_REGISTRY.H_primary
        or len({task.sample_id for task in tasks}) != CORE_MAIN_REGISTRY.H_primary
        or any(task.task_name != run.task_name for task in tasks)
        or identity.concrete_seed_id != str(identity.trajectory_seed)
        or run.trajectory_seed != identity.trajectory_seed
        or (run.production_identity is not None and run.production_identity != identity)
        or hashlib.sha256(json.dumps([task.sample_id for task in tasks], separators=(",", ":")).encode()).hexdigest()
           != identity.ordered_sample_ids_sha256
        or (run.validated_resources is None and not run.allow_test_client)
        or (run.validated_resources is None and
            inputs.fixture_task_bytes != tuple(task.model_dump_json().encode() for task in tasks))
        or (run.validated_resources is not None and inputs.fixture_task_bytes and
            inputs.fixture_task_bytes != run._task_bytes)
        or (identity.scientific_result and run.validated_resources is None)
        or (run.validated_resources is not None and
            (run.validated_resources.checkpoint_sha256 != identity.checkpoint_registry_sha256
             or run.validated_resources.request_client is None))):
        raise SeedResponseError("STRUCTURAL_INPUT_BINDING_MISMATCH")
    deserialize_checkpoint(prefix)
    if run.validated_resources is not None:
        resources = run.validated_resources
        registry = resources.checkpoint_registry
        if (hashlib.sha256(resources.checkpoint_bytes).hexdigest() != resources.checkpoint_sha256
            or run.task_name not in registry.tasks
            or len(registry.tasks[run.task_name].seeds) <= identity.trajectory_seed):
            raise SeedResponseError("STRUCTURAL_REGISTERED_SUFFIX_MISMATCH")
        seed = registry.tasks[run.task_name].seeds[identity.trajectory_seed]
        if (seed.concrete_seed_id != identity.concrete_seed_id
            or seed.suffix_sample_ids_sha256 != identity.ordered_sample_ids_sha256
            or seed.suffix_sample_ids != tuple(task.sample_id for task in tasks)
            or seed.tau_star != prefix.checkpoint_index):
            raise SeedResponseError("STRUCTURAL_REGISTERED_SUFFIX_MISMATCH")
    validate_current_candidate_registry(inputs.candidates)
    candidate_path = REPOSITORY_ROOT / "data/phase12/registries/candidate_registry_v2.json"
    if inputs.candidates != load_current_candidate_registry(candidate_path):
        raise SeedResponseError("STRUCTURAL_CANDIDATE_REGISTRY_MISMATCH")
    capacity = (run.validated_resources.common_capacity_tokens if run.validated_resources is not None
                else validated_common_capacity_tokens())
    if capacity_contract_error(run.baseline_configs, capacity) is not None:
        raise SeedResponseError("STRUCTURAL_CAPACITY_INVALID")
    if run.task_name in ("mmlu_pro_engineering", "mmlu_pro_physics"):
        registry = load_new_mcq_runtime_registry(REPOSITORY_ROOT)
        branches = build_new_mcq_live_branches(prefix=prefix, context=context,
            task=core_task_name(run.task_name), registry=registry,
            runtime_registry=PHASE13_CORE_BASELINE_REGISTRY)
        intervention_sha = hashlib.sha256((REPOSITORY_ROOT /
            "data/phase13/rag/new_mcq/intervention_registry_v1.json").read_bytes()).hexdigest()
    else:
        renderers = RendererRegistry.governed(
            (REPOSITORY_ROOT / "data/phase13/main/legacy_dc_rs_intervention_registry_v2.json").read_bytes(),
            inputs.candidates, hashlib.sha256(candidate_path.read_bytes()).hexdigest())
        branches = build_live_reduced_main_branches(prefix=prefix, context=context,
            candidate_registry=inputs.candidates, registry=PHASE13_CORE_BASELINE_REGISTRY,
            renderers=renderers)
        intervention_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    arms = ("clean", "correct", "irrelevant", "contam")
    if tuple(CORE_MAIN_REGISTRY.arms) != arms:
        raise SeedResponseError("STRUCTURAL_REGISTERED_ARMS_MISMATCH")
    if (set(branches.arms) != set(arms) or run.branch != branches.arms[run.arm]
        or any(branches.arms[arm].prefix_identity != prefix.identity.checkpoint_id
               for arm in arms)):
        raise SeedResponseError("STRUCTURAL_BRANCH_BINDING_MISMATCH")
    branch_hashes = (
        branches.arms["clean"].checkpoint.canonical_sha256,
        branches.arms["correct"].checkpoint.canonical_sha256,
        branches.arms["irrelevant"].checkpoint.canonical_sha256,
        branches.arms["contam"].checkpoint.canonical_sha256,
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY[run.baseline]
    fixture_contract = hashlib.sha256(json.dumps({
        "task_bytes_sha256": [hashlib.sha256(raw).hexdigest() for raw in (
            inputs.fixture_task_bytes if run.validated_resources is None else run._task_bytes)],
        "capacity": capacity,
        "config": run.baseline_configs,
        "model": run.model,
        "decoding": run.decoding,
        "run_id": run.run_id,
        "arm_checkpoints": branch_hashes,
    }, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    route_rows: list[tuple[str, int, str, str]] = []
    transitions: list[tuple[str, int, str, str, str, str]] = []
    recorded: list[tuple[str, str, str, str, str, int | None]] = []
    initial_hashes: dict[str, str] = {}
    unsupported_reason = None
    stages = _BASELINE_STAGES.get(run.baseline)
    if stages is None or any(stage not in STAGES for stage in stages):
        raise SeedResponseError("STRUCTURAL_RUNTIME_ROUTE_UNAVAILABLE")
    for arm in arms:
        branch_run = replace(run, arm=arm, branch=branches.arms[arm])
        if run.baseline == "rag_frozen":
            branch_run = replace(branch_run, initial_states={
                **run.initial_states, "rag_frozen": branches.arms[arm].state})
        client: _ReplayClient | _RecordingClient | None = None
        # Never dispatch the caller's client, even for provider-free conformance.
        # A production-bound client cannot be used to infer suffix capability.
        if execute and (not run.allow_test_client or run.validated_resources is not None
                        or identity.scientific_result):
            raise SeedResponseError("STRUCTURAL_PROSPECTIVE_FIXTURE_REQUIRED")
        if execute:
            client = (_ReplayClient(arm, tuple(row for row in replay_witness if row[0] == arm))
                      if replay else _RecordingClient(deepcopy(run.client), arm, recorded))
            branch_run = replace(branch_run, client=client)
        state = deepcopy(branches.arms[arm].state)
        arm_start = len(transitions)
        for index, task in enumerate(tasks, start=1):
            try:
                rendered = render_model_visible_task(task)
                slot = _context(branch_run, task, index)
                if index == 1:
                    before = entry.serialize_state(state)
                    if not isinstance(before, NativeState):
                        raise SeedResponseError("STRUCTURAL_RESTORE_ROUNDTRIP_MISMATCH")
                    before_hash = _native_digest(before, branches.arms[arm].checkpoint.checkpoint_index)
                    initial_hashes[arm] = before_hash
                route_hash = hashlib.sha256(json.dumps([rendered, stages, slot.identities.trial_id,
                    slot.identities.order_key, identity.execution_template_id,
                    branches.arms[arm].checkpoint.canonical_sha256],
                    separators=(",", ":")).encode()).hexdigest()
                route_rows.append((arm, index, task.sample_id, route_hash))
            except (KeyError, TypeError, ValueError) as error:
                if isinstance(error, SeedResponseError) and (
                    str(error).startswith("STRUCTURAL_REPLAY_")
                    or str(error).startswith("STRUCTURAL_CLIENT_FAILURE_UNAVAILABLE")
                    or str(error).startswith("STRUCTURAL_TRIAL_FAILURE_UNAVAILABLE")
                ):
                    raise
                raise SeedResponseError("STRUCTURAL_INPUT_OR_STATE_UNAVAILABLE") from error
        if execute:
            checkpoint_index = branches.arms[arm].checkpoint.checkpoint_index
            completed_response_count = 0 if replay else len(recorded)
            def record_trial(trial: object) -> None:
                nonlocal completed_response_count
                from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
                if not isinstance(trial, RuntimeTrialResult):
                    raise SeedResponseError("STRUCTURAL_TRIAL_EVIDENCE_UNAVAILABLE")
                position = len(transitions) - arm_start
                if position >= len(tasks) or trial.state_before is None or trial.state_after is None:
                    raise SeedResponseError("STRUCTURAL_NATIVE_STATE_UNAVAILABLE")
                before_hash = _native_digest(trial.state_before, checkpoint_index)
                after_hash = _native_digest(trial.state_after, checkpoint_index)
                expected_before = (initial_hashes[arm] if position == 0 else transitions[-1][5])
                if before_hash != expected_before:
                    raise SeedResponseError("STRUCTURAL_NATIVE_STATE_CHAIN_MISMATCH")
                if trial.outcome.status == "succeeded":
                    _validate_native_transition(trial.state_before, trial.state_after,
                        tuple(call.stage for call in trial.outcome.method_calls))
                for native, native_hash in ((trial.state_before, before_hash),
                                            (trial.state_after, after_hash)):
                    checkpoint = serialize_checkpoint(native, checkpoint_index=checkpoint_index)
                    restored_native = deserialize_checkpoint(checkpoint)
                    restored = entry.restore_state(restored_native, _context(
                        branch_run, tasks[position], position + 1))
                    restored_serialized = entry.serialize_state(restored)
                    if (_native_digest(restored_native, checkpoint_index) != native_hash
                        or not isinstance(restored_serialized, NativeState)
                        or _native_digest(restored_serialized, checkpoint_index) != native_hash):
                        raise SeedResponseError("STRUCTURAL_NATIVE_ROUNDTRIP_MISMATCH")
                route = route_rows[len(route_rows) - len(tasks) + position]
                if inputs.qualification_observer is not None:
                    inputs.qualification_observer(arm, position + 1, trial)
                transitions.append((*route, before_hash, after_hash))
                completed_response_count = (client.position if isinstance(client, _ReplayClient)
                                            else len(recorded))
            branch_run = replace(branch_run, writer_callbacks=RuntimeWriterCallbacks(on_outcome=record_trial))
            try:
                if run.baseline == "fh_bounded":
                    with patch("memcontam.baselines.full_history_adapter.uuid4",
                               lambda: uuid5(NAMESPACE_URL, f"{fixture_contract}:{arm}:{len(transitions) - arm_start + 1}")):
                        result = execute_prospective_ordinary(branch_run)
                else:
                    result = execute_prospective_ordinary(branch_run)
            except (KeyError, TypeError, ValueError) as error:
                if isinstance(error, SeedResponseError) or isinstance(error, (KeyError, TypeError)):
                    raise SeedResponseError("STRUCTURAL_FAILURE_UNAVAILABLE") from error
                from memcontam.baselines.full_history_phase12 import FullHistoryContractError
                if not (
                    isinstance(error, FullHistoryContractError)
                    and error.code == "IMMEDIATE_INJECTED_ROOT_TRUNCATION"
                    and len(transitions) == arm_start
                ):
                    raise SeedResponseError("STRUCTURAL_FAILURE_UNAVAILABLE") from error
                responses_after = len(recorded) if not replay else client.position if isinstance(client, _ReplayClient) else 0
                if responses_after != completed_response_count:
                    raise SeedResponseError("STRUCTURAL_FAILURE_UNAVAILABLE") from error
                unsupported_reason = f"{arm}:{len(transitions) - arm_start + 1}:{type(error).__name__}:{error}"
            else:
                if len(result.trials) != len(tasks) or any(row.outcome.status == "failed" for row in result.trials):
                    raise SeedResponseError("STRUCTURAL_TRIAL_FAILURE_UNAVAILABLE")
        if replay and isinstance(client, _ReplayClient) and client.position != len(client.witness):
            raise SeedResponseError("STRUCTURAL_REPLAY_UNUSED_RESPONSES")
        if unsupported_reason is not None:
            break
    if not execute:
        transitions = list(trace)
        if (unsupported_reason is None and failure_reason is not None):
            if (len(transitions) >= len(route_rows)
                or not any(failure_reason.startswith(
                    f"{route_rows[len(transitions)][0]}:{route_rows[len(transitions)][1]}:{name}:")
                    for name in ("DcRsRuntimeError", "FullHistoryContractError"))):
                raise SeedResponseError("STRUCTURAL_FAILURE_SLOT_MISMATCH")
            unsupported_reason = failure_reason
        if (len(transitions) > len(route_rows)
            or (unsupported_reason is None and len(transitions) != len(route_rows))
            or (unsupported_reason is not None and (
                len(transitions) >= len(route_rows) or failure_reason != unsupported_reason))
            or any(row[:4] != route or (pos > 0 and row[0] == transitions[pos - 1][0]
                        and row[4] != transitions[pos - 1][5])
                   for pos, (row, route) in enumerate(zip(transitions, route_rows)))
            or any(row[4] != initial_hashes[row[0]]
                   for row in transitions if row[1] == 1)):
            raise SeedResponseError("STRUCTURAL_TRACE_CHAIN_MISMATCH")
    suffix_contract_sha256 = hashlib.sha256(json.dumps(transitions, separators=(",", ":")).encode()).hexdigest()
    request_client = None if run.validated_resources is None else run.validated_resources.request_client
    package_sha256 = None if request_client is None else request_client.dispatcher.binding.package_sha256
    if identity.source_package_manifest_sha256 is None and package_sha256 is None:
        raise SeedResponseError("STRUCTURAL_SOURCE_BINDING_REQUIRED")
    proof = StructuralSupportEvidence(
        task=run.task_name, baseline=run.baseline,
        trajectory_seed=identity.trajectory_seed, concrete_seed_id=identity.concrete_seed_id,
        registered_pair=None, supported=unsupported_reason is None,
        source_package_manifest_sha256=identity.source_package_manifest_sha256,
        evidence_sha256="0" * 64,
        checkpoint_id=prefix.identity.checkpoint_id, checkpoint_sha256=prefix.canonical_sha256,
        checkpoint_index=prefix.checkpoint_index,
        ordered_sample_ids_sha256=identity.ordered_sample_ids_sha256,
        execution_template_id=identity.execution_template_id,
        registration_packet_sha256=identity.registration_packet_sha256,
        package_sha256=package_sha256, branch_checkpoint_sha256=branch_hashes,
        candidate_registry_sha256=intervention_sha, suffix_contract_sha256=suffix_contract_sha256,
        handler_contract_sha256=_handler_contract(run.baseline),
        fixture_contract_sha256=fixture_contract,
        native_properties_sha256=_native_properties_contract(),
        conformance_trace=tuple(transitions),
        replay_witness=tuple(recorded) if execute and not replay else replay_witness,
        unsupported_reason=unsupported_reason)
    evidence_sha256 = hashlib.sha256(json.dumps(
        proof.model_dump(mode="json", exclude={"evidence_sha256"}),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return proof.model_copy(update={"evidence_sha256": evidence_sha256})


def validate_structural_support(proof: StructuralSupportEvidence,
                                 inputs: StructuralSupportInputs | StructuralSupportValidationContext) -> None:
    if isinstance(inputs, StructuralSupportValidationContext):
        validate_qualified_structural_support(proof, inputs)
        return
    if proof != _structural_support(inputs, execute=False, trace=proof.conformance_trace,
                                    failure_reason=proof.unsupported_reason,
                                    replay_witness=proof.replay_witness):
        raise SeedResponseError("STRUCTURAL_CERTIFICATE_MISMATCH")
    if (not proof.replay_witness and (proof.supported or proof.conformance_trace)
        or inputs.identity.scientific_result or inputs.run.validated_resources is not None):
        raise SeedResponseError("STRUCTURAL_FROZEN_REPLAY_UNBOUND")
    reproduced = _structural_support(inputs, execute=True, replay=True,
                                      replay_witness=proof.replay_witness)
    if reproduced != proof:
        raise SeedResponseError("STRUCTURAL_FROZEN_REPLAY_MISMATCH")


def registered_pair_support(left: StructuralSupportEvidence, right: StructuralSupportEvidence,
                             left_inputs: StructuralSupportInputs | StructuralSupportValidationContext,
                             right_inputs: StructuralSupportInputs | StructuralSupportValidationContext) -> bool:
    validate_structural_support(left, left_inputs)
    validate_structural_support(right, right_inputs)
    if (left.baseline != "fh_bounded" or right.baseline not in CORE_MAIN_REGISTRY.memory_baselines
        or right.baseline == left.baseline or (left.task, right.baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells
        or (left.task, left.trajectory_seed, left.concrete_seed_id,
            left.ordered_sample_ids_sha256, left.registration_packet_sha256,
            left.source_package_manifest_sha256, left.package_sha256) !=
           (right.task, right.trajectory_seed, right.concrete_seed_id,
            right.ordered_sample_ids_sha256, right.registration_packet_sha256,
            right.source_package_manifest_sha256, right.package_sha256)):
        raise SeedResponseError("UNREGISTERED_OR_MISMATCHED_LEVEL2_PAIR")
    return left.supported and right.supported


class SeedResponseParent(_Frozen):
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registration_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_package_manifest_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    package_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_template_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    baseline: str = Field(min_length=1)
    arm: str = Field(min_length=1)
    trajectory_seed: int = Field(ge=0, le=9)
    concrete_seed_id: str = Field(min_length=1)
    ordered_sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_id: str | None = None
    checkpoint_index: int | None = None
    structural_support: bool | None = None
    structural_support_evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    structural_support_evidence: StructuralSupportEvidence | None = None
    terminal_failure_code: str | None = None
    terminal_parent_id: str | None = None
    terminal_evidence_sha256: str | None = None

    @model_validator(mode="after")
    def _support_binding(self) -> SeedResponseParent:
        if (self.structural_support is None) != (self.structural_support_evidence_sha256 is None):
            raise SeedResponseError("STRUCTURAL_SUPPORT_BINDING_REQUIRED")
        if self.structural_support_evidence is not None and (
            self.structural_support != self.structural_support_evidence.supported
            or self.structural_support_evidence_sha256 != self.structural_support_evidence.evidence_sha256
            or (self.task, self.baseline, self.trajectory_seed, self.concrete_seed_id,
                self.source_package_manifest_sha256) != (
                self.structural_support_evidence.task, self.structural_support_evidence.baseline,
                self.structural_support_evidence.trajectory_seed,
                self.structural_support_evidence.concrete_seed_id,
                self.structural_support_evidence.source_package_manifest_sha256)
        ):
            raise SeedResponseError("STRUCTURAL_SUPPORT_BINDING_MISMATCH")
        if self.concrete_seed_id != str(self.trajectory_seed):
            raise SeedResponseError("SEED_IDENTITY_MISMATCH")
        if (self.terminal_failure_code is None) != (self.terminal_parent_id is None):
            raise SeedResponseError("TERMINAL_PARENT_BINDING_REQUIRED")
        return self


class ObservedSeedResponse(_Frozen):
    evidence_scope: Literal["production_runtime", "provider_free_conformance"]
    parent: SeedResponseParent
    analysis_window_id: Literal["core_prefix_50"] = "core_prefix_50"
    event_times: tuple[int, ...]
    verified_outcomes: tuple[Literal[0, 1], ...]
    observed_sum: int = Field(ge=0)
    observed_count: int = Field(ge=0, le=50)
    exact_mean: float | None
    exact_mean_fraction: str | None = None
    status: Literal["complete", "terminal_technical_missing", "unattempted", "dependency_blocked"]
    missing_reason: str | None = None


class AttemptedSeed(_Frozen):
    task: str
    baseline: str
    arm: str
    trajectory_seed: int = Field(ge=0, le=9)
    concrete_seed_id: str = Field(min_length=1)
    response: ObservedSeedResponse | None = None
    status: Literal["observed", "unattempted", "dependency_blocked"]
    reason: str | None = None
    structural_support: bool | None = None
    structural_support_evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    structural_support_evidence: StructuralSupportEvidence | None = None


class SeedResponseTable(_Frozen):
    task: str
    baseline: str
    attempted_seed_count: Literal[10] = 10
    rows: tuple[AttemptedSeed, ...]


def assemble_seed_response_table(
    rows: tuple[AttemptedSeed, ...], *,
    support_inputs: dict[int, StructuralSupportInputs | StructuralSupportValidationContext] | None = None,
) -> SeedResponseTable:
    if not rows:
        raise SeedResponseError("EMPTY_SEED_RESPONSE_TABLE")
    task, baseline = rows[0].task, rows[0].baseline
    expected_arms = ("nomem",) if baseline == "nomem" else CORE_MAIN_REGISTRY.arms
    expected = {(seed, arm) for seed in range(10) for arm in expected_arms}
    def invalid_row(row: AttemptedSeed) -> bool:
        response = row.response
        if (row.task != task or row.baseline != baseline
            or row.concrete_seed_id != str(row.trajectory_seed)
            or (response is None) != (row.status != "observed")
            or (row.status != "observed" and not row.reason)
            or (row.structural_support is None) != (row.structural_support_evidence_sha256 is None)):
            return True
        if row.status == "observed" and (response is None or row.reason is not None
            or response.parent.task != task or response.parent.baseline != baseline
            or response.parent.arm != row.arm
            or response.parent.trajectory_seed != row.trajectory_seed
            or response.parent.concrete_seed_id != row.concrete_seed_id):
            return True
        if response is not None and (row.structural_support != response.parent.structural_support
            or row.structural_support_evidence_sha256 != response.parent.structural_support_evidence_sha256
            or row.structural_support_evidence != response.parent.structural_support_evidence):
            return True
        if row.structural_support is not None:
            proof = row.structural_support_evidence
            if (proof is None or proof.supported != row.structural_support
                or proof.evidence_sha256 != row.structural_support_evidence_sha256
                or (row.task, row.baseline, row.trajectory_seed, row.concrete_seed_id) != (
                    proof.task, proof.baseline, proof.trajectory_seed, proof.concrete_seed_id)):
                return True
            inputs = None if support_inputs is None else support_inputs.get(row.trajectory_seed)
            if inputs is None:
                return True
            try:
                validate_structural_support(proof, inputs)
            except (SeedResponseError, ValueError):
                return True
        return False

    if ((task, baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells
        or len(rows) != len(expected)
        or {(row.trajectory_seed, row.arm) for row in rows} != expected
        or any(invalid_row(row) for row in rows)):
        raise SeedResponseError("ATTEMPTED_SEED_BLOCK_MISMATCH")
    return SeedResponseTable(task=task, baseline=baseline,
                             rows=tuple(sorted(rows, key=lambda row: (row.trajectory_seed, row.arm))))


def archive_digest(archive: ProductionObservabilityArchive) -> str:
    return hashlib.sha256(archive.model_dump_json().encode()).hexdigest()


def producer_joined_parent(
    run: ProspectiveOrdinaryRun,
    result: ProspectiveOrdinaryResult,
    identity: ProductionOrdinaryRunIdentity,
    archive: ProductionObservabilityArchive,
    *,
    structural_support_evidence: StructuralSupportEvidence | None = None,
    structural_support_inputs: StructuralSupportInputs | StructuralSupportValidationContext | None = None,
) -> SeedResponseParent:
    if structural_support_evidence is not None:
        if (structural_support_inputs is None or structural_support_inputs.identity != identity
            or (isinstance(structural_support_inputs, StructuralSupportInputs)
                and structural_support_inputs.run is not run)):
            raise SeedResponseError("STRUCTURAL_INPUTS_REQUIRED")
        validate_structural_support(structural_support_evidence, structural_support_inputs)
    if identity.scientific_result and (run.allow_test_client or run.validated_resources is None):
        raise SeedResponseError("CONFORMANCE_CANNOT_BE_SCIENTIFIC_RESULT")
    from memcontam.readiness.phase13_production_runtime_join import production_archive_from_completed_prefix

    builder = production_archive_from_completed_prefix if result.terminal_failure is not None and (
        not result.trials or result.trials[-1].outcome.status == "succeeded"
    ) else production_archive_from_ordinary
    if archive != builder(run, result, identity):
        raise SeedResponseError("PRODUCER_ARCHIVE_JOIN_MISMATCH")
    request_client = (None if run.validated_resources is None else run.validated_resources.request_client)
    package_sha256 = (None if request_client is None else request_client.dispatcher.binding.package_sha256)
    if structural_support_evidence is not None and (
        structural_support_evidence.package_sha256 != package_sha256
        or structural_support_evidence.registration_packet_sha256 != identity.registration_packet_sha256
        or structural_support_evidence.execution_template_id != identity.execution_template_id
        or (run.branch is not None and (
             structural_support_evidence.checkpoint_id != run.branch.prefix_identity
             or structural_support_evidence.checkpoint_index != run.branch.checkpoint.checkpoint_index
             or run.arm not in ("clean", "correct", "irrelevant", "contam")
             or structural_support_evidence.branch_checkpoint_sha256[
                 ("clean", "correct", "irrelevant", "contam").index(run.arm)] != run.branch.checkpoint.canonical_sha256))):
        raise SeedResponseError("STRUCTURAL_PRODUCER_BINDING_MISMATCH")
    if (isinstance(structural_support_inputs, StructuralSupportValidationContext)
        and structural_support_inputs.production_origin is not None):
        subject = json.loads(structural_support_inputs.production_origin.subject_bytes)
        if tuple(subject["ordered_task_bytes"]) != tuple(task.model_dump_json() for task in _ordered_tasks(run)):
            raise SeedResponseError("STRUCTURAL_PRODUCER_BINDING_MISMATCH")
    if (run.production_identity is not None and run.production_identity != identity):
        raise SeedResponseError("PRODUCER_IDENTITY_MISMATCH")
    if identity.source_package_manifest_sha256 is None and package_sha256 is None:
        raise SeedResponseError("SOURCE_PACKAGE_BINDING_REQUIRED")
    checkpoint = run.branch.checkpoint if run.branch is not None else None
    return SeedResponseParent(
        archive_sha256=archive_digest(archive),
        registration_packet_sha256=identity.registration_packet_sha256,
        source_package_manifest_sha256=identity.source_package_manifest_sha256,
        package_sha256=package_sha256,
        execution_template_id=identity.execution_template_id,
        run_id=run.run_id,
        task=run.task_name,
        baseline=run.baseline,
        arm="nomem" if run.baseline == "nomem" else run.arm,
        trajectory_seed=identity.trajectory_seed,
        concrete_seed_id=identity.concrete_seed_id,
        ordered_sample_ids_sha256=identity.ordered_sample_ids_sha256,
        checkpoint_id=None if checkpoint is None else checkpoint.identity.checkpoint_id,
        checkpoint_index=None if checkpoint is None else checkpoint.checkpoint_index,
        structural_support=None if structural_support_evidence is None else structural_support_evidence.supported,
        structural_support_evidence_sha256=(None if structural_support_evidence is None
                                            else structural_support_evidence.evidence_sha256),
        structural_support_evidence=structural_support_evidence,
        terminal_failure_code=None if result.terminal_failure is None else result.terminal_failure.code,
        terminal_parent_id=None if result.terminal_failure is None else result.terminal_failure.parent_id,
        terminal_evidence_sha256=None if result.terminal_failure is None else result.terminal_failure.evidence_sha256,
    )


def extract_seed_response(
    archive: ProductionObservabilityArchive,
    packet: ObservabilityRegistrationPacket,
    packet_sha256: str,
    frozen_tasks: tuple[TaskInstance, ...],
    parent: SeedResponseParent,
    *,
    run: ProspectiveOrdinaryRun,
    result: ProspectiveOrdinaryResult,
    identity: ProductionOrdinaryRunIdentity,
    terminal_ledger: TerminalLedgerV3 | None = None,
    terminal_key: RequestKeyV3 | None = None,
    structural_support_inputs: StructuralSupportInputs | StructuralSupportValidationContext | None = None,
    production_unit: ProductionObject | None = None,
    cost_unit: CostUnit | None = None,
) -> ObservedSeedResponse:
    if parent.structural_support is not None and parent.structural_support_evidence is None:
        raise SeedResponseError("REGISTERED_STRUCTURAL_SUPPORT_PROOF_REQUIRED")
    if parent != producer_joined_parent(run, result, identity, archive,
                                        structural_support_evidence=parent.structural_support_evidence,
                                        structural_support_inputs=structural_support_inputs):
        raise SeedResponseError("PRODUCER_PARENT_JOIN_MISMATCH")
    if result.terminal_failure is not None:
        failure = result.terminal_failure
        if (terminal_ledger is None or terminal_key is None or failure.evidence_sha256 is None
            or production_unit is None or cost_unit is None
            or run.validated_resources is None
            or run.validated_resources.selected is None
            or cost_unit not in run.validated_resources.costs.resources.phase4.base.units
            or production_unit.execution_template_id != identity.execution_template_id
            or production_unit.registration_packet_sha256 != identity.registration_packet_sha256
            or production_unit.ordered_sample_ids_sha256 != identity.ordered_sample_ids_sha256
            or production_unit.checkpoint_registry_sha256 != identity.checkpoint_registry_sha256
            or production_unit.unit_id != failure.parent_id
            or cost_unit.unit_id != production_unit.unit_id
            or (production_unit.task, production_unit.seed, production_unit.memory_baseline,
                production_unit.arm) != (parent.task, parent.trajectory_seed,
                None if parent.baseline == "nomem" else parent.baseline,
                "NOT_APPLICABLE" if parent.baseline == "nomem" else parent.arm)
            or terminal_key.parent_id != failure.parent_id
            or (parent.package_sha256 is not None
                and parent.package_sha256 != terminal_ledger.binding.package_sha256)
            or not _completed_prefix_matches(archive, terminal_ledger, terminal_key, parent,
                                              production_unit, cost_unit,
                                              selected=run.validated_resources.selected,
                                             policy=(None if run.validated_resources is None else
                                                     run.validated_resources.costs.resources.phase4.policy))
            or not _terminal_slot_matches(archive, result, terminal_ledger,
                                          terminal_key, parent, frozen_tasks,
                                          production_unit, cost_unit)
            or not _terminal_event_matches(terminal_ledger, failure.evidence_sha256,
                                           terminal_key, failure.code,
                                           semantic_failure_code=(archive.records[-1].terminal_failure_code
                                               if archive.records and archive.records[-1].evidence.trial.execution_status == "failed"
                                               else None))):
            raise SeedResponseError("TERMINAL_LEDGER_EVIDENCE_REQUIRED")
    if (archive_digest(archive) != parent.archive_sha256
        or archive.registration_packet_sha256 != parent.registration_packet_sha256
        or packet_sha256 != parent.registration_packet_sha256):
        raise SeedResponseError("PARENT_ARCHIVE_BINDING_MISMATCH")
    if (parent.task, parent.baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells:
        raise SeedResponseError("EXCLUDED_CURRENT_MAIN_CELL")
    if packet.authority_hashes != AUTHORITY_HASHES:
        raise SeedResponseError("REGISTRATION_AUTHORITY_MISMATCH")
    if (len(frozen_tasks) != 50 or len({task.sample_id for task in frozen_tasks}) != 50
        or any(task.task_name != parent.task for task in frozen_tasks)
        or hashlib.sha256(json.dumps([task.sample_id for task in frozen_tasks], separators=(",", ":")).encode()).hexdigest()
        != parent.ordered_sample_ids_sha256):
        raise SeedResponseError("FROZEN_TASK_ORDER_MISMATCH")
    if archive.records:
        validate_production_archive(archive, packet, packet_sha256, frozen_tasks=frozen_tasks)
    elif result.terminal_failure is None:
        raise SeedResponseError("EMPTY_ARCHIVE_WITHOUT_TERMINAL")
    if len(archive.records) > 50:
        raise SeedResponseError("SEED_EVENT_COUNT_EXCEEDED")
    observed: list[Literal[0, 1]] = []
    times: list[int] = []
    for index, record in enumerate(archive.records):
        evidence = record.evidence
        trial = evidence.trial
        expected_order = index + 1 + (parent.checkpoint_index or 0)
        if (record.execution_template_id != parent.execution_template_id
            or record.run_id != parent.run_id
            or record.ordered_sample_ids_sha256 != parent.ordered_sample_ids_sha256
            or evidence.evidence_scope != "production_runtime"
            or evidence.task != parent.task or evidence.baseline != parent.baseline
            or evidence.trajectory_seed != parent.trajectory_seed
            or evidence.concrete_seed_id != parent.concrete_seed_id
            or evidence.analysis_window_id != "core_prefix_50"
            or evidence.order_key != expected_order
            or trial.event_time != index
            or evidence.trial_id != (f"{parent.run_id}"
                + (f":{parent.arm}" if parent.arm not in ("clean", "nomem") else "")
                + f":trial:{index + 1}:{frozen_tasks[index].sample_id}")
            or (parent.baseline == "nomem") != (parent.arm == "nomem")
            ):
            raise SeedResponseError("SEED_EVENT_IDENTITY_MISMATCH")
        if parent.baseline != "nomem":
            from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence

            if not isinstance(evidence, Phase13TrialEvidence) or (
                evidence.trial.execution_key.arm != parent.arm
                or evidence.trial.checkpoint_id != parent.checkpoint_id
                or evidence.trial.checkpoint_index != parent.checkpoint_index
                or evidence.target_set.source_package_manifest_sha256 != parent.source_package_manifest_sha256
            ):
                raise SeedResponseError("SEED_EVENT_IDENTITY_MISMATCH")
        if trial.execution_status == "failed":
            if index != len(archive.records) - 1 or evidence.verified_outcome is not None:
                raise SeedResponseError("TECHNICAL_TERMINAL_MISMATCH")
            continue
        outcome = evidence.verified_outcome
        if trial.execution_status != "completed" or outcome is None or outcome not in (0, 1):
            raise SeedResponseError("VERIFIED_EVENT_REQUIRED")
        times.append(index)
        observed.append(outcome)
    failed = bool(archive.records and archive.records[-1].evidence.trial.execution_status == "failed")
    failed = failed or parent.terminal_failure_code is not None
    if failed and len(observed) == 50:
        raise SeedResponseError("TERMINAL_OUTSIDE_PRIMARY_WINDOW")
    if not failed and len(observed) != 50:
        raise SeedResponseError("INCOMPLETE_ARCHIVE_WITHOUT_TERMINAL")
    if times != list(range(len(observed))):
        raise SeedResponseError("NONCONTIGUOUS_OBSERVED_PREFIX")
    return ObservedSeedResponse(
        evidence_scope=("production_runtime" if archive.schema_version.endswith("v2")
                        else "provider_free_conformance"),
        parent=parent, event_times=tuple(times), verified_outcomes=tuple(observed),
        observed_sum=sum(observed), observed_count=len(observed),
        exact_mean=sum(observed) / 50 if len(observed) == 50 else None,
        exact_mean_fraction=str(Fraction(sum(observed), 50)) if len(observed) == 50 else None,
        status="terminal_technical_missing" if failed else "complete",
        missing_reason=parent.terminal_failure_code or ("TERMINAL_TECHNICAL_FAILURE" if failed else None),
    )


def _completed_prefix_matches(archive: ProductionObservabilityArchive,
                              ledger: TerminalLedgerV3, terminal_key: RequestKeyV3,
                              parent: SeedResponseParent, unit: ProductionObject,
                              cost_unit: CostUnit, *, policy: ActivatedPolicyV3 | None,
                              selected: SelectedExecutionV3) -> bool:
    from memcontam.readiness.phase13_main_terminal_partial import (
        TerminalPartialParent, scheduled_prefix_keys, validate_terminal_partial_data,
    )
    from memcontam.readiness.phase13_durable_call_join import (
        validated_parent_context, validate_durable_parent_data,
    )
    from memcontam.readiness.phase13_main_v3_runner import DurableParentRecordV3

    try:
        if policy is None:
            return False
        validated_parent_context(selected, ledger, unit, cost_unit)
        if policy != selected.costs.resources.phase4.policy:
            return False
        raw = ledger.read_record(f"{terminal_key.parent_id}.parent.json")
        if json.loads(raw).get("schema_version") == "phase13_main_parent_record_v3":
            persisted_semantic = DurableParentRecordV3.model_validate_json(raw)
            evidence = persisted_semantic.unit_evidence.evidence
            semantic_archive = evidence.runtime_evidence.production_observability_archive
            if (semantic_archive != archive or not archive.records
                or archive.records[-1].evidence.trial.execution_status != "failed"
                or persisted_semantic.unit_evidence.unit_id != unit.unit_id
                or persisted_semantic.identity != ledger.binding.identity.model_dump(mode="json")
                or persisted_semantic.package_sha256 != ledger.binding.package_sha256
                or persisted_semantic.authorization_sha256 != ledger.binding.authorization_sha256):
                return False
            scheduled_semantic = scheduled_prefix_keys(unit, cost_unit, archive.records)
            from memcontam.readiness.phase13_main_live_evidence import MainMethodCall
            semantic_calls = tuple(call for call in persisted_semantic.unit_evidence.provider_calls
                                   if isinstance(call, MainMethodCall))
            if (scheduled_semantic is None or not scheduled_semantic or not semantic_calls
                or len(semantic_calls) != len(persisted_semantic.unit_evidence.provider_calls)
                or scheduled_semantic[-1] != terminal_key):
                return False
            validate_durable_parent_data(selected, ledger, persisted_semantic, unit)
            return True
        persisted = TerminalPartialParent.model_validate_json(
            raw)
        if (persisted.unit_id != terminal_key.parent_id
            or persisted.unit_id != unit.unit_id
            or persisted.terminal_key != terminal_key
            or persisted.archive.records != archive.records
            or persisted.archive.registration_packet_sha256 != archive.registration_packet_sha256
            or persisted.package_sha256 != ledger.binding.package_sha256
            or persisted.authorization_sha256 != ledger.binding.authorization_sha256
            or persisted.identity != ledger.binding.identity.model_dump(mode="json")
            or any(row.run_id != parent.run_id for row in archive.records)):
            return False
        validate_terminal_partial_data(selected, ledger, persisted, unit)
        return True
    except (ValueError, OSError, KeyError, TypeError, StopIteration):
        return False


def _terminal_event_matches(ledger: TerminalLedgerV3, event_hash: str,
                             key: RequestKeyV3, failure_code: str, *,
                             semantic_failure_code: str | None = None) -> bool:
    from memcontam.readiness.phase13_v3_cost_models import digest
    from memcontam.readiness.phase13_main_request_recovery import RequestIdentityReceiptV3
    from memcontam.readiness.phase13_v3_terminal_models import ProviderFailureV3, AmbiguousAttemptV3

    try:
        receipt = RequestIdentityReceiptV3.model_validate_json(
            ledger.read_record(f"{key.dispatch_id}.identity.json"))
        state = ledger.state(key.dispatch_id)
        matches = tuple(event for raw in ledger.rows() if
                        (event := parse_event(raw)).unit_id == key.dispatch_id
                        and digest(event) == event_hash)
        if (receipt.key != key or key.dispatch_id not in ledger.binding.unit_ids
            or not any(parent.parent_id == key.parent_id for parent in receipt.parents)
            or (receipt.binding.identity, receipt.binding.package_sha256,
                receipt.binding.authorization_sha256) != (
                ledger.binding.identity, ledger.binding.package_sha256,
                ledger.binding.authorization_sha256)
            or len(matches) != 1
            or state.compiled is None or state.compiled.stage != key.stage):
            return False
        event = matches[0]
        if event.compiled != state.compiled:
            return False
        if isinstance(event, ProviderFailureV3):
            return ((failure_code == "MAIN_ATTEMPTED_PROVIDER_FAILURE"
                     or (semantic_failure_code is not None and failure_code == semantic_failure_code
                         and event.failure_code == "MAIN_SEMANTIC_RESULT_UNAVAILABLE"))
                    and state.kind == "ATTEMPTED_PROVIDER_FAILURE"
                    and hashlib.sha256(ledger.read_record(
                        f"{key.dispatch_id}.observation.json")).hexdigest() == event.observation_hash)
        if isinstance(event, AmbiguousAttemptV3):
            return failure_code == "MAIN_AMBIGUOUS_ATTEMPT" and state.kind == "AMBIGUOUS_ATTEMPT"
        return event.kind == "TERMINAL_TECHNICAL_MISSING" and (
            failure_code == "MAIN_INPUT_ENVELOPE_EXCEEDED" and state.kind == "TERMINAL_TECHNICAL_MISSING")
    except (ValueError, OSError, KeyError):
        return False


def _terminal_slot_matches(archive: ProductionObservabilityArchive,
                           result: ProspectiveOrdinaryResult, ledger: TerminalLedgerV3,
                           key: RequestKeyV3,
                           parent: SeedResponseParent,
                           tasks: tuple[TaskInstance, ...], unit: ProductionObject,
                           cost_unit: CostUnit) -> bool:
    from memcontam.readiness.phase13_main_request_recovery import RequestIdentityReceiptV3
    from memcontam.readiness.phase13_durable_call_join import scheduled_prefix_keys

    stages = (("no_memory_generate",) if parent.baseline == "nomem"
              else _BASELINE_STAGES.get(parent.baseline))
    if stages is None or len(result.trials) > len(tasks):
        return False
    semantic = bool(archive.records and archive.records[-1].evidence.trial.execution_status == "failed")
    completed = len(archive.records) - int(semantic)
    if semantic:
        return (len(result.trials) == len(archive.records)
                and completed < len(tasks)
                and result.sample_ids[completed] == tasks[completed].sample_id
                and (scheduled := scheduled_prefix_keys(unit, cost_unit, archive.records)) is not None
                and bool(scheduled) and scheduled[-1] == key)
    if completed != len(result.trials) or completed >= len(tasks):
        return False
    if result.sample_ids[completed] != tasks[completed].sample_id:
        return False
    schedule = tuple(slot for slot, _ in schedule_unit(unit, cost_unit))
    if len(schedule) % len(tasks):
        return False
    slots_per_trial = len(schedule) // len(tasks)
    for scheduled in schedule[completed * slots_per_trial:(completed + 1) * slots_per_trial]:
        if key == scheduled:
            return True
        try:
            receipt = RequestIdentityReceiptV3.model_validate_json(
                ledger.read_record(f"{scheduled.dispatch_id}.identity.json"))
            if receipt.key != scheduled or ledger.state(scheduled.dispatch_id).kind != "COMPLETED":
                return False
        except (ValueError, OSError, KeyError):
            return False
    return False
