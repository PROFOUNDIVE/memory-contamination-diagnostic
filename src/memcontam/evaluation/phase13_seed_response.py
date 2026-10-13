from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.evaluation.phase13_observability_registration import AUTHORITY_HASHES, ObservabilityRegistrationPacket
from memcontam.experiment.phase13_ordinary_runtime import REPOSITORY_ROOT, ProspectiveOrdinaryResult, ProspectiveOrdinaryRun, _context, _ordered_tasks
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.contamination.phase12.models import CandidateRegistry
from memcontam.contamination.phase12.registry import load_current_candidate_registry, validate_current_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.memory.checkpoint_v3 import NativeState, Phase12Checkpoint, deserialize_checkpoint, serialize_checkpoint
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
from memcontam.readiness.phase13_main_live_evidence import _BASELINE_STAGES
from memcontam.tasks.dispatch import render_model_visible_task
from memcontam.tasks.base import TaskInstance


class SeedResponseError(ValueError):
    pass


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
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class StructuralSupportInputs:
    prefix: Phase12Checkpoint
    context: Game24RuntimeContext
    candidates: CandidateRegistry
    run: ProspectiveOrdinaryRun
    identity: ProductionOrdinaryRunIdentity


def certify_structural_support(inputs: StructuralSupportInputs) -> StructuralSupportEvidence:
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
    arms = CORE_MAIN_REGISTRY.arms
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
    route_rows: list[tuple[str, int, str, str]] = []
    unsupported_reason = None
    stages = _BASELINE_STAGES.get(run.baseline)
    if stages is None or any(stage not in STAGES for stage in stages):
        raise SeedResponseError("STRUCTURAL_RUNTIME_ROUTE_UNAVAILABLE")
    for arm in arms:
        branch_run = replace(run, arm=arm, branch=branches.arms[arm])
        for index, task in enumerate(tasks, start=1):
            try:
                rendered = render_model_visible_task(task)
                slot = _context(branch_run, task, index)
                state = entry.restore_state(branches.arms[arm].checkpoint.state, slot)
                snapshot = entry.serialize_state(state)
                if not isinstance(snapshot, NativeState):
                    raise SeedResponseError("STRUCTURAL_RESTORE_ROUNDTRIP_MISMATCH")
                serialize_checkpoint(snapshot, checkpoint_index=branches.arms[arm].checkpoint.checkpoint_index)
            except (KeyError, TypeError, ValueError) as error:
                unsupported_reason = f"{arm}:{index}:{type(error).__name__}"
                break
            route_rows.append((arm, index, task.sample_id,
                hashlib.sha256(json.dumps([rendered, stages, slot.identities.trial_id,
                    slot.identities.order_key, identity.execution_template_id,
                    branches.arms[arm].checkpoint.canonical_sha256],
                    separators=(",", ":")).encode()).hexdigest()))
        if unsupported_reason is not None:
            break
    suffix_contract_sha256 = hashlib.sha256(json.dumps(route_rows, separators=(",", ":")).encode()).hexdigest()
    request_client = None if run.validated_resources is None else run.validated_resources.request_client
    package_sha256 = None if request_client is None else request_client.dispatcher.binding.package_sha256
    payload = dict(task=run.task_name, baseline=run.baseline,
        trajectory_seed=identity.trajectory_seed, concrete_seed_id=identity.concrete_seed_id,
        registered_pair=None, supported=unsupported_reason is None,
        source_package_manifest_sha256=identity.source_package_manifest_sha256,
        checkpoint_id=prefix.identity.checkpoint_id, checkpoint_sha256=prefix.canonical_sha256,
        checkpoint_index=prefix.checkpoint_index,
        ordered_sample_ids_sha256=identity.ordered_sample_ids_sha256,
        execution_template_id=identity.execution_template_id,
        registration_packet_sha256=identity.registration_packet_sha256,
        package_sha256=package_sha256, branch_checkpoint_sha256=branch_hashes,
        candidate_registry_sha256=intervention_sha, suffix_contract_sha256=suffix_contract_sha256,
        unsupported_reason=unsupported_reason)
    if payload["source_package_manifest_sha256"] is None and package_sha256 is None:
        raise SeedResponseError("STRUCTURAL_SOURCE_BINDING_REQUIRED")
    evidence_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return StructuralSupportEvidence(**payload, evidence_sha256=evidence_sha256)


def validate_structural_support(proof: StructuralSupportEvidence,
                                inputs: StructuralSupportInputs) -> None:
    if proof != certify_structural_support(inputs):
        raise SeedResponseError("STRUCTURAL_CERTIFICATE_MISMATCH")


def registered_pair_support(left: StructuralSupportEvidence, right: StructuralSupportEvidence,
                            left_inputs: StructuralSupportInputs,
                            right_inputs: StructuralSupportInputs) -> bool:
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
    support_inputs: dict[int, StructuralSupportInputs] | None = None,
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
    structural_support_inputs: StructuralSupportInputs | None = None,
) -> SeedResponseParent:
    if structural_support_evidence is not None:
        if structural_support_inputs is None or structural_support_inputs.run is not run or structural_support_inputs.identity != identity:
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
            or structural_support_evidence.checkpoint_index != run.branch.checkpoint.checkpoint_index))):
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
    structural_support_inputs: StructuralSupportInputs | None = None,
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
            or terminal_key.parent_id != failure.parent_id
            or (parent.package_sha256 is not None
                and parent.package_sha256 != terminal_ledger.binding.package_sha256)
            or not _terminal_slot_matches(archive, result, terminal_ledger,
                                          terminal_key, parent, frozen_tasks)
            or not _terminal_event_matches(terminal_ledger, failure.evidence_sha256,
                                           terminal_key, failure.code)):
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


def _terminal_event_matches(ledger: TerminalLedgerV3, event_hash: str,
                            key: RequestKeyV3, failure_code: str) -> bool:
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
            return (failure_code == "MAIN_ATTEMPTED_PROVIDER_FAILURE"
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
                           tasks: tuple[TaskInstance, ...]) -> bool:
    from memcontam.readiness.phase13_main_request_recovery import RequestIdentityReceiptV3

    stages = (("no_memory_generate",) if parent.baseline == "nomem"
              else _BASELINE_STAGES.get(parent.baseline))
    if stages is None or len(result.trials) >= len(tasks):
        return False
    completed = len(archive.records)
    if completed != len(result.trials) or completed >= len(tasks):
        return False
    stride = 2 if parent.baseline == "reflexion_style" else 1
    if result.sample_ids[completed] != tasks[completed].sample_id:
        return False
    if any(tuple(call.stage for call in record.method_calls) != stages
           for record in archive.records):
        return False
    for position, stage in enumerate(stages):
        scheduled = RequestKeyV3(parent_id=key.parent_id, stage=stage,
            ordinal=completed * stride + (position if stride == 2 else 0))
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
