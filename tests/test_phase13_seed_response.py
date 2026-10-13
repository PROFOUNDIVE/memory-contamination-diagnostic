import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.contamination.phase12.registry import load_current_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.evaluation.phase13_seed_response import (
    AttemptedSeed,
    ObservedSeedResponse,
    SeedResponseError,
    StructuralSupportInputs,
    archive_digest,
    assemble_seed_response_table,
    certify_structural_support,
    extract_seed_response,
    producer_joined_parent,
    registered_pair_support,
)
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun, execute_prospective_ordinary
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.readiness.phase13_production_observability import ProductionObservabilityArchive
from memcontam.readiness.phase13_production_runtime_join import production_archive_from_ordinary
from memcontam.readiness.phase13_production_runtime_join import production_archive_from_completed_prefix
from memcontam.readiness.phase13_main_request_dispatch import DispatchTechnicalFailureV3
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.game24 import verify_expression


class Answers:
    def __init__(self, early_correct: int):
        self.calls = 0
        self.early_correct = early_correct

    def chat(self, messages, model, config):
        self.calls += 1
        answer = "6/(1-3/4)" if self.calls <= self.early_correct or self.calls == 50 else "1+3+4+6"
        return LLMResponse(f"final: {answer}", {"replay": True, "attempts": 1},
                           {"prompt_tokens": 1, "completion_tokens": 1}, 0)


def _producer(early_correct, monkeypatch):
    from memcontam.experiment import phase13_ordinary_runtime

    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    tasks = tuple(TaskInstance(sample_id=f"synthetic-{i}", task_name="game24",
                               input={"numbers": [1, 3, 4, 6], "target": 24},
                               verifier_spec={"target": 24}) for i in range(50))
    run = ProspectiveOrdinaryRun(task_name="game24", baseline="nomem", run_id="conformance-nomem",
                                model="gpt-5.6-luna", client=Answers(early_correct),
                                allow_test_client=True, verifier=lambda answer, task: verify_expression(answer, task.input["numbers"]),
                                decoding={"temperature": 0.0}, tasks=tasks, trajectory_seed=0)
    result = execute_prospective_ordinary(run)
    raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="game24:nomem", trajectory_seed=0, concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(json.dumps(result.sample_ids, separators=(",", ":")).encode()).hexdigest(),
        registration_packet_sha256=hashlib.sha256(raw).hexdigest(), scientific_result=False,
        source_package_manifest_sha256="a" * 64,
    )
    archive = production_archive_from_ordinary(run, result, identity)
    parent = producer_joined_parent(run, result, identity, archive)
    packet = ObservabilityRegistrationPacket.model_validate_json(raw)
    return archive, parent, packet, identity.registration_packet_sha256, tasks, run, result, identity


def test_real_producer_archive_extracts_all_fifty_and_reopens(monkeypatch, tmp_path):
    archive, parent, packet, digest, tasks, run, result, identity = _producer(2, monkeypatch)
    saved = tmp_path / "archive.json"
    saved.write_text(archive.model_dump_json())
    reopened = ProductionObservabilityArchive.model_validate_json(saved.read_bytes())
    response = extract_seed_response(reopened, packet, digest, tasks, parent,
                                     run=run, result=result, identity=identity)
    assert isinstance(ObservedSeedResponse.model_validate_json(response.model_dump_json()), ObservedSeedResponse)
    assert response.evidence_scope == "provider_free_conformance"
    assert response.event_times == tuple(range(50))
    assert response.observed_sum == 3 and response.observed_count == 50
    assert response.exact_mean == 3 / 50 and response.status == "complete"
    assert response.exact_mean_fraction == "3/50"
    other, other_parent, _, _, _, other_run, other_result, other_identity = _producer(8, monkeypatch)
    assert other.records[-1].evidence.verified_outcome == archive.records[-1].evidence.verified_outcome
    assert extract_seed_response(other, packet, digest, tasks, other_parent,
                                 run=other_run, result=other_result, identity=other_identity).exact_mean == 9 / 50
    with pytest.raises(SeedResponseError, match="PRODUCER_ARCHIVE_JOIN_MISMATCH"):
        extract_seed_response(other, packet, digest, tasks, parent,
                              run=run, result=result, identity=identity)
    with pytest.raises(SeedResponseError, match="PRODUCER_PARENT_JOIN_MISMATCH"):
        extract_seed_response(reopened, packet, digest, tasks,
                              parent.model_copy(update={"source_package_manifest_sha256": "b" * 64}),
                              run=run, result=result, identity=identity)
    with pytest.raises(SeedResponseError, match="REGISTERED_STRUCTURAL_SUPPORT_PROOF_REQUIRED"):
        extract_seed_response(reopened, packet, digest, tasks,
                              parent.model_copy(update={"structural_support": True,
                                                        "structural_support_evidence_sha256": "c" * 64}),
                              run=run, result=result, identity=identity)
    forged = reopened.model_copy(update={"records": reopened.records[:-1] + (
        reopened.records[-1].model_copy(update={"parsed_answer": "1+3+4+6"}),)})
    with pytest.raises(SeedResponseError, match="PRODUCER_ARCHIVE_JOIN_MISMATCH"):
        extract_seed_response(forged, packet, digest, tasks, parent,
                              run=run, result=result, identity=identity)
    with pytest.raises(SeedResponseError, match="PRODUCER_ARCHIVE_JOIN_MISMATCH"):
        extract_seed_response(forged, packet, digest, tasks,
                              parent.model_copy(update={"archive_sha256": archive_digest(forged)}),
                              run=run, result=result, identity=identity)


def test_terminal_prefix_is_not_a_complete_response_and_table_keeps_attempts(monkeypatch):
    archive, parent, packet, digest, tasks, _, _, _ = _producer(2, monkeypatch)
    from memcontam.experiment import phase13_ordinary_runtime
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    run = ProspectiveOrdinaryRun(task_name="game24", baseline="nomem", run_id="conformance-nomem",
                                model="gpt-5.6-luna", client=Answers(2), allow_test_client=True,
                                verifier=lambda answer, task: verify_expression(answer, task.input["numbers"]),
                                decoding={"temperature": 0.0}, tasks=tasks, trajectory_seed=0)
    result = execute_prospective_ordinary(run)
    partial = replace(result, trials=result.trials[:7], terminal_failure=DispatchTechnicalFailureV3(
        "REQUEST_PARENT_TERMINAL", "request-8", None, evidence_sha256="b" * 64))
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id=parent.execution_template_id, trajectory_seed=0, concrete_seed_id="0",
        ordered_sample_ids_sha256=parent.ordered_sample_ids_sha256,
        registration_packet_sha256=digest, source_package_manifest_sha256="a" * 64,
        scientific_result=False)
    prefix_archive = production_archive_from_completed_prefix(run, partial, identity)
    prefix_parent = producer_joined_parent(run, partial, identity, prefix_archive)
    with pytest.raises(SeedResponseError, match="TERMINAL_LEDGER_EVIDENCE_REQUIRED"):
        extract_seed_response(prefix_archive, packet, digest, tasks, prefix_parent,
                              run=run, result=partial, identity=identity)
    with pytest.raises(SeedResponseError, match="PRODUCER_PARENT_JOIN_MISMATCH"):
        extract_seed_response(prefix_archive, packet, digest, tasks,
                              prefix_parent.model_copy(update={"terminal_failure_code": None, "terminal_parent_id": None}),
                              run=run, result=partial, identity=identity)
    rows = tuple(AttemptedSeed(task="game24", baseline="nomem", arm="nomem",
                               trajectory_seed=i, concrete_seed_id=str(i),
                               response=None, status="unattempted",
                               reason="PROSPECTIVE_SEED_NOT_YET_ATTEMPTED")
                 for i in range(10))
    table = assemble_seed_response_table(rows)
    assert len(table.rows) == 10 and table.rows[0].response is None
    with pytest.raises(SeedResponseError, match="ATTEMPTED_SEED_BLOCK_MISMATCH"):
        assemble_seed_response_table(rows[:-1])


def test_native_structural_certificate_rebuilds_branches_and_binds_extraction(monkeypatch, tmp_path):
    from memcontam.experiment import phase13_ordinary_runtime
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    tasks = tuple(TaskInstance(sample_id=f"synthetic-{i}", task_name="game24",
                               input={"numbers": [1, 3, 4, 6], "target": 24},
                               verifier_spec={"target": 24}) for i in range(50))
    state = FullHistoryStateV3(records=[])
    snapshot = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"].serialize_state(state)
    assert isinstance(snapshot, NativeState)
    prefix = serialize_checkpoint(snapshot, checkpoint_index=0)
    client = Answers(2)
    context = Game24RuntimeContext(task=tasks[0], client=client, model="gpt-5.6-luna",
                                   verifier=lambda answer, task: verify_expression(answer, task.input["numbers"]),
                                   decoding={"temperature": 0.0}, branch="clean",
                                   identities=RuntimeIdentities("structural", "structural:trial", 0))
    candidates = load_current_candidate_registry(Path("data/phase12/registries/candidate_registry_v2.json"))
    renderers = RendererRegistry.governed(
        Path("data/phase13/main/legacy_dc_rs_intervention_registry_v2.json").read_bytes(),
        candidates, hashlib.sha256(Path("data/phase12/registries/candidate_registry_v2.json").read_bytes()).hexdigest())
    branches = build_live_reduced_main_branches(prefix=prefix, context=context,
                                                candidate_registry=candidates,
                                                registry=PHASE13_CORE_BASELINE_REGISTRY,
                                                renderers=renderers)
    run = ProspectiveOrdinaryRun(task_name="game24", baseline="fh_bounded", run_id="structural",
                                model="gpt-5.6-luna", client=client, allow_test_client=True,
                                verifier=context.verifier, decoding=context.decoding,
                                tasks=tasks, trajectory_seed=0, arm="clean", branch=branches.arms["clean"])
    packet_raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    packet_sha = hashlib.sha256(packet_raw).hexdigest()
    identity = ProductionOrdinaryRunIdentity(execution_template_id="game24:fh_bounded:clean",
        trajectory_seed=0, concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(json.dumps([t.sample_id for t in tasks],
            separators=(",", ":")).encode()).hexdigest(),
        registration_packet_sha256=packet_sha, source_package_manifest_sha256="a" * 64,
        scientific_result=False)
    inputs = StructuralSupportInputs(prefix, context, candidates, run, identity)
    proof = certify_structural_support(inputs)
    assert proof.supported and len(proof.branch_checkpoint_sha256) == 4
    broken_tasks = (*tasks[:49], tasks[49].model_copy(update={"input": {"target": 24}}))
    broken_run = replace(run, tasks=broken_tasks)
    broken_identity = identity.model_copy(update={"ordered_sample_ids_sha256":
        hashlib.sha256(json.dumps([t.sample_id for t in broken_tasks],
            separators=(",", ":")).encode()).hexdigest()})
    broken_inputs = StructuralSupportInputs(prefix, context, candidates, broken_run, broken_identity)
    broken_proof = certify_structural_support(broken_inputs)
    assert broken_proof.supported is False and broken_proof.unsupported_reason == "clean:50:KeyError"
    assert broken_proof.suffix_contract_sha256 != proof.suffix_contract_sha256
    from memcontam.evaluation.phase13_seed_response import validate_structural_support
    validate_structural_support(broken_proof, broken_inputs)
    false_rows = tuple(AttemptedSeed(task="game24", baseline="fh_bounded", arm=arm,
        trajectory_seed=seed, concrete_seed_id=str(seed), status="unattempted",
        reason="PROSPECTIVE_SEED_NOT_YET_ATTEMPTED",
        structural_support=False if seed == 0 else None,
        structural_support_evidence_sha256=broken_proof.evidence_sha256 if seed == 0 else None,
        structural_support_evidence=broken_proof if seed == 0 else None)
        for seed in range(10) for arm in ("clean", "correct", "irrelevant", "contam"))
    assert assemble_seed_response_table(false_rows, support_inputs={0: broken_inputs}).rows[0].structural_support is False
    with pytest.raises(SeedResponseError, match="ATTEMPTED_SEED_BLOCK_MISMATCH"):
        assemble_seed_response_table(false_rows, support_inputs={0: inputs})
    with pytest.raises(SeedResponseError, match="STRUCTURAL_CERTIFICATE_MISMATCH"):
        validate_structural_support(broken_proof.model_copy(update={"supported": True}), broken_inputs)
    with pytest.raises(SeedResponseError, match="STRUCTURAL_CERTIFICATE_MISMATCH"):
        validate_structural_support(proof.model_copy(update={"checkpoint_sha256": "f" * 64}), inputs)
    with pytest.raises(SeedResponseError, match="UNREGISTERED_OR_MISMATCHED_LEVEL2_PAIR"):
        registered_pair_support(proof, proof, inputs, inputs)
    bot_state = BoTStateV3(entries=[])
    bot_snapshot = PHASE13_CORE_BASELINE_REGISTRY["bot_style"].serialize_state(bot_state)
    assert isinstance(bot_snapshot, NativeState)
    bot_prefix = serialize_checkpoint(bot_snapshot, checkpoint_index=0)
    bot_context = Game24RuntimeContext(task=tasks[0], client=client, model="gpt-5.6-luna",
                                       verifier=context.verifier, decoding=context.decoding, branch="clean",
                                       identities=context.identities)
    bot_branches = build_live_reduced_main_branches(prefix=bot_prefix, context=bot_context,
        candidate_registry=candidates, registry=PHASE13_CORE_BASELINE_REGISTRY, renderers=renderers)
    bot_run = replace(run, baseline="bot_style", branch=bot_branches.arms["clean"])
    bot_identity = identity.model_copy(update={"execution_template_id": "game24:bot_style:clean"})
    bot_inputs = StructuralSupportInputs(bot_prefix, bot_context, candidates, bot_run, bot_identity)
    bot_proof = certify_structural_support(bot_inputs)
    assert registered_pair_support(proof, bot_proof, inputs, bot_inputs)
    with pytest.raises(SeedResponseError, match="UNREGISTERED_OR_MISMATCHED_LEVEL2_PAIR"):
        registered_pair_support(bot_proof, proof, bot_inputs, inputs)
    result = execute_prospective_ordinary(run)
    archive = production_archive_from_ordinary(run, result, identity)
    path = tmp_path / "archive.json"
    path.write_text(archive.model_dump_json())
    reopened = ProductionObservabilityArchive.model_validate_json(path.read_bytes())
    parent = producer_joined_parent(run, result, identity, reopened,
                                    structural_support_evidence=proof, structural_support_inputs=inputs)
    packet = ObservabilityRegistrationPacket.model_validate_json(packet_raw)
    response = extract_seed_response(reopened, packet, packet_sha, tasks, parent,
                                     run=run, result=result, identity=identity,
                                     structural_support_inputs=inputs)
    assert response.observed_count == 50 and response.parent.structural_support is True
    with pytest.raises(SeedResponseError, match="STRUCTURAL_INPUTS_REQUIRED"):
        extract_seed_response(reopened, packet, packet_sha, tasks, parent,
                              run=run, result=result, identity=identity)
    rows = tuple(AttemptedSeed(task="game24", baseline="fh_bounded", arm=arm,
        trajectory_seed=seed, concrete_seed_id=str(seed),
        status="observed" if (seed, arm) == (0, "clean") else "unattempted",
        response=response if (seed, arm) == (0, "clean") else None,
        reason=None if (seed, arm) == (0, "clean") else "PROSPECTIVE_SEED_NOT_YET_ATTEMPTED",
        structural_support=True if seed == 0 else None,
        structural_support_evidence_sha256=proof.evidence_sha256 if seed == 0 else None,
        structural_support_evidence=proof if seed == 0 else None)
        for seed in range(10) for arm in ("clean", "correct", "irrelevant", "contam"))
    assert len(assemble_seed_response_table(rows, support_inputs={0: inputs}).rows) == 40
    with pytest.raises(SeedResponseError, match="ATTEMPTED_SEED_BLOCK_MISMATCH"):
        assemble_seed_response_table(rows)
