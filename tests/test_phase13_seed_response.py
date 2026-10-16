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
from .test_phase13_native_production_archive import NativeResponses, NoveltyEmbedder


class Answers:
    def __init__(self, early_correct: int):
        self.calls = 0
        self.early_correct = early_correct

    def chat(self, messages, model, config):
        self.calls += 1
        answer = "6/(1-3/4)" if self.calls <= self.early_correct or self.calls == 50 else "1+3+4+6"
        return LLMResponse(f"final: {answer}", {"replay": True, "attempts": 1},
                           {"prompt_tokens": 1, "completion_tokens": 1}, 0)


class LateFailureAnswers(Answers):
    def chat(self, messages, model, config):
        if self.calls == 3:
            raise ValueError("synthetic later resource failure")
        return super().chat(messages, model, config)


class ReachableBotResponses(NativeResponses):
    def chat(self, messages, model, config):
        response = super().chat(messages, model, config)
        if config["method_stage"] != "bot_instantiate_solve":
            return response
        payload = json.loads(response.content)
        payload["selected_structure"] = (
            "retrieved-template" if "Set selected_structure to retrieved-template" in messages[-1]["content"]
            else "procedure-based"
        )
        return LLMResponse(json.dumps(payload), response.raw, response.token_usage, response.latency_ms)


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
    fixture_bytes = tuple(task.model_dump_json().encode() for task in tasks)
    inputs = StructuralSupportInputs(prefix, context, candidates, run, identity, fixture_bytes)
    proof = certify_structural_support(inputs)
    assert proof.supported and len(proof.branch_checkpoint_sha256) == 4, proof.unsupported_reason
    assert len(proof.conformance_trace) == 200
    assert all(proof.conformance_trace[i][5] == proof.conformance_trace[i + 1][4]
               for i in range(49))
    assert client.calls == 0
    from memcontam.evaluation.phase13_seed_response import validate_structural_support
    validate_structural_support(proof, inputs)
    assert client.calls == 0
    from memcontam.evaluation.phase13_seed_response import (
        StructuralSupportValidationContext, validate_qualified_structural_support,
    )
    qualified_bytes = proof.model_dump_json().encode()
    expected_branches = tuple(branches.arms[arm].checkpoint.canonical_sha256
                              for arm in ("clean", "correct", "irrelevant", "contam"))
    expected_fixture = hashlib.sha256(json.dumps({
        "task_bytes_sha256": [hashlib.sha256(raw).hexdigest() for raw in fixture_bytes],
        "capacity": 8192, "config": run.baseline_configs, "model": "gpt-5.6-luna",
        "decoding": {"temperature": 0.0}, "run_id": "structural",
        "arm_checkpoints": expected_branches,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    validation_context = StructuralSupportValidationContext(
        qualified_evidence_bytes=qualified_bytes,
        qualified_evidence_sha256=hashlib.sha256(qualified_bytes).hexdigest(),
        identity=identity, package_sha256=None, prefix=prefix,
        fixture_contract_sha256=expected_fixture,
        branch_checkpoint_sha256=(expected_branches[0], expected_branches[1],
                                  expected_branches[2], expected_branches[3]),
        candidate_registry_sha256=hashlib.sha256(
            Path("data/phase12/registries/candidate_registry_v2.json").read_bytes()).hexdigest())
    class ForbiddenClient:
        def chat(self, *_args, **_kwargs):
            pytest.fail("data-only validation called the execution client")

        def __deepcopy__(self, _memo):
            pytest.fail("data-only validation copied the execution client")

    with monkeypatch.context() as no_execution:
        no_execution.setattr("memcontam.evaluation.phase13_seed_response.execute_prospective_ordinary",
                             lambda *_args: pytest.fail("data-only validation executed qualification"))
        no_execution.setattr(client, "chat", ForbiddenClient().chat)
        no_execution.setattr(Answers, "__deepcopy__", ForbiddenClient.__deepcopy__, raising=False)
        validate_structural_support(proof, validation_context)
        with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFICATION_ORIGIN_REQUIRED"):
            validate_qualified_structural_support(proof, replace(validation_context,
                identity=identity.model_copy(update={"scientific_result": True})))
        package_forgery = proof.model_copy(update={"package_sha256": "f" * 64})
        package_forgery = package_forgery.model_copy(update={
            "evidence_sha256": hashlib.sha256(json.dumps(
                package_forgery.model_dump(mode="json", exclude={"evidence_sha256"}),
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
        package_bytes = package_forgery.model_dump_json().encode()
        with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFICATION_ORIGIN_REQUIRED"):
            validate_qualified_structural_support(package_forgery, replace(validation_context,
                package_sha256="f" * 64, qualified_evidence_bytes=package_bytes,
                qualified_evidence_sha256=hashlib.sha256(package_bytes).hexdigest()))
        for field in ("fixture_contract_sha256", "candidate_registry_sha256", "package_sha256"):
            with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFIED_CONTEXT_MISMATCH"):
                validate_qualified_structural_support(proof, replace(validation_context, **{field: "f" * 64}))
        with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFICATION_BINDING_MISMATCH"):
            validate_qualified_structural_support(proof, replace(validation_context,
                qualified_evidence_bytes=qualified_bytes + b" "))
        with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFIED_CONTEXT_MISMATCH"):
            validate_qualified_structural_support(proof.model_copy(update={"supported": False}), validation_context)
        for changes in ({"supported": False}, {"conformance_trace": ()},
                        {"suffix_contract_sha256": "f" * 64}):
            forged_qualification = proof.model_copy(update=changes)
            forged_qualification = forged_qualification.model_copy(update={
                "evidence_sha256": hashlib.sha256(json.dumps(
                    forged_qualification.model_dump(mode="json", exclude={"evidence_sha256"}),
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
            forged_bytes = forged_qualification.model_dump_json().encode()
            with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFIED_TRACE_MISMATCH"):
                validate_qualified_structural_support(forged_qualification, replace(validation_context,
                    qualified_evidence_bytes=forged_bytes,
                    qualified_evidence_sha256=hashlib.sha256(forged_bytes).hexdigest()))
    with pytest.raises(SeedResponseError, match="STRUCTURAL_INPUT_BINDING_MISMATCH"):
        certify_structural_support(replace(inputs, fixture_task_bytes=()))
    altered_tasks = (tasks[0].model_copy(update={"input": {"numbers": [2, 3, 4, 6], "target": 24}}), *tasks[1:])
    with pytest.raises(SeedResponseError, match="STRUCTURAL_INPUT_BINDING_MISMATCH"):
        certify_structural_support(replace(inputs, run=replace(run, tasks=altered_tasks)))
    broken_trace = proof.model_copy(update={"conformance_trace":
        proof.conformance_trace[:1] + (proof.conformance_trace[1][:4] +
        ("0" * 64, proof.conformance_trace[1][5]),) + proof.conformance_trace[2:]})
    with pytest.raises(SeedResponseError, match="STRUCTURAL_TRACE_CHAIN_MISMATCH"):
        validate_structural_support(broken_trace, inputs)
    forged_rows = list(proof.conformance_trace)
    changed = "f" * 64
    forged_rows[0] = (*forged_rows[0][:5], changed)
    forged_rows[1] = (*forged_rows[1][:4], changed, forged_rows[1][5])
    forged = proof.model_copy(update={"conformance_trace": tuple(forged_rows),
        "suffix_contract_sha256": hashlib.sha256(json.dumps(forged_rows,
            separators=(",", ":")).encode()).hexdigest()})
    forged = forged.model_copy(update={"evidence_sha256": hashlib.sha256(json.dumps(
        forged.model_dump(mode="json", exclude={"evidence_sha256"}),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    with pytest.raises(SeedResponseError, match="STRUCTURAL_FROZEN_REPLAY_MISMATCH"):
        validate_structural_support(forged, inputs)
    unbound = proof.model_copy(update={"replay_witness": ()})
    unbound = unbound.model_copy(update={"evidence_sha256": hashlib.sha256(json.dumps(
        unbound.model_dump(mode="json", exclude={"evidence_sha256"}),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    with pytest.raises(SeedResponseError, match="STRUCTURAL_FROZEN_REPLAY_UNBOUND"):
        validate_structural_support(unbound, inputs)
    assert client.calls == 0
    late_tasks = (*tasks[:3], tasks[3].model_copy(update={"input": {"target": 24}}), *tasks[4:])
    late_run = replace(run, tasks=late_tasks)
    with pytest.raises(SeedResponseError, match="STRUCTURAL_INPUT_OR_STATE_UNAVAILABLE"):
        certify_structural_support(replace(inputs, run=late_run,
            fixture_task_bytes=tuple(task.model_dump_json().encode() for task in late_tasks)))
    with pytest.raises(SeedResponseError, match="STRUCTURAL_(CLIENT|TRIAL)_FAILURE_UNAVAILABLE"):
        certify_structural_support(replace(inputs, run=replace(run, client=LateFailureAnswers(2))))
    fake_failure = proof.model_copy(update={
        "supported": False,
        "unsupported_reason": "clean:4:FullHistoryContractError:FALSE_EXACT_LABEL",
        "conformance_trace": proof.conformance_trace[:3],
        "suffix_contract_sha256": hashlib.sha256(json.dumps(
            proof.conformance_trace[:3], separators=(",", ":")).encode()).hexdigest(),
        "replay_witness": proof.replay_witness[:3],
    })
    fake_failure = fake_failure.model_copy(update={"evidence_sha256": hashlib.sha256(json.dumps(
        fake_failure.model_dump(mode="json", exclude={"evidence_sha256"}),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    with pytest.raises(SeedResponseError, match="STRUCTURAL_(FAILURE|TRIAL_FAILURE)_UNAVAILABLE"):
        validate_structural_support(fake_failure, inputs)
    completed_failure = proof.model_copy(update={
        "supported": False,
        "unsupported_reason": "contam:50:FullHistoryContractError:FALSE_EXACT_LABEL",
    })
    completed_failure = completed_failure.model_copy(update={"evidence_sha256": hashlib.sha256(json.dumps(
        completed_failure.model_dump(mode="json", exclude={"evidence_sha256"}),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    with pytest.raises(SeedResponseError, match="STRUCTURAL_FAILURE_SLOT_MISMATCH"):
        validate_structural_support(completed_failure, inputs)
    missing_witness = fake_failure.model_copy(update={"replay_witness": ()})
    missing_witness = missing_witness.model_copy(update={"evidence_sha256": hashlib.sha256(json.dumps(
        missing_witness.model_dump(mode="json", exclude={"evidence_sha256"}),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    with pytest.raises(SeedResponseError, match="STRUCTURAL_FROZEN_REPLAY_UNBOUND"):
        validate_structural_support(missing_witness, inputs)
    broken_tasks = (*tasks[:49], tasks[49].model_copy(update={"input": {"target": 24}}))
    broken_run = replace(run, tasks=broken_tasks)
    broken_identity = identity.model_copy(update={"ordered_sample_ids_sha256":
        hashlib.sha256(json.dumps([t.sample_id for t in broken_tasks],
            separators=(",", ":")).encode()).hexdigest()})
    broken_inputs = StructuralSupportInputs(prefix, context, candidates, broken_run, broken_identity,
        tuple(task.model_dump_json().encode() for task in broken_tasks))
    with pytest.raises(SeedResponseError, match="STRUCTURAL_INPUT_OR_STATE_UNAVAILABLE"):
        certify_structural_support(broken_inputs)
    false_rows = tuple(AttemptedSeed(task="game24", baseline="fh_bounded", arm=arm,
        trajectory_seed=seed, concrete_seed_id=str(seed), status="unattempted",
        reason="PROSPECTIVE_SEED_NOT_YET_ATTEMPTED",
        structural_support=False if seed == 0 else None,
        structural_support_evidence_sha256=fake_failure.evidence_sha256 if seed == 0 else None,
        structural_support_evidence=fake_failure if seed == 0 else None)
        for seed in range(10) for arm in ("clean", "correct", "irrelevant", "contam"))
    with pytest.raises(SeedResponseError, match="ATTEMPTED_SEED_BLOCK_MISMATCH"):
        assemble_seed_response_table(false_rows, support_inputs={0: inputs})
    with pytest.raises(SeedResponseError, match="STRUCTURAL_CERTIFICATE_MISMATCH"):
        validate_structural_support(fake_failure.model_copy(update={"supported": True}), inputs)
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
    bot_run = replace(run, baseline="bot_style", branch=bot_branches.arms["clean"],
                      client=ReachableBotResponses(), embedding_provider=NoveltyEmbedder())
    bot_identity = identity.model_copy(update={"execution_template_id": "game24:bot_style:clean"})
    bot_inputs = StructuralSupportInputs(bot_prefix, bot_context, candidates, bot_run, bot_identity, fixture_bytes)
    bot_proof = certify_structural_support(bot_inputs)
    assert bot_proof.supported, bot_proof.unsupported_reason
    validate_structural_support(bot_proof, bot_inputs)
    assert registered_pair_support(proof, bot_proof, inputs, bot_inputs) is True
    with pytest.raises(SeedResponseError, match="UNREGISTERED_OR_MISMATCHED_LEVEL2_PAIR"):
        registered_pair_support(bot_proof, proof, bot_inputs, inputs)
    with pytest.raises(SeedResponseError, match="STRUCTURAL_FAILURE_SLOT_MISMATCH"):
        validate_structural_support(bot_proof.model_copy(update={"unsupported_reason":
            "correct:3:SeedResponseError"}), bot_inputs)
    with pytest.raises(SeedResponseError, match="STRUCTURAL_CERTIFICATE_MISMATCH"):
        validate_structural_support(proof.model_copy(update={"handler_contract_sha256": "f" * 64}), inputs)
    with pytest.raises(SeedResponseError, match="STRUCTURAL_CERTIFICATE_MISMATCH"):
        validate_structural_support(proof.model_copy(update={"fixture_contract_sha256": "f" * 64}), inputs)
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
    with monkeypatch.context() as no_execution:
        no_execution.setattr("memcontam.evaluation.phase13_seed_response.execute_prospective_ordinary",
                             lambda *_args: pytest.fail("parent validation executed qualification"))
        no_execution.setattr(Answers, "__deepcopy__", ForbiddenClient.__deepcopy__, raising=False)
        client_free_run = replace(run, client=ForbiddenClient())
        client_free_parent = producer_joined_parent(client_free_run, result, identity, reopened,
            structural_support_evidence=proof, structural_support_inputs=validation_context)
        assert client_free_parent == parent
        assert extract_seed_response(reopened, packet, packet_sha, tasks, parent,
            run=client_free_run, result=result, identity=identity,
            structural_support_inputs=validation_context) == response
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


def test_native_transition_properties_reject_mutable_store_pruning_and_nonfifo():
    from memcontam.evaluation.phase13_seed_response import _validate_native_transition
    from memcontam.memory.checkpoint_v3 import NativeEntry

    def entry(name):
        return NativeEntry(name, "verbal_reflection", "native_entry_v1", "reflections",
                           name, hashlib.sha256(name.encode()).hexdigest())

    old = (entry("a"), entry("b"))
    new = entry("c")
    rag = NativeState("rag_frozen", old, {"read_only": True})
    _validate_native_transition(rag, rag, ("rag_generate",))
    with pytest.raises(SeedResponseError, match="RAG_IMMUTABILITY"):
        _validate_native_transition(rag, replace(rag, entries=(*old, new)), ("rag_generate",))
    bot = NativeState("bot_style", old, {"active_capacity": None})
    _validate_native_transition(bot, replace(bot, entries=(*old, new)), ())
    _validate_native_transition(bot, bot, ())
    with pytest.raises(SeedResponseError, match="BOT_INSERTION"):
        _validate_native_transition(bot, replace(bot, entries=(old[1], new)), ())
    fh = NativeState("fh_bounded", old, {})
    _validate_native_transition(fh, replace(fh, entries=(old[1], new)), ())
    with pytest.raises(SeedResponseError, match="FH_FIFO"):
        _validate_native_transition(fh, replace(fh, entries=(old[0], new)), ())
    reflexion = NativeState("reflexion_style", old, {})
    _validate_native_transition(reflexion, reflexion, ("reflexion_generate",))
    _validate_native_transition(reflexion, replace(reflexion, entries=(old[1], new)),
                                ("reflexion_generate", "reflexion_reflect"))
    with pytest.raises(SeedResponseError, match="REFLEXION_SUCCESS_WRITE"):
        _validate_native_transition(reflexion, replace(reflexion, entries=(*old, new)),
                                    ("reflexion_generate",))
    dc = NativeState("dc_rs", (), {"archive": ["old"]})
    _validate_native_transition(dc, replace(dc, native_state={"archive": ["old", "new"]}), ())
    with pytest.raises(SeedResponseError, match="DC_ARCHIVE_APPEND"):
        _validate_native_transition(dc, replace(dc, native_state={"archive": ["new"]}), ())
