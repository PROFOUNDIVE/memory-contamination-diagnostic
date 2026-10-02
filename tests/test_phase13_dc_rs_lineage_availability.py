import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.clients.base import LLMResponse
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.evaluation.phase13_observability import reconstruct_phase13_trial
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.evaluation.phase13_observability_sequence import reconstruct_registered_sequence
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun, execute_prospective_ordinary
from memcontam.memory.checkpoint_v3 import CheckpointError, NativeEntry, NativeState, deserialize_checkpoint, serialize_checkpoint
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityArchive,
    ProductionObservabilityError,
    validate_production_archive,
)
from memcontam.readiness.phase13_production_runtime_join import ProductionOrdinaryRunIdentity, production_archive_from_ordinary
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.game24 import verify_expression

from .test_phase13_native_production_archive import NativeResponses
from .test_phase13_readiness0_production_dry_run import _ContractFakeEmbeddingProvider


class UnattributedResponses(NativeResponses):
    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        if config["method_stage"] == "dc_rs_synthesize":
            self.stages.append("dc_rs_synthesize")
            return LLMResponse("<cheatsheet>synthetic strategy</cheatsheet>",
                               {"replay": True, "attempts": 1},
                               {"prompt_tokens": 1, "completion_tokens": 1}, 0)
        return super().chat(messages, model, config)


def test_actual_dc_producer_preserves_unavailable_lineage_on_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memcontam.experiment import phase13_ordinary_runtime

    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    task = TaskInstance(sample_id="availability-synthetic", task_name="game24",
                        input={"numbers": [1, 3, 4, 6], "target": 24}, verifier_spec={"target": 24})
    tasks = (task, task.model_copy(update={"sample_id": "availability-synthetic-2"}))
    client = UnattributedResponses()
    embedder = _ContractFakeEmbeddingProvider(vector_dimension=1024)
    state = DcRsStateV3(archive=[], allow_unparented_strategies=True)
    runtime = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = Game24RuntimeContext(
        task=task, client=client, model="gpt-5.6-luna",
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0}, branch="clean",
        identities=RuntimeIdentities("availability", "availability:trial:1:availability-synthetic", 1),
        embedding_provider=embedder,
        baseline_configs={"dc_rs": {"serialized_cheatsheet_budget_tokens": 8192}},
        initial_states={"dc_rs": state},
    )
    registry_path = Path("data/phase12/registries/candidate_registry_v2.json")
    registry = load_candidate_registry(registry_path)
    renderers = RendererRegistry.governed(
        Path("data/phase13/main/legacy_dc_rs_intervention_registry_v2.json").read_bytes(),
        registry, hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    )
    snapshot = runtime.serialize_state(state)
    assert isinstance(snapshot, NativeState)
    branch = build_live_reduced_main_branches(
        prefix=serialize_checkpoint(snapshot, checkpoint_index=0), context=context,
        candidate_registry=registry, registry=PHASE13_CORE_BASELINE_REGISTRY, renderers=renderers,
    ).arms["contam"]
    run = ProspectiveOrdinaryRun(
        task_name="game24", baseline="dc_rs", run_id="availability", model="gpt-5.6-luna",
        client=client, allow_test_client=True, verifier=context.verifier, decoding=context.decoding,
        arm="contam", branch=branch, tasks=tasks, trajectory_seed=0, embedding_provider=embedder,
        baseline_configs=context.baseline_configs,
    )
    result = execute_prospective_ordinary(run)
    raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    packet = ObservabilityRegistrationPacket.model_validate_json(raw)
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="game24:dc_rs:contam", trajectory_seed=0, concrete_seed_id="0",
        scientific_result=True, registration_packet_sha256=hashlib.sha256(raw).hexdigest(),
        ordered_sample_ids_sha256=hashlib.sha256(json.dumps(result.sample_ids, separators=(",", ":")).encode()).hexdigest(),
    )
    archive = production_archive_from_ordinary(run, result, identity)
    archive_path = tmp_path / "archive.json"
    archive_path.write_text(archive.model_dump_json())
    reopened = ProductionObservabilityArchive.model_validate_json(archive_path.read_bytes())
    evidence_rows = tuple(record.evidence for record in reopened.records
                          if isinstance(record.evidence, Phase13TrialEvidence))
    analyses = tuple(reconstruct_phase13_trial(row) for row in evidence_rows)
    sequence = reconstruct_registered_sequence(evidence_rows, analyses, packet.recurrence_lookback_h, packet.failure_classes)
    assert client.stages == ["dc_rs_synthesize", "dc_rs_generate"] * 2
    assert all(envelope.lineage_status == "unavailable"
               for trial in result.trials for envelope in trial.write_envelopes
               if envelope.native_component == "strategy")
    assert len(evidence_rows) == 2
    for evidence, analysis in zip(evidence_rows, sequence, strict=True):
        assert evidence.context is not None
        strategy_id = evidence.context.final_entry_ids[0]
        strategy = next(node for node in evidence.lineage if node.entry_id == strategy_id)
        assert strategy.lineage_status == "unavailable"
        assert strategy.direct_parent_ids == ()
        assert strategy.injected_root_ids == ()
        for metric in (analysis.theory_exposure, analysis.target_final_context_included,
                       analysis.descendant_storage_persistence, analysis.descendant_prompt_visibility,
                       analysis.propagation, analysis.descendant_retention_duration):
            assert metric.status == "unavailable"
            assert metric.value is None
        assert analysis.target_present_in_store_before_answer.value is True
    second = evidence_rows[1]
    assert any(node.version_predecessor_id is not None and node.lineage_status == "unavailable"
               for node in second.lineage)
    assert validate_production_archive(reopened, packet, identity.registration_packet_sha256,
                                       frozen_tasks=tasks).status == "PASS"
    final = result.trials[-1].state_after
    assert isinstance(final, NativeState)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = serialize_checkpoint(final)
    checkpoint_path.write_bytes(checkpoint.canonical_bytes)
    restored_snapshot = NativeState.from_mapping(json.loads(checkpoint_path.read_bytes()))
    root = next(entry for entry in restored_snapshot.entries
                if isinstance(entry, NativeEntry) and entry.entry_id == branch.injected_root_id)
    restored = runtime.restore_state(restored_snapshot, replace(
        context, branch="contam", expected_intervention=root,
        identities=RuntimeIdentities("availability", "availability:trial:3:availability-synthetic-3", 3),
    ))
    assert runtime.serialize_state(restored) == final
    assert all(entry.to_mapping()["lineage_status"] == "unavailable"
               for entry in restored_snapshot.entries
               if isinstance(entry, NativeEntry) and entry.native_component == "strategy")
    forged_snapshot = replace(restored_snapshot, entries=tuple(
        replace(entry, lineage_status="exact")
        if isinstance(entry, NativeEntry) and entry.native_component == "strategy" else entry
        for entry in restored_snapshot.entries
    ))
    with pytest.raises(CheckpointError, match="CHECKPOINT_HASH_MISMATCH"):
        deserialize_checkpoint(replace(checkpoint, state=forged_snapshot))
    first_record = reopened.records[0]
    first = evidence_rows[0]
    assert first.context is not None
    forged = first.model_copy(update={"lineage": tuple(
        node.model_copy(update={"lineage_status": "exact"})
        if node.entry_id in first.context.final_entry_ids else node for node in first.lineage
    )})
    tampered = reopened.model_copy(update={"records": (
        first_record.model_copy(update={"evidence": forged}), *reopened.records[1:],
    )})
    with pytest.raises(ProductionObservabilityError):
        validate_production_archive(tampered, packet, identity.registration_packet_sha256, frozen_tasks=tasks)
