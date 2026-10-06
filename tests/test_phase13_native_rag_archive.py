import hashlib
import json
from pathlib import Path

import pytest

from memcontam.baselines.retrieval_rag_phase12 import (
    RagFrozenPhase12Adapter,
    RagFrozenStateV3,
    RagFrozenTrialContextV3,
)
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import (
    PHASE13_CORE_BASELINE_REGISTRY,
    RuntimeTrialResult,
)
from memcontam.experiment.phase13_ordinary_runtime import (
    ProspectiveOrdinaryResult,
    ProspectiveOrdinaryRun,
    execute_prospective_ordinary,
)
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.readiness.phase13_legacy_rag_runtime import (
    LegacyRagRuntimeRequest,
    load_legacy_rag_state,
)
from memcontam.readiness.phase13_main_checkpoint import CommonCheckpointRegistry
from memcontam.readiness.phase13_production_observability import validate_production_archive
from memcontam.readiness.phase13_production_runtime_join import (
    ProductionOrdinaryRunIdentity,
    production_archive_from_ordinary,
)
from memcontam.tasks.game24 import build_instance as build_game24
from memcontam.verifiers.game24 import verify_expression

from .test_phase13_native_production_archive import NativeResponses
from .test_phase13_native_rendering import _governed_renderers


def test_frozen_rag_carrier_stays_stable_across_current_seed0_50_row_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    from memcontam.experiment import phase13_ordinary_runtime
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    root = Path(__file__).resolve().parents[1]
    checkpoint = CommonCheckpointRegistry.model_validate_json(
        (root / "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json").read_bytes())
    sample_ids = checkpoint.tasks["game24"].seeds[0].suffix_sample_ids
    by_id = {row.sample_id: row for row in (
        build_game24(json.loads(line)) for line in (root / "data/phase13/main/game24_main_v1.jsonl").read_text().splitlines())}
    tasks = tuple(by_id[sample_id] for sample_id in sample_ids)
    embedder = BgeM3EmbeddingProvider()
    seal = json.loads((root / "data/phase13/rag/legacy_seal_v2.json").read_bytes())
    state = load_legacy_rag_state(LegacyRagRuntimeRequest(
        root / "data/phase13/rag/legacy_v2", root, "game24", "clean", embedder,
        seal["manifest_sha256"])).state
    runtime = PHASE13_CORE_BASELINE_REGISTRY["rag_frozen"]
    snapshot = runtime.serialize_state(state)
    assert isinstance(snapshot, NativeState)
    client = NativeResponses()
    context = Game24RuntimeContext(task=tasks[0], client=client, model="gpt-5.6-luna",
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0}, branch="clean", identities=RuntimeIdentities("rag", "rag:trial:1", 1),
        embedding_provider=embedder, initial_states={"rag_frozen": state})
    branch = build_live_reduced_main_branches(prefix=serialize_checkpoint(snapshot, checkpoint_index=0),
        context=context, candidate_registry=load_candidate_registry(Path("data/phase12/registries/candidate_registry_v2.json")),
        registry=PHASE13_CORE_BASELINE_REGISTRY, renderers=_governed_renderers()).arms["contam"]
    run = ProspectiveOrdinaryRun(task_name="game24", baseline="rag_frozen", run_id="native-frozen-rag",
        model="gpt-5.6-luna", client=client, allow_test_client=True,
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0}, arm="contam", branch=branch, tasks=tasks, trajectory_seed=0,
        embedding_provider=embedder)
    result = execute_prospective_ordinary(run)
    raw = (root / "data/phase13/observability/registration_packet_v2.json").read_bytes()
    identity = ProductionOrdinaryRunIdentity(execution_template_id="game24:rag_frozen:contam",
        trajectory_seed=0, concrete_seed_id="0", ordered_sample_ids_sha256=hashlib.sha256(
            json.dumps(result.sample_ids, separators=(",", ":")).encode()).hexdigest(),
        registration_packet_sha256=hashlib.sha256(raw).hexdigest(), scientific_result=True)

    archive = production_archive_from_ordinary(run, result, identity)

    assert len(archive.records) == 50
    memory_rows = tuple(row.evidence for row in archive.records if isinstance(row.evidence, Phase13TrialEvidence))
    assert len(memory_rows) == 50
    assert all(row.memory_before_ids == memory_rows[0].memory_before_ids for row in memory_rows)
    target_id = branch.injected_root_id
    assert target_id is not None
    assert any(target_id in row.memory_before_ids and target_id not in row.retrievals[0].retrieved_entry_ids
               for row in memory_rows)
    assert any(target_id in row.retrievals[0].retrieved_entry_ids for row in memory_rows)
    assert validate_production_archive(archive, ObservabilityRegistrationPacket.model_validate_json(raw),
                                       identity.registration_packet_sha256, frozen_tasks=run.tasks).status == "PASS"
    selected = next(index for index, row in enumerate(memory_rows)
                    if target_id in row.retrievals[0].retrieved_entry_ids)
    selected_task = tasks[selected]
    assert isinstance(branch.state, RagFrozenStateV3)
    included = tuple(document_id for document_id in memory_rows[selected].retrievals[0].retrieved_entry_ids
                     if document_id != target_id)
    formatted_run = ProspectiveOrdinaryRun(task_name="game24", baseline="rag_frozen",
        run_id="native-formatted-rag", model="gpt-5.6-luna", client=client, allow_test_client=True,
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0}, arm="contam", branch=branch,
        tasks=(selected_task,), trajectory_seed=0, embedding_provider=embedder)
    filtered = RagFrozenPhase12Adapter().execute(RagFrozenTrialContextV3(
        task=selected_task, client=client, model="gpt-5.6-luna", run_id=formatted_run.run_id,
        trial_id=f"{formatted_run.run_id}:contam:trial:1:{selected_task.sample_id}",
        condition_id="rag_frozen", branch="contam", rag_mode="frozen",
        included_document_ids=included,
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"])), branch.state)
    branch_snapshot = runtime.serialize_state(branch.state)
    assert isinstance(branch_snapshot, NativeState)
    formatted = ProspectiveOrdinaryResult("game24", "rag_frozen", "contam", (selected_task.sample_id,),
        (RuntimeTrialResult(filtered.outcome, branch.state, retrieval_event=filtered.retrieval_event,
                            context_event=filtered.context_event, state_before=branch_snapshot, state_after=branch_snapshot),))
    formatted_identity = identity.model_copy(update={
        "ordered_sample_ids_sha256": hashlib.sha256(json.dumps(formatted.sample_ids, separators=(",", ":")).encode()).hexdigest(),
        "execution_template_id": "game24:rag_frozen:contam:formatted",
    })
    formatted_archive = production_archive_from_ordinary(formatted_run, formatted, formatted_identity)
    formatted_evidence = formatted_archive.records[0].evidence
    assert isinstance(formatted_evidence, Phase13TrialEvidence)
    assert target_id in formatted_evidence.retrievals[0].retrieved_entry_ids
    assert formatted_evidence.context is not None
    assert target_id in formatted_evidence.context.removed_entry_ids
    assert target_id not in formatted_evidence.context.final_entry_ids
    assert validate_production_archive(formatted_archive, ObservabilityRegistrationPacket.model_validate_json(raw),
                                       formatted_identity.registration_packet_sha256,
                                       frozen_tasks=formatted_run.tasks).status == "PASS"
