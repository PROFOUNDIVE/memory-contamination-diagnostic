from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Literal

import pytest

from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.clients.base import LLMResponse
from memcontam.evaluation.phase13_observability import reconstruct_phase13_trial
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.experiment import phase13_ordinary_runtime as ordinary
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityError,
    validate_production_archive,
)
from memcontam.readiness.phase13_production_runtime_join import (
    ProductionOrdinaryRunIdentity,
    production_archive_from_ordinary,
)
from memcontam.tasks.base import TaskInstance


class QueryEmbedder:
    embedding_contract = {
        "dimension": 2, "normalized": True, "provider": "test",
        "production_identity": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
    }

    def encode_document(self, text: str) -> list[float]:
        return {"first": [1.0, 0.0], "second": [0.0, 1.0], "target": [-1.0, 0.0]}[text]

    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if json.loads(text)["input"]["numbers"] == [1, 3, 4, 6] else [0.0, 1.0]


class Answers:
    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        return LLMResponse("final: 1+3+4+6", {"replay": True, "attempts": 1}, {}, 0)


@pytest.mark.parametrize("arm,horizon", [("clean", 2), ("correct", 2), ("irrelevant", 2), ("contam", 2), ("contam", 50)])
def test_two_native_rag_reads_preserve_carrier_and_reject_manual_continuity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    arm: Literal["clean", "correct", "irrelevant", "contam"],
    horizon: int,
) -> None:
    def denied(*args, **kwargs):
        raise AssertionError("provider-free test attempted network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(ordinary, "_validated_common_capacity_tokens", lambda: 8192)
    corpora = build_branch_corpora(
        CleanCorpus.from_documents(
            [{"id": f"doc-{index}", "text": "first" if index < 3 else "second"} for index in range(6)],
            corpus_id="native-carrier",
        ),
        {kind: {"id": kind, "text": "target"} for kind in ("false", "correct", "irrelevant")},
    )
    indices = build_branch_indices(corpora, QueryEmbedder(), filter_policy=None)
    state = RagFrozenStateV3(arm, corpora.branches[arm], indices.branches[arm])
    snapshot = PHASE13_CORE_BASELINE_REGISTRY["rag_frozen"].serialize_state(state)
    assert isinstance(snapshot, NativeState)
    checkpoint = serialize_checkpoint(snapshot, checkpoint_index=0)
    branch = LiveArmBranch(arm, "prefix", "prefix", checkpoint, state, 0 if arm == "clean" else 1,
                           None if arm == "clean" else "false" if arm == "contam" else arm,
                           None if arm == "clean" else "game24-parentheses-restriction-v2",
                           None if arm == "clean" else "native-rag-render")
    tasks = tuple(
        TaskInstance(sample_id=f"native-{index}", task_name="game24", input={"numbers": numbers, "target": 24}, verifier_spec={"target": 24})
        for index, numbers in enumerate(([1, 3, 4, 6], [1, 1, 1, 8]) * (horizon // 2))
    )
    run = ordinary.ProspectiveOrdinaryRun(
        task_name="game24", baseline="rag_frozen", arm=arm, run_id="native-carrier",
        model="gpt-5.6-luna", client=Answers(), allow_test_client=True,
        verifier=lambda _answer, _task: False, decoding={"temperature": 0.0},
        branch=branch, tasks=tasks, trajectory_seed=0,
    )
    result = ordinary.execute_prospective_ordinary(run)
    raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="game24:rag_frozen:contam", trajectory_seed=0, concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(json.dumps(result.sample_ids, separators=(",", ":")).encode()).hexdigest(),
        registration_packet_sha256=hashlib.sha256(raw).hexdigest(), scientific_result=False,
    )
    archive = production_archive_from_ordinary(run, result, identity)
    path = tmp_path / "archive.json"
    path.write_text(archive.model_dump_json())
    reopened = type(archive).model_validate_json(path.read_bytes())
    packet = ObservabilityRegistrationPacket.model_validate_json(raw)
    assert validate_production_archive(reopened, packet, identity.registration_packet_sha256, frozen_tasks=tasks).status == "PASS"
    assert len(reopened.records) == horizon
    first, second = (record.evidence for record in reopened.records[:2])
    assert isinstance(first, Phase13TrialEvidence)
    assert isinstance(second, Phase13TrialEvidence)
    assert first.memory_before_ids == first.memory_after_ids == second.memory_before_ids == second.memory_after_ids
    assert all(record.evidence.memory_before_ids == record.evidence.memory_after_ids == first.memory_before_ids
               for record in reopened.records if isinstance(record.evidence, Phase13TrialEvidence))
    assert len(first.memory_before_ids) == (6 if arm == "clean" else 7)
    assert reopened.schema_version == "phase13_production_observability_archive_v1"
    assert all(record.scientific_result is False for record in reopened.records)
    assert reopened.u_t_status == "NOT_REGISTERED_FOR_CURRENT_MAIN"
    assert first.retrievals[0].retrieved_entry_ids != second.retrievals[0].retrieved_entry_ids
    for evidence in (first, second):
        assert evidence.context is not None
        assert evidence.target_set.target_entry_ids == (("false",) if arm == "contam" else ())
        assert evidence.context.final_entry_ids == evidence.retrievals[0].retrieved_entry_ids
        assert "false" not in evidence.context.final_entry_ids
        assert evidence.verified_outcome == 0
        assert evidence.trial.execution_status == "completed"
        exposure = reconstruct_phase13_trial(evidence).theory_exposure
        assert exposure.value is (False if arm == "contam" else None)
    if arm != "contam":
        return
    forged = reopened.model_copy(update={"records": (
        reopened.records[0], reopened.records[1].model_copy(update={
            "evidence": second.model_copy(update={"memory_before_ids": tuple(second.retrievals[0].retrieved_entry_ids)})
        }),
    )})
    with pytest.raises(ProductionObservabilityError, match="PRODUCTION_RECONSTRUCTION_FAILED") as failure:
        validate_production_archive(forged, packet, identity.registration_packet_sha256, frozen_tasks=tasks)
    assert "ORDINARY_SEQUENCE_CONTINUITY_MISMATCH" in str(failure.value.__cause__)
