from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from memcontam.baselines.bot_phase12 import BoTPhase12Adapter, BoTStateV3, _as_memory_entry
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.experiment import phase13_ordinary_runtime as ordinary
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.readiness.phase13_production_observability import validate_production_archive
from memcontam.readiness.phase13_production_runtime_join import (
    ProductionOrdinaryRunIdentity,
    production_archive_from_ordinary,
)

from .test_phase12_bot import _memory_template, _trial
from .test_phase13_native_production_archive import NativeResponses, NoveltyEmbedder
from .test_phase12_bot import _task


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args, **kwargs):
        raise AssertionError("provider-free test attempted network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(ordinary, "_validated_common_capacity_tokens", lambda: 8192)


def test_bot_ancestry_survives_native_publication_and_reopen(tmp_path: Path) -> None:
    state = BoTStateV3(entries=[_memory_template("seed")])
    result = BoTPhase12Adapter().execute(
        _trial(branch="clean", used_ids=[], verifier=lambda _answer: True), state
    )

    assert result.native_entry is not None
    assert result.write_envelope is not None
    assert result.native_entry.lineage_status == result.write_envelope.lineage_status == "unavailable"
    runtime = PHASE13_CORE_BASELINE_REGISTRY["bot_style"]
    snapshot = runtime.serialize_state(state)
    assert isinstance(snapshot, NativeState)
    path = tmp_path / "native.json"
    path.write_bytes(serialize_checkpoint(snapshot).canonical_bytes)
    restored = runtime.restore_state(NativeState.from_mapping(json.loads(path.read_bytes())), None)
    assert isinstance(restored, BoTStateV3)
    entry = restored.entries[-1]
    assert isinstance(entry, NativeEntry)
    assert entry.lineage_status == "unavailable"
    assert entry.direct_parent_ids == ()
    assert _as_memory_entry(entry).metadata["source_lineage_status"] == "unavailable"
    assert _as_memory_entry(entry).metadata["lineage_status"] == "unavailable"


def test_bot_native_archive_keeps_unknown_ancestry_unavailable(tmp_path: Path) -> None:
    runtime = PHASE13_CORE_BASELINE_REGISTRY["bot_style"]
    embedder = NoveltyEmbedder()
    state = BoTStateV3(entries=[_memory_template("seed")])
    snapshot = runtime.serialize_state(state)
    assert isinstance(snapshot, NativeState)
    checkpoint = serialize_checkpoint(snapshot, checkpoint_index=0)
    branch = LiveArmBranch("clean", "prefix", "prefix", checkpoint, state, 0)
    task = _task().model_copy(update={"verifier_spec": {"target": 24}})
    run = ordinary.ProspectiveOrdinaryRun(
        task_name="game24", baseline="bot_style", run_id="bot-evidence",
        model="gpt-5.6-luna", client=NativeResponses(), allow_test_client=True,
        verifier=lambda _answer, _task: True, decoding={"temperature": 0.0},
        branch=branch, tasks=(task, task.model_copy(update={"sample_id": "second"})),
        trajectory_seed=0, embedding_provider=embedder,
    )
    result = ordinary.execute_prospective_ordinary(run)
    raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="game24:bot_style:clean", trajectory_seed=0, concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(json.dumps(result.sample_ids, separators=(",", ":")).encode()).hexdigest(),
        registration_packet_sha256=hashlib.sha256(raw).hexdigest(), scientific_result=False,
    )
    archive = production_archive_from_ordinary(run, result, identity)
    evidence = archive.records[0].evidence
    assert isinstance(evidence, Phase13TrialEvidence)
    assert evidence.new_entry_ids
    node = next(node for node in evidence.lineage if node.entry_id == evidence.new_entry_ids[0])
    assert node.lineage_status == "unavailable"
    assert node.direct_parent_ids == ()
    path = tmp_path / "archive.json"
    path.write_text(archive.model_dump_json())
    reopened = type(archive).model_validate_json(path.read_bytes())
    assert validate_production_archive(
        reopened, ObservabilityRegistrationPacket.model_validate_json(raw),
        identity.registration_packet_sha256, frozen_tasks=run.tasks,
    ).status == "PASS"


def test_bot_missing_claimed_parent_is_rejected_before_native_write() -> None:
    state = BoTStateV3(entries=[_memory_template("seed")])
    result = BoTPhase12Adapter().execute(
        _trial(branch="clean", used_ids=["absent"], verifier=lambda _answer: True), state
    )
    assert result.outcome.status == "failed"
    assert result.native_entry is None
    assert result.write_envelope is None
    assert result.outcome.memory_before == result.outcome.memory_after


def test_bot_restored_reader_preserves_exact_parent_ids() -> None:
    from dataclasses import replace
    from .test_phase12_bot import _native_template

    entry = replace(_native_template("child", "procedure"), direct_parent_ids=("root",))
    reader_entry = _as_memory_entry(entry)
    assert reader_entry.metadata["source_entry_ids"] == ["root"]
    assert reader_entry.metadata["source_lineage_status"] == "exact"
