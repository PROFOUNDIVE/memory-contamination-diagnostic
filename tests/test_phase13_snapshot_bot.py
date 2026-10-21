from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.bot_read import DistilledProblem, build_distilled_query
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.game24 import verify_expression


@pytest.fixture(autouse=True)
def capacity_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    from memcontam.experiment import phase13_ordinary_runtime

    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)


class SnapshotTransport:
    def __init__(self, *, hit: bool = True, fail_stage: str | None = None) -> None:
        self.hit = hit
        self.fail_stage = fail_stage
        self.stages: list[str] = []

    def chat(self, messages, model, config) -> LLMResponse:
        stage = config["method_stage"]
        self.stages.append(stage)
        if stage == self.fail_stage:
            raise TimeoutError("task10 deterministic transport failure")
        payloads = {
            "bot_problem_distill": {
                "key_information": "four numbers 1 3 4 6", "restrictions": "exact use",
                "distilled_task": "make 24 using arithmetic",
            },
            "bot_instantiate_solve": {
                "selected_structure": "retrieved-template" if self.hit else "procedure-based",
                "solution_trace": "divide", "final_answer": "final: 6/(1-3/4)",
            },
            "bot_thought_distill": {
                "description": "Identify the migratory plumage of Arctic seabirds",
                "template": "Check all numbers", "category": "procedure-based",
                "explicitly_used_memory_ids": [],
            },
        }
        return LLMResponse(json.dumps(payloads[stage]), {"replay": True, "attempts": 1},
                           {"prompt_tokens": 1, "completion_tokens": 1}, 0)


@pytest.fixture(scope="module")
def embedder() -> BgeM3EmbeddingProvider:
    return BgeM3EmbeddingProvider()


def make_fixture(tmp_path: Path, embedder: BgeM3EmbeddingProvider, *, hit: bool = True):
    from memcontam.experiment.phase13_snapshot_diagnostic import (
        SnapshotBindings, SnapshotFileBinding, snapshot_input_sha256,
    )

    task = TaskInstance(sample_id="snapshot-game24", task_name="game24",
                        input={"numbers": [1, 3, 4, 6], "target": 24},
                        verifier_spec={"target": 24})
    description = (build_distilled_query(DistilledProblem(
        key_information="four numbers 1 3 4 6", restrictions="exact use",
        distilled_task="make 24 using arithmetic")) if hit else
        "Publish Kubernetes deployment manifests and rotate TLS certificates")
    root = NativeEntry("fixture-template", "thought_template", "phase12_native_entry_v1",
                       "buffer", "Check arithmetic", hashlib.sha256(b"Check arithmetic").hexdigest(),
                       retrieval_description=description, template_body="Check arithmetic",
                       category="procedure-based")
    original = BoTStateV3(entries=[root])
    runtime = PHASE13_CORE_BASELINE_REGISTRY["bot_style"]
    state = runtime.serialize_state(original)
    assert isinstance(state, NativeState)
    checkpoint = serialize_checkpoint(state, checkpoint_index=0)
    branch = LiveArmBranch("clean", checkpoint.identity.checkpoint_id,
                           checkpoint.identity.checkpoint_id, checkpoint, original, 0)
    run = ProspectiveOrdinaryRun(
        task_name="game24", baseline="bot_style", run_id="snapshot-one", model="gpt-5.6-luna",
        client=SnapshotTransport(hit=hit), allow_test_client=True,
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0}, tasks=(task,), trajectory_seed=0,
        branch=branch, embedding_provider=embedder,
    )
    candidate = tmp_path / "pending-fixture-candidate.json"
    candidate.write_bytes(b'{"fixture_only":true,"adopted":false}')
    root_path = Path(__file__).resolve().parents[1] / "src/memcontam"
    sources = tuple(root_path / relative for relative in (
        "experiment/phase12/runtime_registry.py", "experiment/phase13_ordinary_runtime.py",
        "baselines/bot_phase12.py", "baselines/bot_runtime.py",
        "baselines/bot_read.py", "baselines/bot_solve.py", "baselines/bot_write.py",
        "experiment/phase13_snapshot_diagnostic.py",
    ))
    packet = Path("data/phase13/observability/registration_packet_v2.json")
    bindings = SnapshotBindings(
        state_sha256=checkpoint.canonical_sha256,
        input_sha256=snapshot_input_sha256(run),
        source_files=tuple(SnapshotFileBinding(source, hashlib.sha256(source.read_bytes()).hexdigest()) for source in sources),
        candidate=SnapshotFileBinding(candidate, hashlib.sha256(candidate.read_bytes()).hexdigest()),
        archive_identity=ProductionOrdinaryRunIdentity(
            execution_template_id="game24:bot_style:clean", trajectory_seed=0, concrete_seed_id="0",
            ordered_sample_ids_sha256=hashlib.sha256(json.dumps((task.sample_id,), separators=(",", ":")).encode()).hexdigest(),
            registration_packet_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
            scientific_result=False,
        ),
    )
    return run, bindings, original


@pytest.mark.parametrize("hit", (True, False))
def test_native_hit_or_miss_when_snapshot_executes(tmp_path: Path, embedder, hit: bool) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import execute_snapshot_unit

    run, bindings, original = make_fixture(tmp_path, embedder, hit=hit)
    assert run.branch is not None
    assert isinstance(run.client, SnapshotTransport)
    assert bindings.candidate is not None
    before = run.branch.checkpoint.canonical_bytes
    candidate = bindings.candidate.path.read_bytes()
    result = execute_snapshot_unit(run, bindings, tmp_path)

    assert run.client.stages == ["bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill"]
    trial = result.ordinary.trials[0]
    assert isinstance(trial.state, BoTStateV3)
    assert trial.outcome.status == "succeeded"
    assert trial.outcome.metadata["retrieval_decision"]["decision"] == ("matched" if hit else "miss")
    assert bool(trial.outcome.retrieved_memory) is hit
    assert trial.native_entries and len(trial.state.entries) == 2
    assert len(original.entries) == 1
    assert run.branch.checkpoint.canonical_bytes == before
    assert bindings.candidate.path.read_bytes() == candidate
    assert result.provenance == "provider_free_qa"
    assert len(result.archive.records) == 1
    assert result.archive.records[0].scientific_result is False
    assert result.archive.records[0].evidence.trial.execution_status == "completed"


def test_clones_are_independent_when_same_snapshot_is_restored(tmp_path: Path, embedder) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import execute_snapshot_unit

    run, bindings, _ = make_fixture(tmp_path, embedder)
    first = execute_snapshot_unit(run, bindings, tmp_path)
    second = execute_snapshot_unit(replace(run, run_id="snapshot-two", client=SnapshotTransport()), bindings, tmp_path)

    one, two = first.ordinary.trials[0], second.ordinary.trials[0]
    assert isinstance(one.state, BoTStateV3) and isinstance(two.state, BoTStateV3)
    assert one.state_before == two.state_before
    assert one.state is not two.state
    assert one.state.entries is not two.state.entries
    assert {call.call_id for call in one.outcome.method_calls}.isdisjoint(call.call_id for call in two.outcome.method_calls)
    assert {entry.entry_id for entry in one.native_entries}.isdisjoint(entry.entry_id for entry in two.native_entries)


@pytest.mark.parametrize("field", ("state_sha256", "input_sha256", "source_files", "candidate", "archive_identity", "template", "registration", "empty_source", "wrong_source"))
def test_refuses_bad_binding_before_native_calls(tmp_path: Path, embedder, field: str) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import SnapshotError, execute_snapshot_unit

    run, bindings, _ = make_fixture(tmp_path, embedder)
    assert bindings.candidate is not None
    assert isinstance(run.client, SnapshotTransport)
    replacements = {
        "state_sha256": "0" * 64, "input_sha256": "0" * 64,
        "source_files": (replace(bindings.source_files[0], sha256="0" * 64),),
        "candidate": replace(bindings.candidate, sha256="0" * 64),
        "archive_identity": bindings.archive_identity.model_copy(update={"scientific_result": True}),
        "template": bindings.archive_identity.model_copy(update={"execution_template_id": "wrong-template"}),
        "registration": bindings.archive_identity.model_copy(update={"registration_packet_sha256": "0" * 64}),
        "empty_source": (),
        "wrong_source": (bindings.candidate,),
    }
    target = {"template": "archive_identity", "registration": "archive_identity", "empty_source": "source_files", "wrong_source": "source_files"}.get(field, field)
    with pytest.raises(SnapshotError):
        execute_snapshot_unit(run, replace(bindings, **{target: replacements[field]}), tmp_path)
    assert run.client.stages == []


def test_refuses_changed_task_and_post_state_before_native_calls(tmp_path: Path, embedder) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import SnapshotError, execute_snapshot_unit

    run, bindings, _ = make_fixture(tmp_path, embedder)
    assert run.branch is not None
    completed = execute_snapshot_unit(run, bindings, tmp_path)
    client = SnapshotTransport()
    post = completed.ordinary.trials[0].state
    reused = replace(run, run_id="bad-post", client=client, branch=replace(run.branch, state=post))
    with pytest.raises(SnapshotError, match="STATE"):
        execute_snapshot_unit(reused, bindings, tmp_path)
    altered = replace(run, run_id="bad-input", client=client,
                      tasks=(run.tasks[0].model_copy(update={"sample_id": "changed"}),))
    with pytest.raises(SnapshotError, match="INPUT"):
        execute_snapshot_unit(altered, bindings, tmp_path)
    assert client.stages == []


def test_refuses_reused_clone_when_completed(tmp_path: Path, embedder) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import SnapshotError, execute_snapshot_unit

    run, bindings, _ = make_fixture(tmp_path, embedder)
    execute_snapshot_unit(run, bindings, tmp_path)
    client = SnapshotTransport()
    with pytest.raises(SnapshotError, match="CLONE"):
        execute_snapshot_unit(replace(run, client=client), bindings, tmp_path)
    assert client.stages == []


@pytest.mark.parametrize("stage", ("bot_instantiate_solve", "bot_thought_distill"))
def test_keeps_partial_evidence_when_native_stage_fails(tmp_path: Path, embedder, stage: str) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import SnapshotError, execute_snapshot_unit

    run, bindings, original = make_fixture(tmp_path, embedder)
    client = SnapshotTransport(fail_stage=stage)
    result = execute_snapshot_unit(replace(run, client=client), bindings, tmp_path)

    trial = result.ordinary.trials[0]
    assert trial.outcome.status == "failed"
    assert trial.outcome.method_calls[-1].error_type is not None
    assert len(trial.outcome.method_calls) == (2 if stage == "bot_instantiate_solve" else 3)
    assert all(call.error_type is None for call in trial.outcome.method_calls[:-1])
    assert not trial.native_entries and len(original.entries) == 1
    assert trial.state_before == trial.state_after
    record = result.archive.records[0]
    assert record.evidence.trial.execution_status != "completed"
    assert record.evidence.verified_outcome is None
    assert record.terminal_provider_evidence is not None
    retry = SnapshotTransport()
    with pytest.raises(SnapshotError, match="CLONE"):
        execute_snapshot_unit(replace(run, client=retry), bindings, tmp_path)
    assert retry.stages == []


@pytest.mark.parametrize("kind", ("candidate", "source"))
def test_refuses_changed_file_bytes_when_pins_are_unchanged(tmp_path: Path, embedder, kind: str) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import SnapshotError, SnapshotFileBinding, execute_snapshot_unit

    run, bindings, _ = make_fixture(tmp_path, embedder)
    assert bindings.candidate is not None and isinstance(run.client, SnapshotTransport)
    target = bindings.candidate.path
    if kind == "source":
        target = tmp_path / "fixture-source.py"
        target.write_bytes(b"fixture = 1\n")
        pin = SnapshotFileBinding(target, hashlib.sha256(target.read_bytes()).hexdigest())
        bindings = replace(bindings, source_files=(*bindings.source_files, pin))
    target.write_bytes(b"tampered bytes")

    with pytest.raises(SnapshotError, match="FILE_BINDING"):
        execute_snapshot_unit(run, bindings, tmp_path)
    assert run.client.stages == []


def test_candidate_stays_pending_when_no_qualified_object_is_bound(tmp_path: Path, embedder) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import execute_snapshot_unit

    run, bindings, _ = make_fixture(tmp_path, embedder)
    result = execute_snapshot_unit(run, replace(bindings, candidate=None), tmp_path)

    assert result.bindings.candidate is None
    assert result.archive.records[0].scientific_result is False
