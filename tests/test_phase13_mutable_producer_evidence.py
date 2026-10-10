from __future__ import annotations

import json
import hashlib
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.baselines.reflexion_phase12 import ReflexionPhase12Adapter, ReflexionStateV3
from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.clients.replay import ReplayClient
from memcontam.experiment.phase12.game24_runner import RuntimeIdentities
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY, RuntimeStateError
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.memory.stores import MemoryEntry
from memcontam.logging.schema_v3 import RetrievalEvent
from memcontam.readiness.phase13_production_runtime_evidence import _context as answer_context

from .test_phase12_reflexion import _trial, _corrective_reflection
from .test_phase13_dc_rs_runtime import _context, _BombClient


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args, **kwargs):
        raise AssertionError("provider-free test attempted network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


@pytest.mark.parametrize("answer,reflection", [
    ("final: 24", None), ("not a final answer", None),
    ("final: wrong", "invalid reflection"),
    ("final: wrong", _corrective_reflection("check arithmetic", [])),
])
def test_reflexion_native_empty_context_binds_actor_not_auxiliary(answer, reflection) -> None:
    responses = {"reflexion_generate": answer}
    if reflection is not None:
        responses["reflexion_reflect"] = reflection
    result = ReflexionPhase12Adapter().execute(
        _trial(branch="clean", max_attempts=1, responses=responses), ReflexionStateV3(reflections=[])
    )
    outcome = result.outcome
    context = answer_context(None, outcome.metadata, "phase12-reflexion", "phase12-reflexion:clean", (), outcome.method_calls, outcome.answer_call_id)
    assert context is not None
    assert context.final_entry_ids == []
    assert outcome.answer_call_id == outcome.method_calls[0].call_id
    assert len(result.write_envelopes) == int(reflection is not None and reflection != "invalid reflection")


@pytest.mark.parametrize("generated", ["no answer envelope", "final: A"])
def test_dc_failure_history_survives_durable_restore_and_later_curation(generated: str, tmp_path: Path) -> None:
    runtime = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()
    client = ReplayClient(responses_by_sample={context.task.sample_id: {
        "dc_rs_synthesize": "<cheatsheet>ordinary strategy</cheatsheet><source_ids>src01</source_ids>",
        "dc_rs_generate": generated,
    }})
    context = replace(context, client=client)
    state = runtime.initial_state(context)
    before = runtime.serialize_state(state)
    result = runtime.execute_trial(context, state)
    assert isinstance(result.state, DcRsStateV3)
    assert result.outcome.status == ("failed" if generated == "no answer envelope" else "succeeded")
    assert result.outcome.verifier_result is (None if generated == "no answer envelope" else False)
    assert len(result.state.archive) == 2
    archived = result.state.archive[-1]
    assert isinstance(archived, MemoryEntry)
    assert archived.metadata["generated_output"] == generated
    assert archived.metadata["parsed_answer"] == (None if generated == "no answer envelope" else "A")
    assert result.write_envelopes[0].source_outcome is None
    snapshot = runtime.serialize_state(result.state)
    assert isinstance(snapshot, NativeState)
    assert before != snapshot
    path = tmp_path / "dc-checkpoint.json"
    checkpoint = serialize_checkpoint(snapshot, checkpoint_index=2)
    path.write_bytes(checkpoint.canonical_bytes)
    reopened = NativeState.from_mapping(json.loads(path.read_bytes()))
    later = replace(_context(), identities=RuntimeIdentities("run-1", "run-1:trial:3:later", 3, "dc_rs"))
    restored = runtime.restore_state(reopened, later)
    assert runtime.serialize_state(restored) == snapshot
    subsequent = runtime.execute_trial(later, restored)
    assert subsequent.outcome.status == "succeeded"
    assert isinstance(subsequent.retrieval_event, RetrievalEvent)
    assert archived.entry_id in subsequent.retrieval_event.retrieved_entry_ids
    assert [call.stage for call in subsequent.outcome.method_calls] == ["dc_rs_synthesize", "dc_rs_generate"]


def test_dc_unknown_native_key_rejects_before_provider_attempt() -> None:
    runtime = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()
    snapshot = runtime.serialize_state(runtime.initial_state(context))
    assert isinstance(snapshot, NativeState)
    bomb = _BombClient()
    malformed = replace(snapshot, native_state={**snapshot.native_state, "checkpoint_index": 1})
    with pytest.raises(RuntimeStateError, match="INVALID_DC_RS_SNAPSHOT"):
        runtime.restore_state(malformed, replace(context, client=bomb))
    assert bomb.calls == 0


@pytest.mark.parametrize("admit_later", [False, True])
def test_rejected_intermediate_reflection_stays_quarantined_when_later_write_routes(
    monkeypatch: pytest.MonkeyPatch, admit_later: bool,
) -> None:
    from memcontam.baselines import reflexion_phase12 as reflexion
    from memcontam.memory.admission import AdmissionContext
    from memcontam.memory.checkpoint_v3 import NativeEntry
    from memcontam.memory.filtered_state import partition_native_checkpoint
    from memcontam.memory.writer_registry import WriterRegistry
    from .test_phase12_reflexion import _reflection, _envelope

    clean = _reflection("clean", "Reflection: clean guidance")
    envelope = _envelope(clean, trial_id="prefix", order_key=1)
    admission = AdmissionContext(
        writer_event_ids=frozenset({envelope.writer_event_id}),
        trial_record_ids=frozenset({"prefix"}), evidence_envelopes=(envelope,),
    )
    filtered = partition_native_checkpoint(
        serialize_checkpoint(NativeState("reflexion_style", (clean,), {"reflections": []})), admission,
    )
    state = ReflexionStateV3(
        reflections=[clean], active_capacity=3, filter_state=filtered,
        admission_context=replace(admission, writer_registry=WriterRegistry(())),
    )
    route = reflexion._route_filter_write

    def route_and_enable_later(state, native, envelope, trial):
        transition = route(state, native, envelope, trial)
        if admit_later:
            state.admission_context = replace(state.admission_context, writer_registry=admission.writer_registry)
        return transition

    monkeypatch.setattr(reflexion, "_route_filter_write", route_and_enable_later)
    result = ReflexionPhase12Adapter().execute(_trial(
        branch="filter", max_attempts=2, responses={
            "reflexion_generate": ["final: wrong", "final: still-wrong"],
            "reflexion_reflect": [_corrective_reflection("first", []), _corrective_reflection("second", [])],
        },
    ), state)
    first, second = result.native_reflections
    assert result.filter_transition is not None
    quarantine = {entry.entry_id for entry in result.filter_transition.state.quarantine.state.entries if isinstance(entry, NativeEntry)}
    assert first.entry_id in quarantine
    assert (second.entry_id in quarantine) is not admit_later
    active_ids = [row["entry_id"] for row in result.outcome.memory_after]
    assert active_ids == (["clean", second.entry_id] if admit_later else ["clean"])
    assert result.outcome.memory_write_event is not None
    assert result.outcome.memory_write_event["status"] == ("accepted" if admit_later else "quarantined")
    answer = next(call for call in result.outcome.method_calls if call.call_id == result.outcome.answer_call_id)
    assert [span.entry_id for span in answer.source_spans] == ["clean"]


def test_native_reflexion_evicted_root_remains_in_reopened_archive_lineage(monkeypatch, tmp_path: Path) -> None:
    from memcontam.clients.base import LLMResponse
    from memcontam.experiment import phase13_ordinary_runtime as ordinary
    from memcontam.experiment.phase12.live_branch import LiveArmBranch
    from memcontam.evaluation.phase13_observability import reconstruct_phase13_trial
    from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
    from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
    from memcontam.readiness.phase13_production_runtime_join import ProductionOrdinaryRunIdentity, production_archive_from_ordinary
    from memcontam.readiness.phase13_production_observability import validate_production_archive
    from .test_phase12_reflexion import _reflection
    from .test_phase12_bot import _task

    class Responses:
        def __init__(self) -> None:
            self.actors = 0

        def chat(self, messages, model, config) -> LLMResponse:
            if config["method_stage"] == "reflexion_generate":
                self.actors += 1
                output = "final: 1+3+4+6" if self.actors == 1 else "final: 6/(1-3/4)"
            else:
                output = _corrective_reflection("check arithmetic", ["root"])
            return LLMResponse(output, {"replay": True, "attempts": 1}, {}, 0)

    monkeypatch.setattr(ordinary, "_validated_common_capacity_tokens", lambda: 8192)
    runtime = PHASE13_CORE_BASELINE_REGISTRY["reflexion_style"]
    state = ReflexionStateV3(reflections=[_reflection("root", "Reflection: root")], injected_root_id="root", active_capacity=1)
    snapshot = runtime.serialize_state(state)
    assert isinstance(snapshot, NativeState)
    checkpoint = serialize_checkpoint(snapshot, checkpoint_index=0)
    branch = LiveArmBranch("contam", "prefix", "prefix", checkpoint, state, 1, "root", "game24-parentheses-restriction-v2", "native-reflection")
    task = _task().model_copy(update={"verifier_spec": {"target": 24}})
    run = ordinary.ProspectiveOrdinaryRun(
        task_name="game24", baseline="reflexion_style", arm="contam", run_id="eviction",
        model="gpt-5.6-luna", client=Responses(), allow_test_client=True,
        verifier=lambda answer, _task: answer == "6/(1-3/4)", decoding={"temperature": 0.0},
        branch=branch, tasks=(task, task.model_copy(update={"sample_id": "later"})), trajectory_seed=0,
        baseline_configs={"reflexion_style": {"max_attempts": 1}},
    )
    result = ordinary.execute_prospective_ordinary(run)
    raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="game24:reflexion_style:contam", trajectory_seed=0, concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(json.dumps(result.sample_ids, separators=(",", ":")).encode()).hexdigest(),
        registration_packet_sha256=hashlib.sha256(raw).hexdigest(), scientific_result=False,
    )
    archive = production_archive_from_ordinary(run, result, identity)
    path = tmp_path / "reflexion-archive.json"
    path.write_text(archive.model_dump_json())
    reopened = type(archive).model_validate_json(path.read_bytes())
    first, second = (record.evidence for record in reopened.records)
    assert isinstance(first, Phase13TrialEvidence)
    assert isinstance(second, Phase13TrialEvidence)
    assert "root" in first.removed_entry_ids
    assert "root" not in second.memory_before_ids
    assert any(node.entry_id == "root" for node in second.lineage)
    assert second.context is not None
    assert second.context.final_entry_ids == list(first.new_entry_ids)
    assert second.target_set.answer_call_spans[0].direct_parent_ids == ["root"]
    assert reconstruct_phase13_trial(second).theory_exposure.value is True
    assert validate_production_archive(reopened, ObservabilityRegistrationPacket.model_validate_json(raw), identity.registration_packet_sha256, frozen_tasks=run.tasks).status == "PASS"
