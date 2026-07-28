from __future__ import annotations

import hashlib
import json
from importlib import import_module

from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3, ReflexionTrialContextV3
from memcontam.clients.replay import ReplayClient
from memcontam.experiment.phase12.filter_challenge.provenance import AnswerCallProvenanceObserver
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


def _task() -> TaskInstance:
    return TaskInstance(sample_id="sample-1", task_name="game24", input={"numbers": [1, 2, 3, 4]})


def _reflection_response(text: str) -> str:
    return json.dumps(
        {
            "mode": "corrective",
            "failure_class": "incorrect_answer",
            "reflection_text": text,
            "explicitly_used_memory_ids": [],
        }
    )


def _native_reflection(entry_id: str) -> NativeEntry:
    content = f"Reflection: {entry_id}"
    return NativeEntry(
        entry_id=entry_id,
        semantic_kind="verbal_reflection",
        schema_version=NATIVE_ENTRY_V1,
        native_component="reflections",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def _state_hash(state: ReflexionStateV3) -> str:
    checkpoint = serialize_checkpoint(
        NativeState(
            "reflexion_style",
            tuple(state.reflections),
            {
                "active_capacity": state.active_capacity,
                "first_injected_eviction_trial_id": state.first_injected_eviction_trial_id,
                "injected_root_id": state.injected_root_id,
                "reflections": [entry.entry_id for entry in state.reflections],
            },
        )
    )
    return hashlib.sha256(checkpoint.canonical_bytes).hexdigest()


def test_default_updater_retries_and_writes_after_a_failed_actor_answer() -> None:
    state = ReflexionState()

    outcome = ReflexionAdapter().execute(
        _task(),
        state,
        client=ReplayClient(
            responses_by_sample={
                "sample-1": {
                    "reflexion_generate": ["final: wrong", "final: right"],
                    "reflexion_reflect": _reflection_response("Retry with the correct route."),
                }
            }
        ),
        model="replay",
        config={"run_id": "default-update"},
        verifier=lambda answer, _task: answer == "right",
    )

    assert [call.stage for call in outcome.method_calls] == [
        "reflexion_generate",
        "reflexion_reflect",
        "reflexion_generate",
    ]
    assert outcome.memory_write_event is not None
    assert len(state.reflections) == 1


def test_read_only_update_disabled_answers_once_without_reflection_retry_or_write() -> None:
    state = ReflexionState(
        reflections=[MemoryEntry(entry_id="source", content="Reflection: source", memory_type="verbal_reflection")]
    )

    outcome = ReflexionAdapter().execute(
        _task(),
        state,
        client=ReplayClient(
            responses_by_sample={
                "sample-1": {
                    "reflexion_generate": ["final: wrong", "final: right"],
                    "reflexion_reflect": _reflection_response("Must not be called."),
                }
            }
        ),
        model="replay",
        config={"run_id": "read-only-update", "update_enabled": False},
        verifier=lambda answer, _task: answer == "right",
    )

    assert [call.stage for call in outcome.method_calls] == ["reflexion_generate"]
    assert outcome.verifier_result is False
    assert outcome.memory_write_event is None
    assert [entry.entry_id for entry in state.reflections] == ["source"]
    assert outcome.metadata["reflexion_reflection_events"] == []


def test_provisional_adapter_appends_newest_native_candidate_and_uses_answer_source_ids() -> None:
    source_state = ReflexionStateV3(
        reflections=[_native_reflection(entry_id) for entry_id in ("one", "two", "three", "four")],
        active_capacity=4,
    )
    source_hash = _state_hash(source_state)
    candidate = _native_reflection("candidate")
    observer = AnswerCallProvenanceObserver()
    trial = ReflexionTrialContextV3(
        task=_task(),
        client=ReplayClient(responses_by_sample={"sample-1": {"reflexion_generate": "final: right"}}),
        model="replay",
        run_id="provisional-reflexion",
        trial_id="provisional-reflexion:challenge",
        condition_id="reflexion_style",
        branch="contam",
        config={"_logging_answer_call_provenance_observer": observer},
        order_key=1,
        verifier=lambda answer, _task: answer == "right",
    )
    adapter_module = import_module(
        "memcontam.experiment.phase12.filter_challenge.adapters.reflexion_style"
    )

    result = adapter_module.ReflexionProvisionalAdapter().execute(trial, source_state, candidate)

    answer_call = result.outcome.method_calls[0]
    assert [entry["entry_id"] for entry in result.outcome.memory_before] == [
        "two",
        "three",
        "four",
        "candidate",
    ]
    assert result.displaced_reflection_ids == ("one",)
    assert result.final_context_source_ids == tuple(span.entry_id for span in answer_call.source_spans)
    assert result.final_context_source_ids == ("three", "four", "candidate")
    assert result.candidate_final_context_inclusion is True
    assert result.candidate_final_context_inclusion == (
        candidate.entry_id in result.final_context_source_ids
    )
    assert observer._finalized[answer_call.call_id].answer_call_provenance_status == "explicit_matched"
    assert [entry.entry_id for entry in source_state.reflections] == ["one", "two", "three", "four"]
    assert _state_hash(source_state) == source_hash
