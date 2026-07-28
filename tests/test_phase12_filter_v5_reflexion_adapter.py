from __future__ import annotations

import json
from importlib import import_module

import pytest

from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
from memcontam.baselines.reflexion_phase12 import ReflexionTrialContextV3
from memcontam.clients.replay import ReplayClient
from memcontam.experiment.phase12.filter_challenge.adapters.base import ChallengeAdapter
from memcontam.experiment.phase12.filter_challenge.adapters.reflexion_style import (
    ReflexionProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.experiment.phase12.filter_challenge.provenance import AnswerCallProvenanceObserver
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import (
    NATIVE_ENTRY_V1,
    NativeEntry,
    NativeState,
    Phase12Checkpoint,
    serialize_checkpoint,
)
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


def _checkpoint(entries: tuple[NativeEntry, ...], active_capacity: int) -> Phase12Checkpoint:
    return serialize_checkpoint(
        NativeState(
            "reflexion_style",
            entries,
            {
                "active_capacity": active_capacity,
                "first_injected_eviction_trial_id": None,
                "injected_root_id": None,
                "reflections": [entry.entry_id for entry in entries],
            },
        )
    )


def _candidate(checkpoint: Phase12Checkpoint) -> ChallengeCandidate:
    return ChallengeCandidate.model_validate(
        {
            "candidate_entry_id": "candidate",
            "candidate_native_content": "Reflection: candidate",
            "candidate_native_kind": "verbal_reflection",
            "baseline_family": "reflexion_style",
            "rag_mode": "not_applicable",
            "source_checkpoint_id": checkpoint.identity.checkpoint_id,
            "source_active_state_hash": checkpoint.canonical_sha256,
            "routability": {"routability": "challenge_routable_v1", "challenge_suite_key": "synthetic"},
        }
    )


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


def test_provisional_adapter_conforms_to_challenge_contract_and_binds_candidate_exposure() -> None:
    source_checkpoint = _checkpoint(
        tuple(_native_reflection(entry_id) for entry_id in ("one", "two", "three", "four")), 4
    )
    source_bytes, source_hash = source_checkpoint.canonical_bytes, source_checkpoint.canonical_sha256
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

    checkpoint = adapter_module.ReflexionFrozenCheckpoint(source_checkpoint, trial)
    adapter = ReflexionProvisionalAdapter()
    contract: ChallengeAdapter = adapter
    assert contract is adapter
    result = adapter.execute(checkpoint, _candidate(source_checkpoint))

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
    assert result.candidate_entry_id == "candidate"
    assert result.candidate_final_context_source_ids == result.final_context_source_ids
    assert observer._finalized[answer_call.call_id].answer_call_provenance_status == "explicit_matched"
    assert source_checkpoint.canonical_bytes == source_bytes
    assert source_checkpoint.canonical_sha256 == source_hash


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("baseline_family", "full_history", "REFLEXION_BASELINE_MISMATCH"),
        ("rag_mode", "frozen", "REFLEXION_RAG_MODE_MISMATCH"),
        ("candidate_native_kind", "thought_template", "REFLEXION_NATIVE_KIND_MISMATCH"),
        ("source_checkpoint_id", "other-checkpoint", "REFLEXION_CHECKPOINT_ID_MISMATCH"),
        ("source_active_state_hash", "other-hash", "REFLEXION_CHECKPOINT_HASH_MISMATCH"),
    ],
)
def test_provisional_adapter_rejects_mismatched_candidate_binding(
    field: str, value: str, code: str
) -> None:
    source_checkpoint = _checkpoint((_native_reflection("source"),), 4)
    trial = ReflexionTrialContextV3(
        task=_task(),
        client=ReplayClient(responses_by_sample={"sample-1": {"reflexion_generate": "final: right"}}),
        model="replay",
        run_id="provisional-reflexion",
        trial_id="provisional-reflexion:challenge",
        condition_id="reflexion_style",
        branch="contam",
        config={},
        order_key=1,
        verifier=lambda answer, _task: answer == "right",
    )
    adapter_module = import_module(
        "memcontam.experiment.phase12.filter_challenge.adapters.reflexion_style"
    )
    candidate = _candidate(source_checkpoint).model_copy(update={field: value})

    with pytest.raises(adapter_module.ReflexionChallengeError, match=code):
        ReflexionProvisionalAdapter().execute(
            adapter_module.ReflexionFrozenCheckpoint(source_checkpoint, trial), candidate
        )
