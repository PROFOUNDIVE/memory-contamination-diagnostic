from __future__ import annotations

from dataclasses import replace
import importlib
import json
from types import SimpleNamespace
from typing import Literal

import pytest

from memcontam.baselines.reflexion_phase12 import (
    ReflexionContractError,
    ReflexionPhase12Adapter,
    ReflexionStateV3,
    ReflexionTrialContextV3,
)
from memcontam.clients.replay import ReplayClient
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import (
    NATIVE_ENTRY_V1,
    NativeEntry,
    NativeState,
    serialize_checkpoint,
)
from memcontam.memory.filtered_state import partition_native_checkpoint
from memcontam.memory.writer_registry import WriterRegistry
from memcontam.tasks.base import TaskInstance


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="game24-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
    )


def _reflection(entry_id: str, content: str) -> NativeEntry:
    return NativeEntry(
        entry_id=entry_id,
        semantic_kind="verbal_reflection",
        schema_version=NATIVE_ENTRY_V1,
        native_component="reflections",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def _envelope(
    entry: NativeEntry,
    *,
    trial_id: str | None,
    order_key: int,
    writer_id: str = "reflexion_reflector",
    writer_stage: str = "reflexion_reflect",
) -> MemoryCardEnvelopeV3:
    return MemoryCardEnvelopeV3(
        entry_id=entry.entry_id,
        baseline="reflexion_style",
        semantic_kind="verbal_reflection",
        schema_version=MEMORY_CARD_V3,
        writer_id=writer_id,
        writer_event_id=f"event:{entry.entry_id}",
        writer_stage=writer_stage,
        created_trial_id=trial_id,
        source_trial_ids=() if trial_id is None else (trial_id,),
        source_outcome=None,
        trial_support_ids=() if trial_id is None else (trial_id,),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=order_key,
        native_component="reflections",
        content=entry.content,
        content_hash=entry.content_hash,
    )


def _trial(
    *,
    branch: Literal["clean", "correct", "irrelevant", "contam", "filter"],
    max_attempts: int,
    responses: dict[str, object],
) -> ReflexionTrialContextV3:
    return ReflexionTrialContextV3(
        task=_task(),
        client=ReplayClient(responses_by_sample={"game24-1": responses}),
        model="replay",
        run_id="phase12-reflexion",
        trial_id=f"phase12-reflexion:{branch}",
        condition_id="reflexion_style",
        branch=branch,
        config={"max_attempts": max_attempts},
        order_key=2,
        verifier=lambda answer, _task: answer == "24",
    )


def _corrective_reflection(text: str, used_ids: list[str]) -> str:
    return json.dumps(
        {
            "mode": "corrective",
            "failure_class": "incorrect_answer",
            "reflection_text": text,
            "explicitly_used_memory_ids": used_ids,
        }
    )


def test_injected_reflection_conditions_actor_and_records_failed_actor_parent() -> None:
    injected = _reflection("injected-reflection", "Reflection: Use the controlled route.")
    trial = _trial(
        branch="contam",
        max_attempts=2,
        responses={
            "reflexion_generate": ["final: wrong", "final: 24"],
            "reflexion_reflect": _corrective_reflection(
                "Check the controlled route.", [injected.entry_id]
            ),
        },
    )
    state = ReflexionStateV3(
        reflections=[injected], injected_root_id=injected.entry_id, active_capacity=3
    )

    result = ReflexionPhase12Adapter().execute(trial, state)

    actor_call, reflection_call, retry_call = result.outcome.method_calls
    assert [call.stage for call in result.outcome.method_calls] == [
        "reflexion_generate",
        "reflexion_reflect",
        "reflexion_generate",
    ]
    assert injected.entry_id in actor_call.messages[1]["content"]
    assert "Failed actor response" not in retry_call.messages[1]["content"]
    assert result.outcome.verifier_result is True
    assert result.write_envelope is not None
    assert result.write_envelope.source_outcome is False
    assert result.write_envelope.direct_parent_ids == (injected.entry_id,)
    assert result.native_reflections[-1].direct_parent_ids == (injected.entry_id,)
    assert result.call_lineage_events[-1].actor_call_id == actor_call.call_id
    assert result.call_lineage_events[-1].reflection_call_id == reflection_call.call_id
    assert result.call_lineage_events[-1].failed_actor_call_id == actor_call.call_id


def test_failed_actor_produces_supported_admitted_reflection() -> None:
    injected = _reflection("injected-reflection", "Reflection: Use the controlled route.")
    trial = _trial(
        branch="contam",
        max_attempts=1,
        responses={
            "reflexion_generate": "final: wrong",
            "reflexion_reflect": _corrective_reflection(
                "Use the controlled route.", [injected.entry_id]
            ),
        },
    )
    result = ReflexionPhase12Adapter().execute(
        trial,
        ReflexionStateV3(
            reflections=[injected], injected_root_id=injected.entry_id, active_capacity=3
        ),
    )

    assert result.write_envelope is not None
    assert result.write_envelope.source_trial_ids == (trial.trial_id,)
    assert result.write_envelope.trial_support_ids == (trial.trial_id,)
    assert result.write_envelope.memory_support_ids == (injected.entry_id,)


def test_filter_quarantines_unadmitted_reflection_without_expanding_active_capacity() -> None:
    clean = _reflection("clean-reflection", "Reflection: Clean guidance.")
    injected = _reflection("injected-reflection", "Reflection: Quarantined guidance.")
    clean_envelope = _envelope(clean, trial_id="prefix-clean", order_key=1)
    injected_envelope = _envelope(
        injected,
        trial_id=None,
        order_key=2,
        writer_id="protocol_injector",
        writer_stage="protocol_inject",
    )
    context = AdmissionContext(
        writer_event_ids=frozenset({clean_envelope.writer_event_id}),
        trial_record_ids=frozenset({"prefix-clean"}),
        evidence_envelopes=(clean_envelope, injected_envelope),
    )
    filtered = partition_native_checkpoint(
        serialize_checkpoint(
            NativeState("reflexion_style", (clean, injected), {"reflections": []})
        ),
        context,
    )
    state = ReflexionStateV3(
        reflections=[clean, injected],
        injected_root_id=injected.entry_id,
        active_capacity=1,
        filter_state=filtered,
        admission_context=replace(context, writer_registry=WriterRegistry(())),
    )
    trial = _trial(
        branch="filter",
        max_attempts=1,
        responses={
            "reflexion_generate": "final: wrong",
            "reflexion_reflect": _corrective_reflection("Try another path.", []),
        },
    )

    result = ReflexionPhase12Adapter().execute(trial, state)

    assert result.filter_transition is not None
    assert result.filter_transition.decision.admitted is False
    assert result.eviction_events == ()
    assert result.outcome.memory_write_event is not None
    assert result.outcome.memory_write_event["status"] == "quarantined"
    assert [entry["entry_id"] for entry in result.outcome.memory_after] == [clean.entry_id]
    assert tuple(
        entry.entry_id if isinstance(entry, NativeEntry) else entry
        for entry in result.filter_transition.state.reader_entries
    ) == (clean.entry_id,)
    assert result.write_envelope is not None
    assert result.write_envelope.entry_id in {
        entry.entry_id
        for entry in result.filter_transition.state.quarantine.state.entries
        if isinstance(entry, NativeEntry)
    }


def test_reflection_calls_reject_non_text_tool_mode() -> None:
    with pytest.raises(ReflexionContractError, match="PRIMARY_TOOL_FORBIDDEN"):
        ReflexionTrialContextV3(
            task=_task(),
            client=ReplayClient(responses=["final: 24"]),
            model="replay",
            run_id="phase12-reflexion",
            trial_id="phase12-reflexion:tool",
            condition_id="reflexion_style",
            branch="clean",
            config={"tool_mode": "text_only"},
            order_key=2,
            tool_mode="python_sandbox",  # type: ignore[arg-type]
        )


def test_second_reflection_uses_a_later_native_order_than_its_explicit_parent(monkeypatch) -> None:
    first_id = "reflexion:game24:game24-1:first"
    second_id = "reflexion:game24:game24-1:second"
    reflexion_adapter = importlib.import_module("memcontam.baselines.reflexion_adapter")
    monkeypatch.setattr(
        reflexion_adapter,
        "uuid4",
        iter((SimpleNamespace(hex="first"), SimpleNamespace(hex="second"))).__next__,
    )
    clean = _reflection("clean-reflection", "Reflection: Clean guidance.")
    clean_envelope = _envelope(clean, trial_id="prefix-clean", order_key=1)
    context = AdmissionContext(
        writer_event_ids=frozenset({clean_envelope.writer_event_id}),
        trial_record_ids=frozenset({"prefix-clean"}),
        evidence_envelopes=(clean_envelope,),
    )
    filtered = partition_native_checkpoint(
        serialize_checkpoint(NativeState("reflexion_style", (clean,), {"reflections": []})), context
    )
    trial = _trial(
        branch="filter",
        max_attempts=2,
        responses={
            "reflexion_generate": ["final: wrong", "final: still-wrong"],
            "reflexion_reflect": [
                _corrective_reflection("Try a different operation.", []),
                _corrective_reflection("Build on the first reflection.", [first_id]),
            ],
        },
    )

    result = ReflexionPhase12Adapter().execute(
        trial,
        ReflexionStateV3(
            reflections=[clean], active_capacity=3, filter_state=filtered, admission_context=context
        ),
    )

    assert [reflection.entry_id for reflection in result.native_reflections] == [
        first_id,
        second_id,
    ]
    assert result.native_reflections[-1].direct_parent_ids == (first_id,)
    assert result.filter_transition is not None
    assert result.filter_transition.decision.admitted


def test_target_set_lineage_does_not_erase_explicit_reflection_parent() -> None:
    injected = _reflection("injected-reflection", "Reflection: Controlled guidance.")
    trial = replace(
        _trial(
            branch="contam",
            max_attempts=1,
            responses={
                "reflexion_generate": "final: wrong",
                "reflexion_reflect": _corrective_reflection(
                    "Use the controlled guidance.", [injected.entry_id]
                ),
            },
        ),
        config={"max_attempts": 1, "_logging_target_set_id": "phase12-target"},
    )

    result = ReflexionPhase12Adapter().execute(
        trial,
        ReflexionStateV3(
            reflections=[injected], injected_root_id=injected.entry_id, active_capacity=3
        ),
    )

    assert result.write_envelope is not None
    assert result.write_envelope.direct_parent_ids == (injected.entry_id,)


def test_active_capacity_evicts_the_oldest_reflection_after_a_new_write() -> None:
    injected = _reflection("injected-reflection", "Reflection: Controlled guidance.")
    trial = _trial(
        branch="contam",
        max_attempts=1,
        responses={
            "reflexion_generate": "final: wrong",
            "reflexion_reflect": _corrective_reflection("Try a different operation.", []),
        },
    )
    state = ReflexionStateV3(
        reflections=[injected], injected_root_id=injected.entry_id, active_capacity=1
    )

    result = ReflexionPhase12Adapter().execute(trial, state)

    assert [event.entry_id for event in result.eviction_events] == [injected.entry_id]
    assert state.first_injected_eviction_trial_id == trial.trial_id
    assert [entry.entry_id for entry in state.reflections] == [
        result.native_reflections[-1].entry_id
    ]


def test_rejects_future_access_capacity_error_and_visible_parent_union() -> None:
    future = _reflection("future-reflection", "Reflection: Future guidance.")
    before = _reflection("before-reflection", "Reflection: Earlier guidance.")
    invalid = replace(before, direct_parent_ids=(future.entry_id,))

    with pytest.raises(ReflexionContractError, match="FUTURE_REFLECTION_ACCESS"):
        ReflexionStateV3(reflections=[invalid, future], active_capacity=3)

    trial = _trial(
        branch="clean",
        max_attempts=1,
        responses={
            "reflexion_generate": "final: wrong",
            "reflexion_reflect": _corrective_reflection("Use no prior reflection.", []),
        },
    )
    result = ReflexionPhase12Adapter().execute(
        trial,
        ReflexionStateV3(reflections=[before, future], active_capacity=3),
    )

    assert result.write_envelope is not None
    assert result.write_envelope.direct_parent_ids == ()
    assert result.write_envelope.memory_support_ids == ()
