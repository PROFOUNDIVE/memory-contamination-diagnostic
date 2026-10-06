from dataclasses import replace
import json
import socket

import httpx
import openai
import pytest

from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.experiment.phase12.game24_runner import (
    Game24RuntimeContext, RuntimeIdentities, RuntimeWriterCallbacks,
)
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment import phase13_ordinary_runtime as ordinary_runtime
from memcontam.logging.schema import MethodCall
from memcontam.logging.schema_v3 import ContextEvent, RetrievalEvent
from memcontam.memory.checkpoint_v3 import CheckpointError, NativeState, serialize_checkpoint
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.game24 import verify_expression

from .test_phase13_dc_rs_runtime import _context as dc_context
from .test_phase13_native_production_archive import NativeResponses, ReflectionResponses


@pytest.fixture(autouse=True)
def deny_external_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: str, **kwargs: str) -> None:
        raise AssertionError("PHASE13_EXTERNAL_TRANSPORT_FORBIDDEN")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(httpx.Client, "send", denied)
    monkeypatch.setattr(httpx.AsyncClient, "send", denied)
    monkeypatch.setattr(openai, "OpenAI", denied)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


@pytest.mark.parametrize("reflect", (False, True))
def test_reflexion_producer_emits_selected_answer_context_when_actor_finishes(
    reflect: bool,
) -> None:
    task = TaskInstance(
        sample_id="context-game24",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
        verifier_spec={"target": 24},
    )
    client = ReflectionResponses(final_success=True) if reflect else NativeResponses()
    context = Game24RuntimeContext(
        task=task, client=client, model="replay",
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0}, branch="clean",
        identities=RuntimeIdentities("context", "context:trial:1", 1, "reflexion_style"),
    )

    result = PHASE13_CORE_BASELINE_REGISTRY["reflexion_style"].execute_trial(
        context, ReflexionStateV3(reflections=[])
    )

    assert isinstance(result.context_event, ContextEvent)
    selected = next(
        call for call in result.outcome.method_calls
        if isinstance(call, MethodCall) and call.call_id == result.outcome.answer_call_id
    )
    assert result.context_event.final_entry_ids == list(
        dict.fromkeys(span.entry_id for span in selected.source_spans)
    )
    assert bool(result.context_event.final_entry_ids) is reflect
    assert result.context_event.trial_id == context.identities.trial_id


def test_dc_producer_emits_answer_context_separate_from_curator_retrieval() -> None:
    context = dc_context()
    runtime = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    result = runtime.execute_trial(context, runtime.initial_state(context))

    assert isinstance(result.context_event, ContextEvent)
    assert isinstance(result.retrieval_event, RetrievalEvent)
    selected = next(
        call for call in result.outcome.method_calls
        if isinstance(call, MethodCall) and call.call_id == result.outcome.answer_call_id
    )
    assert result.context_event.final_entry_ids == [span.entry_id for span in selected.source_spans]
    assert "archive-root" not in result.context_event.final_entry_ids
    assert result.context_event.event_seq > result.retrieval_event.event_seq


@pytest.mark.parametrize("location", ("state", "entry"))
def test_checkpoint_reopen_rejects_unknown_native_fields_before_dispatch(location: str) -> None:
    context = dc_context()
    runtime = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    snapshot = runtime.serialize_state(runtime.initial_state(context))
    assert isinstance(snapshot, NativeState)
    checkpoint = serialize_checkpoint(snapshot, checkpoint_index=2)
    payload = json.loads(checkpoint.canonical_bytes)
    if location == "entry":
        payload["entries"][0]["fabricated_provenance"] = "root"
    else:
        payload["fabricated_provenance"] = "root"

    with pytest.raises(CheckpointError, match="UNKNOWN_NATIVE_.*_FIELD"):
        NativeState.from_mapping(payload)


def test_dc_strict_restore_rejects_envelope_index_inside_native_payload() -> None:
    context = dc_context()
    runtime = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    snapshot = runtime.serialize_state(runtime.initial_state(context))
    assert isinstance(snapshot, NativeState)
    malformed = replace(snapshot, native_state={**snapshot.native_state, "checkpoint_index": 2})

    with pytest.raises(ValueError, match="INVALID_DC_RS_SNAPSHOT"):
        runtime.restore_state(malformed, context)


def test_ordinary_callbacks_publish_selected_context_and_frozen_mutation_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    task = TaskInstance(
        sample_id="context-suffix-1", task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24}, verifier_spec={"target": 24},
    )
    events: list[ContextEvent] = []
    run = ordinary_runtime.ProspectiveOrdinaryRun(
        task_name="game24", baseline="reflexion_style", run_id="context-suffix",
        model="replay", client=ReflectionResponses(final_success=True), allow_test_client=True,
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0},
        tasks=(task, task.model_copy(update={"sample_id": "context-suffix-2"})),
        writer_callbacks=RuntimeWriterCallbacks(
            on_context=lambda event: events.append(ContextEvent.model_validate(event)),
        ),
    )

    result = ordinary_runtime.execute_prospective_ordinary(run)

    first, second = result.trials
    assert events == [first.context_event, second.context_event]
    assert first.state_before is not None and first.state_before.entries == ()
    assert first.state_after is not None and len(first.state_after.entries) == 1
    assert first.state_after == second.state_before
    assert second.state_after is not None and len(second.state_after.entries) > 1
    for event in events:
        assert ContextEvent.model_validate_json(event.model_dump_json()) == event
