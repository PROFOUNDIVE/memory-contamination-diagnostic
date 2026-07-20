from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest

from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
from memcontam.baselines.reflexion_style import ReflexionStylePolicy
from memcontam.clients.replay import ReplayClient
from memcontam.logging.provenance import compute_exposure_from_spans, normalize_memory_event
from memcontam.logging.schema import MemoryEvent, MemoryItemLog, VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


def test_reflexion_contract_requires_structured_generation_and_authenticated_attempts() -> None:
    module = __import__("memcontam.baselines.reflexion_style", fromlist=["*"])
    fixture_dir = Path(__file__).parent / "fixtures/prompts"
    generation_fixture = json.loads((fixture_dir / "reflexion_generate.json").read_text(encoding="utf-8"))
    reflection_fixture = json.loads((fixture_dir / "reflexion_reflect.json").read_text(encoding="utf-8"))

    assert getattr(module, "ReflectionGenerationResult", None) is not None
    assert callable(getattr(module, "apply_keep_last_3", None))
    assert generation_fixture["stage"] == "reflexion_generate"
    assert reflection_fixture == {
        "stage": "reflexion_reflect",
        "schema": "ReflectionGenerationResult",
        "strict_json": True,
    }


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample_001",
        task_name="math_equation_balancer",
        input={"input": "2 + 2"},
        verifier_spec={"gold": "TOP_SECRET_GOLD"},
    )


def _config(*, max_attempts: int | None = None) -> dict[str, object]:
    config: dict[str, object] = {
        "run_id": "run_001",
        "baseline": "reflexion_style",
        "arm": "clean",
        "model": "replay",
    }
    if max_attempts is not None:
        config["max_attempts"] = max_attempts
    return config


def _reflection(text: str, used_ids: tuple[str, ...] = ()) -> str:
    return json.dumps(
        {
            "mode": "corrective",
            "failure_class": "incorrect_answer",
            "reflection_text": text,
            "explicitly_used_memory_ids": list(used_ids),
        }
    )


def _adapter() -> ReflexionAdapter:
    adapter = getattr(import_module("memcontam.baselines.reflexion_style"), "ReflexionAdapter", None)
    assert adapter is not None
    return adapter()


def _state() -> ReflexionState:
    state = getattr(import_module("memcontam.baselines.reflexion_style"), "ReflexionState", None)
    assert state is not None
    return state()


def test_adapter_stores_terminal_reflection_after_two_authenticated_incorrect_attempts() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/replay/baseline_fidelity_v1/reflexion_style_failure_failure.json"
        ).read_text(encoding="utf-8")
    )
    state = _state()
    outcome = _adapter().execute(
        _task(),
        state,
        client=ReplayClient(responses_by_sample={"sample_001": fixture["stages"]}),
        model="replay",
        config=_config(max_attempts=2),
        verifier=lambda answer, _task: answer == "24",
    )

    assert outcome.status == "succeeded"
    assert outcome.verifier_result is False
    assert outcome.answer_call_id == outcome.method_calls[2].call_id
    assert outcome.error_type is None
    assert outcome.failure_disposition is None
    assert outcome.scientific_ineligibility_reason is None
    assert [call.stage for call in outcome.method_calls] == [
        "reflexion_generate",
        "reflexion_reflect",
        "reflexion_generate",
        "reflexion_reflect",
    ]
    assert [call.retry_count for call in outcome.method_calls] == [0, 0, 0, 0]
    assert [attempt["attempt_index"] for attempt in outcome.metadata["reflexion_attempt_outcomes"]] == [
        1,
        2,
    ]
    assert [attempt["failure_class"] for attempt in outcome.metadata["reflexion_attempt_outcomes"]] == [
        "incorrect_answer",
        "incorrect_answer",
    ]
    assert len(outcome.metadata["reflexion_reflection_events"]) == 2
    assert [entry.content for entry in state.reflections] == [
        "Reflection: Retry carefully.",
        "Reflection: Terminal reflection.",
    ]


def test_adapter_keeps_transport_retries_out_of_semantic_attempt_metadata() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/replay/baseline_fidelity_v1/reflexion_attempt2_transport_retry.json"
        ).read_text(encoding="utf-8")
    )
    outcome = _adapter().execute(
        _task(),
        _state(),
        client=ReplayClient(
            responses_by_sample={
                "sample_001": {
                    **fixture["stages"],
                    "reflexion_reflect": _reflection("Retry carefully."),
                }
            }
        ),
        model="replay",
        config={**_config(max_attempts=2), "retry_count": fixture["transport_retry_count"]},
        verifier=lambda answer, _task: answer == "24",
    )

    assert [call.retry_count for call in outcome.method_calls] == [3, 3, 3]
    assert [attempt["attempt_index"] for attempt in outcome.metadata["reflexion_attempt_outcomes"]] == [
        1,
        2,
    ]


def test_adapter_stops_before_authentication_on_malformed_generation() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/replay/baseline_fidelity_v1/reflexion_invalid_generation.json"
        ).read_text(encoding="utf-8")
    )
    state = _state()
    outcome = _adapter().execute(
        _task(),
        state,
        client=ReplayClient(responses_by_sample={"sample_001": {fixture["stage"]: fixture["response"]}}),
        model="replay",
        config=_config(max_attempts=2),
        verifier=lambda answer, _task: False,
    )

    assert outcome.status == "failed"
    assert outcome.failure_disposition == "reflexion_invalid_generation"
    assert outcome.answer_call_id == outcome.method_calls[0].call_id
    assert [call.stage for call in outcome.method_calls] == ["reflexion_generate"]
    assert outcome.metadata["reflexion_attempt_outcomes"] == []
    assert outcome.metadata["reflexion_reflection_events"] == []
    assert state.reflections == []


def test_adapter_stops_before_authentication_when_the_verifier_contract_fails() -> None:
    state = _state()
    outcome = _adapter().execute(
        _task(),
        state,
        client=ReplayClient(responses=["final: wrong"]),
        model="replay",
        config=_config(max_attempts=2),
        verifier=lambda answer, _task: cast(VerifierResult | bool, "not-a-boolean"),
    )

    assert outcome.status == "failed"
    assert outcome.failure_disposition == "verifier_contract_failed"
    assert [call.stage for call in outcome.method_calls] == ["reflexion_generate"]
    assert outcome.metadata["reflexion_attempt_outcomes"] == []
    assert outcome.metadata["reflexion_reflection_events"] == []
    assert state.reflections == []


def test_adapter_rejects_malformed_reflection_without_a_write_or_second_attempt() -> None:
    state = _state()
    outcome = _adapter().execute(
        _task(),
        state,
        client=ReplayClient(
            responses_by_sample={
                "sample_001": {
                    "reflexion_generate": ["final: wrong", "final: 4"],
                    "reflexion_reflect": "not-json",
                }
            }
        ),
        model="replay",
        config=_config(max_attempts=2),
        verifier=lambda answer, _task: False,
    )

    assert outcome.status == "failed"
    assert outcome.failure_disposition == "reflexion_invalid_reflection"
    assert [call.stage for call in outcome.method_calls] == [
        "reflexion_generate",
        "reflexion_reflect",
    ]
    assert len(outcome.metadata["reflexion_attempt_outcomes"]) == 1
    assert outcome.metadata["reflexion_reflection_events"] == []
    assert outcome.memory_write_event is None
    assert state.reflections == []


def test_adapter_preserves_an_authenticated_reflection_when_retry_generation_is_malformed() -> None:
    state = _state()
    outcome = _adapter().execute(
        _task(),
        state,
        client=ReplayClient(
            responses_by_sample={
                "sample_001": {
                    "reflexion_generate": ["final: wrong", "not a final answer"],
                    "reflexion_reflect": _reflection("Retry carefully."),
                }
            }
        ),
        model="replay",
        config=_config(max_attempts=2),
        verifier=lambda answer, _task: False,
    )

    assert outcome.status == "failed"
    assert outcome.failure_disposition == "reflexion_invalid_generation"
    assert len(outcome.metadata["reflexion_attempt_outcomes"]) == 1
    assert len(outcome.metadata["reflexion_reflection_events"]) == 1
    assert outcome.memory_write_event is not None
    assert outcome.memory_write_event["status"] == "accepted"
    assert [entry.content for entry in state.reflections] == ["Reflection: Retry carefully."]


def test_adapter_keeps_only_last_three_reflections_and_uses_explicit_parents() -> None:
    assert getattr(import_module("memcontam.baselines.reflexion_style"), "ReflexionState", None) is ReflexionState
    state = ReflexionState(
        reflections=[
            MemoryEntry(entry_id="one", content="Reflection: one", memory_type="verbal_reflection"),
            MemoryEntry(entry_id="two", content="Reflection: two", memory_type="verbal_reflection"),
            MemoryEntry(entry_id="three", content="Reflection: three", memory_type="verbal_reflection"),
            MemoryEntry(entry_id="four", content="Reflection: four", memory_type="verbal_reflection"),
        ]
    )
    assert [entry.entry_id for entry in state.reflections] == ["two", "three", "four"]
    outcome = _adapter().execute(
        _task(),
        state,
        client=ReplayClient(
            responses_by_sample={
                "sample_001": {
                    "reflexion_generate": ["final: wrong", "final: 4"],
                    "reflexion_reflect": _reflection("five", ("two",)),
                }
            }
        ),
        model="replay",
        config=_config(max_attempts=2),
        verifier=lambda answer, _task: answer == "4",
    )

    assert [entry.entry_id for entry in state.reflections] == ["three", "four", state.reflections[-1].entry_id]
    assert state.reflections[-1].metadata["direct_parent_ids"] == ["two"]
    assert state.reflections[-1].metadata["memory_support_ids"] == ["two"]
    assert state.reflections[-1].metadata["source_entry_ids"] == ["two"]
    assert outcome.memory_write_event is not None
    assert outcome.memory_write_event["parent_entry_ids"] == ["two"]


def test_build_prompt_keeps_legacy_last_three_entries_behavior() -> None:
    memory = MemoryState(
        entries=[
            MemoryEntry(entry_id="one", content="one", memory_type="seed"),
            MemoryEntry(entry_id="two", content="two", memory_type="seed"),
            MemoryEntry(entry_id="three", content="three", memory_type="seed"),
            MemoryEntry(entry_id="four", content="four", memory_type="seed"),
        ]
    )

    assert ReflexionStylePolicy().build_prompt(_task(), memory) == [
        {"role": "user", "content": "Reflections:\ntwo\nthree\nfour\n\nSolve: {'input': '2 + 2'}"}
    ]


def test_run_success_generates_once_without_writing_and_uses_last_three_reflections() -> None:
    task = _task()
    memory = MemoryState(
        entries=[
            MemoryEntry(entry_id="seed", content="seed corpus instruction", memory_type="seed"),
            MemoryEntry(
                entry_id="old", content="Reflection: oldest", memory_type="verbal_reflection"
            ),
            MemoryEntry(entry_id="middle", content="middle", memory_type="verbal_reflection"),
            MemoryEntry(
                entry_id="newer", content="Reflection: newer", memory_type="verbal_reflection"
            ),
            MemoryEntry(entry_id="latest", content="latest", memory_type="verbal_reflection"),
        ]
    )
    client = ReplayClient(
        responses_by_sample={"sample_001": {"reflexion_generate": " final: 4 "}}
    )

    def verify(*args: object) -> VerifierResult:
        assert args == ("4", task)
        return VerifierResult(is_correct=True, parsed_answer="4")

    result = ReflexionStylePolicy().run(
        task,
        memory,
        client=client,
        model="replay",
        config=_config(max_attempts=2),
        verifier=verify,
    )

    assert result["final_response"] == " final: 4 "
    assert result["parsed_answer"] == "4"
    assert result["verifier_result"].is_correct is True
    assert [call.stage for call in result["method_calls"]] == ["reflexion_generate"]
    assert [entry["entry_id"] for entry in result["memory_after"]] == [
        "seed",
        "middle",
        "newer",
        "latest",
    ]
    assert result["memory_write_event"] is None
    assert result["retrieved_records"] == []
    assert result["retrieved_memory"] == []
    assert result["retrieved_scores"] == []
    assert result["answer_call_id"] == result["method_calls"][0].call_id
    answer_call = result["method_calls"][0]
    assert [
        answer_call.messages[1]["content"][span.start : span.end]
        for span in answer_call.source_spans
    ] == ["Reflection: middle", "Reflection: newer", "Reflection: latest"]
    actor_prompt = result["method_calls"][0].messages[1]["content"]
    assert "Reflection: middle\nReflection: newer\nReflection: latest" in actor_prompt
    assert "oldest" not in actor_prompt
    assert "seed corpus instruction" not in actor_prompt


def test_run_max_attempts_one_preserves_reflect_then_return_without_retry() -> None:
    task = _task()
    memory = MemoryState(
        entries=[
            MemoryEntry(entry_id="seed", content="seed corpus instruction", memory_type="seed"),
            MemoryEntry(entry_id="one", content="one", memory_type="verbal_reflection"),
            MemoryEntry(
                entry_id="two",
                content="Reflection: two",
                memory_type="verbal_reflection",
                clean_or_contaminated="contaminated",
            ),
            MemoryEntry(entry_id="three", content="three", memory_type="verbal_reflection"),
        ]
    )
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": ["final: incorrect"],
                    "reflexion_reflect": _reflection(
                        "Re-check operator precedence.", ("one", "two", "three")
                    ),
            }
        }
    )

    def verify(*args: object) -> VerifierResult:
        assert args == ("incorrect", task)
        return VerifierResult(
            is_correct=False,
            parsed_answer="incorrect",
            reason="TOP_SECRET_REASON",
        )

    result = ReflexionStylePolicy().run(
        task,
        memory,
        client=client,
        model="replay",
        config=_config(max_attempts=1),
        verifier=verify,
    )

    assert [call.stage for call in result["method_calls"]] == [
        "reflexion_generate",
        "reflexion_reflect",
    ]
    assert result["verifier_result"].is_correct is False
    assert result["parsed_answer"] == "incorrect"
    assert result["retrieved_records"] == []
    assert result["retrieved_scores"] == []
    appended = memory.entries[-1]
    assert appended.entry_id.startswith("reflexion:math_equation_balancer:sample_001:")
    assert appended.content == "Reflection: Re-check operator precedence."
    assert appended.memory_type == "verbal_reflection"
    assert appended.clean_or_contaminated == "contaminated"
    assert appended.source_trial_id == "run_001:math_equation_balancer:sample_001:reflexion_style:clean:replay"
    assert appended.metadata["parent_entry_ids"] == ["one", "two", "three"]
    assert appended.metadata["source_entry_ids"] == ["one", "two", "three"]
    assert appended.metadata["reflection_lineage"]["stage"] == "reflexion_reflect"
    assert result["memory_after"][-1] == appended.model_dump()
    assert result["memory_write_event"]["type"] == "reflexion_append"
    assert result["memory_write_event"]["status"] == "accepted"
    assert result["memory_write_event"]["new_entry_id"] == appended.entry_id
    prompts = "\n".join(
        message["content"] for call in result["method_calls"] for message in call.messages
    )
    assert "Correct: false" in prompts
    assert "TOP_SECRET_GOLD" not in prompts
    assert "TOP_SECRET_REASON" not in prompts


def test_run_max_attempts_two_retries_same_sample_with_latest_three_memory_only() -> None:
    task = _task()
    memory = MemoryState(
        entries=[
            MemoryEntry(entry_id="old", content="old", memory_type="verbal_reflection"),
            MemoryEntry(entry_id="one", content="one", memory_type="verbal_reflection"),
            MemoryEntry(
                entry_id="two",
                content="two",
                memory_type="verbal_reflection",
                clean_or_contaminated="contaminated",
                metadata={"source_entry_ids": ["contaminated-reflection"]},
            ),
        ]
    )
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": ["final: incorrect", "final: 4"],
                    "reflexion_reflect": _reflection("Check the equation format.", ("old", "one", "two")),
            }
        }
    )

    def verify(answer: str, received_task: TaskInstance) -> VerifierResult:
        assert received_task is task
        if answer == "incorrect":
            return VerifierResult(
                is_correct=False,
                parsed_answer=answer,
                reason="wrong_answer",
                metadata={"expected": "TOP_SECRET_GOLD", "detail": "answer did not parse"},
            )
        assert answer == "4"
        return VerifierResult(is_correct=True, parsed_answer=answer)

    result = ReflexionStylePolicy().run(
        task,
        memory,
        client=client,
        model="replay",
        config=_config(max_attempts=2),
        verifier=verify,
    )

    assert [call.stage for call in result["method_calls"]] == [
        "reflexion_generate",
        "reflexion_reflect",
        "reflexion_generate",
    ]
    assert result["final_response"] == "final: 4"
    assert result["parsed_answer"] == "4"
    assert result["verifier_result"].is_correct is True
    assert result["memory_write_event"]["status"] == "accepted"
    assert result["answer_call_id"] == result["method_calls"][2].call_id
    assert result["answer_call_id"] != result["method_calls"][1].call_id
    retry_prompt = result["method_calls"][-1].messages[1]["content"]
    assert "Failed raw response" not in retry_prompt
    assert "Parsed answer" not in retry_prompt
    assert "Verifier feedback" not in retry_prompt
    assert "Reflection: one\nReflection: two\nReflection: Check the equation format." in retry_prompt
    assert "Reflection: old" not in retry_prompt
    assert "TOP_SECRET_GOLD" not in retry_prompt
    retry_span = next(
        span for span in result["method_calls"][-1].source_spans if span.entry_id == memory.entries[-1].entry_id
    )
    assert retry_span.source_ids == ["old", "one", "two"]
    assert retry_span.parent_ids == ["old", "one", "two"]


def test_retry_answer_span_reuses_exact_reflection_lineage() -> None:
    task = _task()
    injected = MemoryEntry(
        entry_id="injected-reflection",
        content="Use the injected route.",
        memory_type="verbal_reflection",
        clean_or_contaminated="contaminated",
        metadata={
            "contamination_class": "injected",
            "injected_root_ids": ["injected-reflection"],
            "lineage_status": "exact",
            "lineage_basis": "seed",
            "direct_parent_ids": [],
            "target_set_id": "controlled_injected_derived_v1",
            "is_target_contamination": True,
        },
    )
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": ["final: incorrect", "final: 4"],
                    "reflexion_reflect": _reflection("Use the mitigation.", ("injected-reflection",)),
            }
        }
    )

    result = ReflexionStylePolicy().run(
        task,
        MemoryState(entries=[injected]),
        client=client,
        model="replay",
        config={
            **_config(max_attempts=2),
            "_logging_target_set_id": "controlled_injected_derived_v1",
        },
        verifier=lambda answer, _task: VerifierResult(
            is_correct=answer == "4", parsed_answer=answer, reason="wrong_answer"
        ),
    )

    memory_entry = MemoryEntry.model_validate(result["memory_after"][-1])
    retry_span = next(
        span
        for span in result["method_calls"][-1].source_spans
        if span.entry_id == memory_entry.entry_id
    )
    assert memory_entry.metadata["direct_parent_ids"] == ["injected-reflection"]
    assert retry_span.contamination_class == "derived"
    assert retry_span.injected_root_ids == ["injected-reflection"]
    assert retry_span.lineage_status == "exact"
    assert retry_span.lineage_basis == "recorded_parent"
    assert retry_span.direct_parent_ids == ["injected-reflection"]
    assert retry_span.target_set_id == "controlled_injected_derived_v1"
    assert retry_span.is_target_contamination is True


def test_reflection_only_auxiliary_call_does_not_set_answer_exposure() -> None:
    task = _task()
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": "final: incorrect",
                "reflexion_reflect": "Ignore the contaminated failed trajectory.",
            }
        }
    )
    result = ReflexionStylePolicy().run(
        task,
        MemoryState(),
        client=client,
        model="replay",
        config=_config(max_attempts=1),
        verifier=lambda answer, received_task: VerifierResult(
            is_correct=False, parsed_answer=answer, reason="wrong_answer"
        ),
    )

    answer_call = result["method_calls"][0]
    assert result["answer_call_id"] == answer_call.call_id
    assert result["answer_call_id"] != result["method_calls"][1].call_id
    exposure = compute_exposure_from_spans(
        result["answer_call_id"], answer_call.source_spans, "contaminated"
    )
    assert exposure.is_exposed is False
    assert exposure.exposure_mode == "not_in_final_prompt"


def test_run_stops_after_failed_retry_without_second_reflection() -> None:
    task = _task()
    memory = MemoryState()
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": ["final: first", "final: second"],
                "reflexion_reflect": [
                    _reflection("Try a different operation."),
                    _reflection("Terminal reflection."),
                ],
            }
        }
    )

    def verify(answer: str, received_task: TaskInstance) -> VerifierResult:
        assert received_task is task
        return VerifierResult(is_correct=False, parsed_answer=answer, reason="wrong_answer")

    result = ReflexionStylePolicy().run(
        task,
        memory,
        client=client,
        model="replay",
        config=_config(max_attempts=2),
        verifier=verify,
    )

    assert [call.stage for call in result["method_calls"]] == [
        "reflexion_generate",
        "reflexion_reflect",
        "reflexion_generate",
        "reflexion_reflect",
    ]
    assert result["final_response"] == "final: second"
    assert result["parsed_answer"] == "second"
    assert result["verifier_result"].is_correct is False
    assert memory.entries[-1].content == "Reflection: Terminal reflection."


def test_run_rejects_empty_failure_reflection_without_mutating_memory() -> None:
    task = _task()
    memory = MemoryState(
        entries=[MemoryEntry(entry_id="one", content="one", memory_type="verbal_reflection")]
    )
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": "final: incorrect",
                    "reflexion_reflect": "not-json",
            }
        }
    )

    def verify(*args: object) -> VerifierResult:
        assert args == ("incorrect", task)
        return VerifierResult(is_correct=False, parsed_answer="incorrect")

    result = ReflexionStylePolicy().run(
        task,
        memory,
        client=client,
        model="replay",
        config=_config(max_attempts=2),
        verifier=verify,
    )

    assert [call.stage for call in result["method_calls"]] == [
        "reflexion_generate",
        "reflexion_reflect",
    ]
    assert result["memory_after"] == result["memory_before"]
    assert result["retrieved_records"] == []
    assert result["retrieved_scores"] == []
    assert result["status"] == "failed"
    assert result["failure_disposition"] == "reflexion_invalid_reflection"
    assert result["verifier_result"].is_correct is False
    assert result["memory_write_event"] is None


def test_run_rejects_max_attempts_outside_one_or_two() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ReflexionStylePolicy().run(
            _task(),
            MemoryState(),
            client=ReplayClient(responses=["final: 4"]),
            model="replay",
            config=_config(max_attempts=3),
        )


def _normalize_reflexion_event(
    result: dict[str, object], source_trial_id: str
) -> MemoryEvent | None:
    before = [MemoryEntry.model_validate(entry) for entry in result["memory_before"]]  # type: ignore[index]
    after = [MemoryEntry.model_validate(entry) for entry in result["memory_after"]]  # type: ignore[index]
    return normalize_memory_event(
        "reflexion_style",
        source_trial_id,
        before,
        after,
        result["memory_write_event"],  # type: ignore[index]
    )


def test_accepted_reflection_memory_event_normalizes_append() -> None:
    task = _task()
    memory = MemoryState(
        entries=[
            MemoryEntry(entry_id="seed", content="seed corpus instruction", memory_type="seed"),
            MemoryEntry(
                entry_id="one",
                content="Reflection: one",
                memory_type="verbal_reflection",
                clean_or_contaminated="contaminated",
                metadata={"source_entry_ids": ["cont-1"]},
            ),
        ]
    )
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": "final: incorrect",
                    "reflexion_reflect": _reflection("Check units.", ("one",)),
            }
        }
    )

    def verify(*args: object) -> VerifierResult:
        assert args == ("incorrect", task)
        return VerifierResult(is_correct=False, parsed_answer="incorrect", reason="wrong_units")

    result = ReflexionStylePolicy().run(
        task,
        memory,
        client=client,
        model="replay",
        config=_config(max_attempts=1),
        verifier=verify,
    )

    source_trial_id = "run_001:math_equation_balancer:sample_001:reflexion_style:clean:replay"
    event = _normalize_reflexion_event(result, source_trial_id)

    assert event is not None
    assert event.status == "accepted"
    assert event.operation == "append"
    assert event.before_entry_ids == ["seed", "one"]
    assert event.after_entry_ids == ["seed", "one", result["memory_write_event"]["new_entry_id"]]
    assert event.new_entry_ids == [result["memory_write_event"]["new_entry_id"]]
    assert event.removed_entry_ids == []
    assert event.before_snapshot_hash != event.after_snapshot_hash
    assert event.parent_entry_ids == ["one"]
    assert event.source_entry_ids == ["one"]
    assert event.contaminated_source_ids == ["cont-1"]
    assert event.creation_origin == "reflexion_reflect"


def test_rejected_empty_reflection_memory_event_preserves_snapshot() -> None:
    task = _task()
    memory = MemoryState(
        entries=[MemoryEntry(entry_id="one", content="one", memory_type="verbal_reflection")]
    )
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": "final: incorrect",
                    "reflexion_reflect": "not-json",
            }
        }
    )

    def verify(*args: object) -> VerifierResult:
        assert args == ("incorrect", task)
        return VerifierResult(is_correct=False, parsed_answer="incorrect")

    result = ReflexionStylePolicy().run(
        task,
        memory,
        client=client,
        model="replay",
        config=_config(max_attempts=2),
        verifier=verify,
    )

    source_trial_id = "run_001:math_equation_balancer:sample_001:reflexion_style:clean:replay"
    event = _normalize_reflexion_event(result, source_trial_id)

    assert result["status"] == "failed"
    assert result["failure_disposition"] == "reflexion_invalid_reflection"
    assert event is None


def test_clean_ancestry_reflexion_mitigation_is_not_natural() -> None:
    entry = MemoryEntry(
        entry_id="clean-mitigation",
        content="Reflection: check the operator order.",
        memory_type="verbal_reflection",
        source_trial_id="trial-1",
        metadata={
            "parent_entry_ids": ["clean-seed"],
            "memory_error_status": "satisfied",
        },
    )

    item = MemoryItemLog.from_memory_entry(entry)

    assert item.contamination_class == "clean"
    assert item.injected_root_ids == []
