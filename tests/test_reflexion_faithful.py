from __future__ import annotations

from memcontam.baselines.reflexion_style import ReflexionStylePolicy
from memcontam.clients.replay import ReplayClient
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample_001",
        task_name="math_equation_balancer",
        input={"input": "2 + 2"},
        verifier_spec={"gold": "TOP_SECRET_GOLD"},
    )


def _config() -> dict[str, str]:
    return {
        "run_id": "run_001",
        "baseline": "reflexion_style",
        "arm": "clean",
        "model": "replay",
    }


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
        config=_config(),
        verifier=verify,
    )

    assert result["final_response"] == " final: 4 "
    assert result["parsed_answer"] == "4"
    assert result["verifier_result"].is_correct is True
    assert [call.stage for call in result["method_calls"]] == ["reflexion_generate"]
    assert result["memory_after"] == result["memory_before"]
    assert result["memory_write_event"] is None
    assert result["retrieved_records"] == []
    assert result["retrieved_memory"] == []
    assert result["retrieved_scores"] == []
    actor_prompt = result["method_calls"][0].messages[1]["content"]
    assert "Reflection: middle\nReflection: newer\nReflection: latest" in actor_prompt
    assert "oldest" not in actor_prompt
    assert "seed corpus instruction" not in actor_prompt


def test_run_failure_reflects_once_and_appends_lineaged_reflection() -> None:
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
                "reflexion_generate": "final: incorrect",
                "reflexion_reflect": "  Re-check operator precedence.  ",
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
        config=_config(),
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
    assert appended.metadata["source_entry_ids"] == ["two"]
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


def test_run_rejects_empty_failure_reflection_without_mutating_memory() -> None:
    task = _task()
    memory = MemoryState(
        entries=[MemoryEntry(entry_id="one", content="one", memory_type="verbal_reflection")]
    )
    client = ReplayClient(
        responses_by_sample={
            "sample_001": {
                "reflexion_generate": "final: incorrect",
                "reflexion_reflect": "  \n\t",
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
        config=_config(),
        verifier=verify,
    )

    assert [call.stage for call in result["method_calls"]] == [
        "reflexion_generate",
        "reflexion_reflect",
    ]
    assert result["memory_after"] == result["memory_before"]
    assert result["retrieved_records"] == []
    assert result["retrieved_scores"] == []
    assert result["memory_write_event"] == {
        "type": "reflexion_append",
        "status": "rejected_empty",
        "source_trial_id": "run_001:math_equation_balancer:sample_001:reflexion_style:clean:replay",
        "parent_entry_ids": ["one"],
        "source_entry_ids": [],
    }
