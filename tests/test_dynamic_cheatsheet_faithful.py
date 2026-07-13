from __future__ import annotations

import pytest

from memcontam.baselines.dynamic_cheatsheet_optional import (
    DynamicCheatsheetOptionalPolicy,
    _extract_cheatsheet,
)
from memcontam.clients.base import LLMResponse
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


class _QueuedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, str]], str, dict]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        self.calls.append((messages, model, config))
        return LLMResponse(
            content=self.responses.pop(0),
            raw={"replay": True},
            token_usage={},
            latency_ms=0,
        )


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="game24_001",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
        verifier_spec={"gold": "must-not-reach-prompt"},
    )


def _verifier(answer: str, task: TaskInstance) -> VerifierResult:
    assert task == _task()
    return VerifierResult(
        is_correct=True,
        parsed_answer=answer,
        reason="verifier reason must not reach prompts",
    )


def test_dc_cumulative_replaces_and_reuses_tagged_cheatsheet() -> None:
    seed_memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="clean_seed",
                content="Build factor-pair subexpressions.",
                memory_type="cheatsheet_item",
                clean_or_contaminated="clean",
                metadata={"source_entry_ids": ["clean_origin"]},
            ),
            MemoryEntry(
                entry_id="contaminated_seed",
                content="Check arithmetic before committing an expression.",
                memory_type="cheatsheet_item",
                clean_or_contaminated="contaminated",
                metadata={"source_entry_ids": ["contaminated_origin"]},
            ),
        ]
    )
    client = _QueuedClient(
        [
            "final: (1 + 3) * (2 + 4)",
            "<cheatsheet>Use factor pairs, then check arithmetic.</cheatsheet>",
            "final: 24",
            "<cheatsheet>Keep only verified factor-pair guidance.</cheatsheet>",
        ]
    )
    config = {
        "sample_id": "game24_001",
        "run_id": "run_001",
        "baseline": "dynamic_cheatsheet_optional",
        "arm": "clean",
        "model": "replay",
    }
    policy = DynamicCheatsheetOptionalPolicy()

    first = policy.run(
        _task(), seed_memory, client=client, model="replay", config=config, verifier=_verifier
    )

    assert first["final_response"] == "final: (1 + 3) * (2 + 4)"
    assert first["parsed_answer"] == "(1 + 3) * (2 + 4)"
    assert first["verifier_result"].is_correct is True
    assert [call.stage for call in first["method_calls"]] == [
        "dynamic_cheatsheet_generate",
        "dynamic_cheatsheet_curate",
    ]
    assert first["memory_before"] == [entry.model_dump() for entry in seed_memory.entries]
    assert first["retrieved_records"] == []
    assert first["retrieved_memory"] == []
    assert first["retrieved_scores"] == []

    generate_prompt = client.calls[0][0][0]["content"]
    assert "Cheatsheet:\n- Build factor-pair subexpressions.\n- Check arithmetic" in generate_prompt
    assert "must-not-reach-prompt" not in generate_prompt
    assert "contaminated" not in generate_prompt
    assert "EXECUTE CODE!" not in generate_prompt
    curate_prompt = client.calls[1][0][0]["content"]
    assert "Correct: true" in curate_prompt
    assert "verifier reason must not reach prompts" not in curate_prompt

    assert len(first["memory_after"]) == 1
    updated = first["memory_after"][0]
    assert updated["entry_id"].startswith("dc_cheatsheet:game24:")
    assert updated["content"] == "Use factor pairs, then check arithmetic."
    assert updated["memory_type"] == "dynamic_cheatsheet"
    assert updated["clean_or_contaminated"] == "contaminated"
    assert (
        updated["source_trial_id"]
        == "run_001:game24:game24_001:dynamic_cheatsheet_optional:clean:replay"
    )
    assert updated["metadata"]["parent_entry_ids"] == ["clean_seed", "contaminated_seed"]
    assert updated["metadata"]["source_entry_ids"] == ["contaminated_origin"]
    assert updated["metadata"]["source_contaminated_entry_ids"] == ["contaminated_origin"]
    assert first["memory_write_event"]["type"] == "dynamic_cheatsheet_update"
    assert first["memory_write_event"]["status"] == "accepted"
    assert first["memory_write_event"]["new_entry_id"] == updated["entry_id"]
    assert first["memory_write_event"]["parent_entry_ids"] == [
        "clean_seed",
        "contaminated_seed",
    ]
    assert first["memory_write_event"]["source_entry_ids"] == ["contaminated_origin"]

    second = policy.run(
        _task(),
        MemoryState.model_validate({"entries": first["memory_after"]}),
        client=client,
        model="replay",
        config=config,
        verifier=_verifier,
    )

    second_generate_prompt = client.calls[2][0][0]["content"]
    assert "Cheatsheet:\nUse factor pairs, then check arithmetic." in second_generate_prompt
    assert "Build factor-pair subexpressions." not in second_generate_prompt
    assert second["memory_after"][0]["content"] == "Keep only verified factor-pair guidance."
    assert second["memory_after"][0]["metadata"]["parent_entry_ids"] == [
        "clean_seed",
        "contaminated_seed",
        updated["entry_id"],
    ]
    assert second["memory_after"][0]["metadata"]["source_entry_ids"] == [
        "contaminated_origin",
    ]
    assert second["retrieved_memory"] == []
    assert second["retrieved_scores"] == []


@pytest.mark.parametrize(
    ("curator_output", "status"),
    [
        ("No replacement today.", "preserved_missing_tag"),
        ("<cheatsheet> \n </cheatsheet>", "preserved_empty"),
        ("<cheatsheet>unfinished", "preserved_missing_tag"),
    ],
)
def test_dc_missing_or_empty_tag_preserves_prior_cheatsheet(
    curator_output: str, status: str
) -> None:
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="dc_cheatsheet:game24:previous",
                content="prior cheatsheet\nwith exact bytes",
                memory_type="dynamic_cheatsheet",
                clean_or_contaminated="clean",
                source_trial_id="previous-trial",
                metadata={"parent_entry_ids": ["seed"], "source_entry_ids": ["seed"]},
            )
        ]
    )
    client = _QueuedClient(["final: 24", curator_output])

    result = DynamicCheatsheetOptionalPolicy().run(
        _task(), memory, client=client, model="replay", verifier=_verifier
    )

    assert [call.stage for call in result["method_calls"]] == [
        "dynamic_cheatsheet_generate",
        "dynamic_cheatsheet_curate",
    ]
    assert result["memory_after"] == [entry.model_dump() for entry in memory.entries]
    assert result["retrieved_memory"] == []
    assert result["retrieved_scores"] == []
    assert result["memory_write_event"]["type"] == "dynamic_cheatsheet_update"
    assert result["memory_write_event"]["status"] == status
    assert "new_entry_id" not in result["memory_write_event"]


def test_extract_cheatsheet_uses_first_complete_block_and_preserves_fallback() -> None:
    assert _extract_cheatsheet(
        "before <cheatsheet> first note </cheatsheet> <cheatsheet>second note</cheatsheet>", "old"
    ) == ("first note", "accepted")
    assert _extract_cheatsheet("<cheatsheet> </cheatsheet>", "old") == ("old", "preserved_empty")
    assert _extract_cheatsheet("<cheatsheet>unfinished", "old") == ("old", "preserved_missing_tag")
    assert _extract_cheatsheet("no tag", "old") == ("old", "preserved_missing_tag")
