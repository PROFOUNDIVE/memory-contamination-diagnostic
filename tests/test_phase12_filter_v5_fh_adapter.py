from __future__ import annotations

from memcontam.baselines.full_history import FullHistoryPayload, FullHistoryPolicy, render_full_history
from memcontam.baselines.full_history import FullHistoryState
from memcontam.baselines.full_history_adapter import FullHistoryAdapter
from memcontam.baselines.prompt_budget import count_prompt_tokens
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.filter_challenge.adapters.full_history import (
    FullHistoryPairRequest,
    FullHistoryProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


class _ScriptedClient:
    def __init__(self) -> None:
        self.fake_answer_calls = 0
        self.provider_calls_issued = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model, config
        self.fake_answer_calls += 1
        return LLMResponse(content="final: 24", raw={}, token_usage={}, latency_ms=0)


def test_full_history_default_execution_appends_the_answer_interaction() -> None:
    # Given: ordinary Full History state and a scripted native response.
    state = FullHistoryState()
    task = TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
    )

    # When: the ordinary execution path answers the task.
    outcome = FullHistoryAdapter().execute(task, state, client=_ScriptedClient(), model="replay")

    # Then: legacy default behavior retains the answer interaction.
    assert len(state.records) == 1
    assert outcome.memory_write_event is not None


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
    )


def test_full_history_read_only_hook_prevents_answer_writeback() -> None:
    # Given: an empty active memory state and a native answer response.
    memory = MemoryState()

    # When: the public Full History policy runs with updates disabled.
    result = FullHistoryPolicy().run(
        _task(), memory, client=_ScriptedClient(), model="replay", update_enabled=False
    )

    # Then: the answer is returned without appending an interaction to active memory.
    assert memory.entries == []
    assert result["memory_write_event"] is None


def _entry(entry_id: str, response: str) -> NativeEntry:
    content = render_full_history(entry_id, FullHistoryPayload("1 3 4 6", response))
    return NativeEntry(
        entry_id=entry_id,
        semantic_kind="full_history_transcript",
        schema_version=NATIVE_ENTRY_V1,
        native_component="history",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def _candidate(entry_id: str = "candidate") -> ChallengeCandidate:
    entry = _entry(entry_id, "candidate response")
    return ChallengeCandidate(
        candidate_entry_id=entry.entry_id,
        candidate_native_content=entry.content,
        candidate_native_kind=entry.semantic_kind,
        baseline_family="full_history",
        rag_mode="not_applicable",
        source_checkpoint_id="checkpoint-source",
        source_active_state_hash="source-hash",
        routability={"routability": "challenge_routable_v1", "challenge_suite_key": "suite-1"},
    )


def _checkpoint(entries: tuple[NativeEntry, ...]):
    return serialize_checkpoint(NativeState("fh_bounded", entries, {"records": []}))


def _context_config(history_budget: int) -> dict[str, object]:
    task_tokens = count_prompt_tokens(
        [{"role": "user", "content": f"TASK:\n{canonical_task_json(_task())}"}], "cl100k_base"
    )
    return {
        "mode": "context_bounded_pair_atomic",
        "token_encoding": "cl100k_base",
        "context_window_tokens": task_tokens + history_budget + 1,
        "max_output_tokens": 1,
        "fixed_prompt_overhead_tokens": 0,
        "safety_margin_tokens": 0,
    }


def _request(checkpoint, context_config: dict[str, object]) -> FullHistoryPairRequest:
    return FullHistoryPairRequest(
        task=_task(),
        checkpoint=checkpoint,
        candidate=_candidate(),
        control_client=_ScriptedClient(),
        challenge_client=_ScriptedClient(),
        model="replay",
        context_config=context_config,
    )


def test_provisional_full_history_appends_candidate_through_native_records_when_all_fit() -> None:
    # Given: a frozen full-history checkpoint and enough budget for the provisional record.
    checkpoint = _checkpoint((_entry("history-1", "first"), _entry("history-2", "second")))
    request = _request(checkpoint, _context_config(history_budget=10_000))

    # When: the read-only control and challenge executions run through the adapter.
    result = FullHistoryProvisionalAdapter().execute(request)

    # Then: only the challenge receives the native record and neither execution writes an answer record.
    assert result.control_final_source_ids == ("history-1", "history-2")
    assert result.challenge_final_source_ids == ("history-1", "history-2", "candidate")
    assert result.candidate_exposure.candidate_final_context_inclusion
    assert result.challenge_removed_entry_ids == ()
    assert result.control_outcome.memory_write_event is None
    assert result.challenge_outcome.memory_write_event is None
    assert result.control_outcome.memory_after == result.control_outcome.memory_before
    assert result.challenge_outcome.memory_after == result.challenge_outcome.memory_before
    assert [span.entry_id for span in result.control_outcome.method_calls[0].source_spans] == [
        "history-1",
        "history-2",
    ]
    assert [span.entry_id for span in result.challenge_outcome.method_calls[0].source_spans] == [
        "history-1",
        "history-2",
        "candidate",
    ]
    assert checkpoint.canonical_sha256 == result.source_checkpoint_sha256_after
    assert [entry.entry_id for entry in checkpoint.state.entries] == ["history-1", "history-2"]
    assert request.control_client.fake_answer_calls == request.challenge_client.fake_answer_calls == 1
    assert request.control_client.provider_calls_issued == request.challenge_client.provider_calls_issued == 0


def test_provisional_full_history_logs_fifo_displacement_after_candidate_insertion() -> None:
    # Given: a frozen history whose bounded context can retain only the newest source and candidate.
    oldest = _entry("history-oldest", "oldest")
    newest = _entry("history-newest", "newest")
    candidate = _entry("candidate", "candidate response")
    budget = count_prompt_tokens(
        [{"role": "user", "content": f"{newest.content}\n\n{candidate.content}\n\nTASK:\n{canonical_task_json(_task())}"}],
        "cl100k_base",
    ) - count_prompt_tokens(
        [{"role": "user", "content": f"TASK:\n{canonical_task_json(_task())}"}], "cl100k_base"
    )
    request = _request(_checkpoint((oldest, newest)), _context_config(budget))

    # When: the challenge appends the candidate at the native chronological position.
    result = FullHistoryProvisionalAdapter().execute(request)

    # Then: native FIFO removes only the oldest source record and the candidate remains exposed.
    assert result.challenge_removed_entry_ids == ("history-oldest",)
    assert result.challenge_final_source_ids == ("history-newest", "candidate")
    assert result.candidate_exposure.candidate_final_context_source_ids == (
        "history-newest",
        "candidate",
    )
    assert [span.entry_id for span in result.challenge_outcome.method_calls[0].source_spans] == [
        "history-newest",
        "candidate",
    ]


def test_provisional_full_history_marks_candidate_unexposed_when_budget_removes_every_record() -> None:
    # Given: a frozen history and a budget that cannot retain even the candidate record.
    request = _request(_checkpoint((_entry("history-1", "first"),)), _context_config(1))

    # When: the challenge runs through the unchanged bounded renderer.
    result = FullHistoryProvisionalAdapter().execute(request)

    # Then: exposure is derived only from the final source IDs, never provisional storage.
    assert result.challenge_final_source_ids == ()
    assert result.challenge_removed_entry_ids == ("history-1", "candidate")
    assert not result.candidate_exposure.candidate_final_context_inclusion
    assert result.candidate_exposure.candidate_final_context_source_ids == ()
    assert result.challenge_outcome.method_calls[0].source_spans == []
