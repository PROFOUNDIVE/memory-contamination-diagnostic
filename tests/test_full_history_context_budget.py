from __future__ import annotations

import pytest

from memcontam.baselines.full_history import (
    FullHistoryPayload,
    FullHistoryState,
    render_full_history,
)
from memcontam.baselines.full_history_adapter import FullHistoryAdapter
from memcontam.baselines.full_history_context import render_context_bounded_history
from memcontam.baselines.prompt_budget import count_prompt_tokens
from memcontam.clients.base import LLMResponse
from memcontam.config.resolution import _redact, _validate_full_history_budget
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


def _record(entry_id: str, payload: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=render_full_history(entry_id, FullHistoryPayload(payload, f"final: {payload}")),
        memory_type="full_history_transcript",
        clean_or_contaminated="clean",
    )


def _task() -> TaskInstance:
    return TaskInstance(sample_id="sample-1", task_name="game24", input={"numbers": [1, 3, 4, 6]})


def _config(context_window_tokens: int) -> dict[str, object]:
    return {
        "mode": "context_bounded_pair_atomic",
        "token_encoding": "cl100k_base",
        "context_window_tokens": context_window_tokens,
        "max_output_tokens": 1,
        "fixed_prompt_overhead_tokens": 0,
        "safety_margin_tokens": 0,
    }


def _prompt_tokens(records: list[MemoryEntry]) -> int:
    history = "\n\n".join(record.content for record in records)
    prefix = f"{history}\n\n" if history else ""
    return count_prompt_tokens(
        [{"role": "user", "content": f"{prefix}TASK:\n{canonical_task_json(_task())}"}],
        "cl100k_base",
    )


def test_context_budget_keeps_every_complete_pair_when_all_fit() -> None:
    records = [_record("history-1", "first"), _record("history-2", "second")]
    decision = render_context_bounded_history(
        _task(), records, _config(_prompt_tokens(records) + 1)
    )

    assert [record.entry_id for record in decision.records] == ["history-1", "history-2"]
    assert decision.pre_record_ids == decision.post_record_ids == ["history-1", "history-2"]
    assert decision.removed_record_ids == []
    assert decision.post_token_count == _prompt_tokens(records)


def test_context_budget_drops_oldest_whole_pair_and_never_clips_text() -> None:
    oldest = _record("history-1", "oldest-pair-must-not-appear")
    newest = _record("history-2", "newest-pair-must-remain-complete")
    decision = render_context_bounded_history(
        _task(), [oldest, newest], _config(_prompt_tokens([newest]) + 1)
    )

    assert [record.entry_id for record in decision.records] == ["history-2"]
    assert decision.pre_record_ids == ["history-1", "history-2"]
    assert decision.post_record_ids == ["history-2"]
    assert decision.removed_record_ids == ["history-1"]
    assert oldest.content not in decision.messages[0]["content"]
    assert newest.content in decision.messages[0]["content"]
    assert decision.post_token_count == _prompt_tokens([newest])


def test_adapter_retains_all_pairs_but_records_only_visible_sources_on_verifier_failure() -> None:
    oldest = _record("history-1", "oldest-hidden-pair")
    newest = _record("history-2", "newest-visible-pair")

    class Client:
        def chat(
            self, messages: list[dict[str, str]], model: str, config: dict[str, object]
        ) -> LLMResponse:
            del messages, model, config
            return LLMResponse(content="final: 24", raw={}, token_usage={}, latency_ms=0)

    outcome = FullHistoryAdapter().execute(
        _task(),
        FullHistoryState(records=[oldest, newest]),
        client=Client(),
        model="replay",
        config=_config(_prompt_tokens([newest]) + 1),
        verifier=lambda answer, task: (_ for _ in ()).throw(RuntimeError("verifier unavailable")),
    )

    assert outcome.status == "failed"
    assert [span.entry_id for span in outcome.method_calls[0].source_spans] == ["history-2"]
    assert len(outcome.memory_after) == 3
    assert outcome.memory_after[-1]["metadata"]["source_entry_ids"] == ["history-2"]
    assert outcome.memory_write_event is not None
    assert outcome.memory_write_event["source_entry_ids"] == ["history-2"]
    assert outcome.metadata["full_history_context"]["pre_record_ids"] == ["history-1", "history-2"]
    assert outcome.metadata["full_history_context"]["post_record_ids"] == ["history-2"]
    assert outcome.metadata["full_history_context"]["removed_record_ids"] == ["history-1"]


def test_context_budget_fields_are_not_redacted_as_provider_secrets() -> None:
    budget = _config(123)

    assert _redact({"full_history": budget}) == {"full_history": budget}


@pytest.mark.parametrize("missing_key", ["mode", "token_encoding"])
def test_context_bounded_full_history_requires_its_explicit_mode_and_encoding(
    missing_key: str,
) -> None:
    budget = _config(123)
    del budget[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        _validate_full_history_budget({"full_history": budget})
