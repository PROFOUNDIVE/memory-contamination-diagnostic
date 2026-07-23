from __future__ import annotations

import pytest
from typing import Literal

from memcontam.baselines.full_history import FullHistoryPayload, render_full_history
from memcontam.baselines.full_history_phase12 import (
    FullHistoryContractError,
    FullHistoryPhase12Adapter,
    FullHistoryStateV3,
    TrialContextV3,
    verify_complete_fit,
)
from memcontam.baselines.prompt_budget import count_prompt_tokens
from memcontam.clients.base import LLMResponse
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.filtered_state import partition_native_checkpoint
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


class _Client:
    def chat(self, messages, model, config) -> LLMResponse:
        del messages, model, config
        return LLMResponse(content="final: 24", raw={"replay": True}, token_usage={}, latency_ms=0)


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    )


def _record(entry_id: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=render_full_history(
            entry_id, FullHistoryPayload("1 3 4 6", "final: (6 / (1 - 3 / 4))")
        ),
        memory_type="full_history_transcript",
        clean_or_contaminated="contaminated" if entry_id == "injected-root" else "clean",
    )


def _context_config(*, history_budget: int, policy: str = "oldest_first_pair_atomic") -> dict[str, object]:
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
        "eviction_policy": policy,
    }


def _trial(
    *,
    trial_id: str,
    order_key: int,
    context_config: dict[str, object],
    fh_mode: Literal["exact", "bounded"] = "bounded",
) -> TrialContextV3:
    return TrialContextV3(
        task=_task(),
        client=_Client(),
        model="replay",
        trial_id=trial_id,
        condition_id="fh_bounded",
        fh_mode=fh_mode,
        context_config=context_config,
        context_budget_id="fh-budget-1",
        order_key=order_key,
    )


def test_root_is_visible_then_evicted_but_persists_in_store() -> None:
    state = FullHistoryStateV3(records=[_record("injected-root")], injected_root_id="injected-root")
    adapter = FullHistoryPhase12Adapter()

    initially_visible = adapter.execute(
        _trial(trial_id="trial-1", order_key=1, context_config=_context_config(history_budget=10_000)),
        state,
    )
    evicted = adapter.execute(
        _trial(trial_id="trial-2", order_key=2, context_config=_context_config(history_budget=1)),
        state,
    )

    assert initially_visible.retention.post_record_ids == ("injected-root",)
    assert initially_visible.retention.injected_root_visible
    assert evicted.retention.removed_record_ids[0] == "injected-root"
    assert evicted.retention.first_eviction_trial_id == "trial-2"
    assert evicted.retention.injected_root_persists_in_store
    assert evicted.retention.storage_persisted
    assert evicted.retention.context_budget_id == "fh-budget-1"
    assert [record.entry_id for record in state.records][0] == "injected-root"
    assert len(state.records) == 3


def test_rejects_exact_label_after_truncation_or_non_fifo_policy() -> None:
    root = _record("injected-root")

    with pytest.raises(FullHistoryContractError, match="FALSE_EXACT_LABEL"):
        verify_complete_fit(
            _task(),
            [root],
            _context_config(history_budget=1),
            requested_fh_mode="exact",
        )
    with pytest.raises(FullHistoryContractError, match="NON_FIFO_EVICTION_POLICY"):
        verify_complete_fit(
            _task(),
            [root],
            _context_config(history_budget=10_000, policy="newest_first"),
            requested_fh_mode="bounded",
        )


def test_rejects_an_injected_root_that_cannot_be_shown_initially() -> None:
    state = FullHistoryStateV3(records=[_record("injected-root")], injected_root_id="injected-root")

    with pytest.raises(FullHistoryContractError, match="IMMEDIATE_INJECTED_ROOT_TRUNCATION"):
        FullHistoryPhase12Adapter().execute(
            _trial(trial_id="trial-1", order_key=1, context_config=_context_config(history_budget=1)),
            state,
        )


def test_routes_post_trial_writes_through_the_production_filter() -> None:
    record = _record("history-1")
    entry = NativeEntry(
        entry_id=record.entry_id,
        semantic_kind="full_history_transcript",
        schema_version=NATIVE_ENTRY_V1,
        native_component="history",
        content=record.content,
        content_hash=canonical_content_hash(record.content),
    )
    envelope = MemoryCardEnvelopeV3(
        entry_id=entry.entry_id,
        baseline="fh_bounded",
        semantic_kind=entry.semantic_kind,
        schema_version=MEMORY_CARD_V3,
        writer_id="fh_appender",
        writer_event_id="event-history-1",
        writer_stage="full_history_generate",
        created_trial_id="trial-history-1",
        source_trial_ids=("trial-history-1",),
        source_outcome=None,
        trial_support_ids=("trial-history-1",),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=1,
        native_component=entry.native_component,
        content=entry.content,
        content_hash=entry.content_hash,
    )
    context = AdmissionContext(
        writer_event_ids=frozenset({envelope.writer_event_id}),
        trial_record_ids=frozenset({"trial-history-1"}),
        evidence_envelopes=(envelope,),
    )
    partition = partition_native_checkpoint(
        serialize_checkpoint(NativeState("fh_bounded", (entry,), {"records": []})), context
    )
    state = FullHistoryStateV3(
        records=[record], filter_state=partition, admission_context=context
    )

    result = FullHistoryPhase12Adapter().execute(
        _trial(
            trial_id="trial-2", order_key=2, context_config=_context_config(history_budget=10_000)
        ),
        state,
    )

    assert result.filter_transition is not None
    assert result.filter_transition.decision.admitted
    assert result.write_envelope is not None
    assert result.write_envelope.source_outcome is None
    assert result.write_envelope.trial_support_ids == ("trial-2",)
    assert result.write_envelope.entry_id in {
        entry.entry_id if isinstance(entry, NativeEntry) else entry
        for entry in result.filter_transition.state.reader_entries
    }
