from __future__ import annotations

import importlib

from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
from memcontam.clients.replay import ReplayClient
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


def test_exact_lineage_never_promotes_visible_context_or_sources_to_parents() -> None:
    provenance = importlib.import_module("memcontam.logging.provenance")
    event = provenance.normalize_memory_event(
        baseline="bot_style",
        source_trial_id="trial-1",
        memory_before=[],
        memory_after=[
            MemoryEntry(
                entry_id="derived-1",
                content="derived",
                memory_type="thought_template",
                clean_or_contaminated="clean",
                source_trial_id="trial-1",
                metadata={"source_entry_ids": ["visible-source"]},
            )
        ],
        memory_write_event={"type": "insert", "source_entry_ids": ["visible-source"]},
    )

    assert event is not None
    assert event.parent_entry_ids == []
    assert event.source_entry_ids == ["visible-source"]


def test_bot_write_uses_only_explicitly_used_visible_ids_as_parents_and_supports() -> None:
    bot_write = importlib.import_module("memcontam.baselines.bot_write")
    payload = bot_write.BoTTemplatePayload(
        description="A reusable arithmetic procedure.",
        template="Build factor pairs before combining them.",
        category="procedure-based",
        explicitly_used_memory_ids=("used-template",),
    )

    entry = bot_write.build_template_entry(
        payload=payload,
        source_trial_id="trial-1",
        visible_entry_ids=["visible-but-unused", "used-template"],
    )

    assert entry.metadata["declared_updater_context_ids"] == [
        "visible-but-unused",
        "used-template",
    ]
    assert entry.metadata["direct_parent_ids"] == ["used-template"]
    assert entry.metadata["memory_support_ids"] == ["used-template"]
    assert entry.metadata["source_entry_ids"] == ["used-template"]


def test_reflexion_reflection_call_and_entry_record_failed_actor_lineage() -> None:
    state = ReflexionState(
        reflections=[
            MemoryEntry(
                entry_id="visible-one",
                content="Reflection: one",
                memory_type="verbal_reflection",
            ),
            MemoryEntry(
                entry_id="visible-two",
                content="Reflection: two",
                memory_type="verbal_reflection",
                clean_or_contaminated="contaminated",
                metadata={"source_entry_ids": ["contaminated-origin"]},
            ),
            MemoryEntry(
                entry_id="visible-three",
                content="Reflection: three",
                memory_type="verbal_reflection",
            ),
        ]
    )
    task = TaskInstance(sample_id="sample-1", task_name="game24", input={})
    outcome = ReflexionAdapter().execute(
        task,
        state,
        client=ReplayClient(
            responses_by_sample={
                "sample-1": {
                    "reflexion_generate": "final: wrong",
                    "reflexion_reflect": (
                        '{"mode":"corrective","failure_class":"incorrect_answer",'
                        '"reflection_text":"retry","explicitly_used_memory_ids":["visible-two"]}'
                    ),
                }
            }
        ),
        model="replay",
        config={"run_id": "run-1", "max_attempts": 1},
        verifier=lambda _answer, _task: False,
    )

    failed_actor_call, reflection_call = outcome.method_calls
    trajectory_span = reflection_call.source_spans[-1]
    assert [span.entry_id for span in reflection_call.source_spans[:-1]] == [
        "visible-one",
        "visible-two",
        "visible-three",
    ]
    assert trajectory_span.entry_id == f"reflexion_failed_actor:{failed_actor_call.call_id}"
    assert trajectory_span.parent_call_id == failed_actor_call.call_id
    assert trajectory_span.source_ids == ["contaminated-origin"]
    assert trajectory_span.parent_ids == ["visible-one", "visible-two", "visible-three"]

    entry = state.reflections[-1]
    assert entry.metadata["creation_call_id"] == reflection_call.call_id
    assert entry.metadata["failed_actor_call_id"] == failed_actor_call.call_id
    assert entry.metadata["parent_call_ids"] == [failed_actor_call.call_id]
    assert entry.metadata["declared_updater_context_ids"] == [
        "visible-one",
        "visible-two",
        "visible-three",
    ]
    assert entry.metadata["direct_parent_ids"] == ["visible-two"]
    assert entry.metadata["memory_support_ids"] == ["visible-two"]
    assert entry.metadata["source_entry_ids"] == ["visible-two"]
