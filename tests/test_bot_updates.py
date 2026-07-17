from __future__ import annotations

from memcontam.clients.replay import ReplayClient
from memcontam.logging.provenance import normalize_memory_event
from memcontam.memory.bot_buffer import (
    BotBufferIdentity,
    BotBufferRegistry,
    ThoughtTemplate,
    maybe_update,
)
from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.stores import MemoryEntry


def _identity() -> BotBufferIdentity:
    return BotBufferIdentity("run1", "game24", "bot_style", "clean", "replay")


def _candidate(content: str = "Solved by isolating a denominator close to one.") -> ThoughtTemplate:
    return ThoughtTemplate(
        entry_id="candidate:trial-1",
        content=content,
        source_trial_id="trial-1",
        source_entry_ids=["seed-template"],
        metadata={"raw_response": "final: 6 / (1 - 3 / 4)"},
    )


def test_verified_novel_solution_updates_buffer() -> None:
    registry = BotBufferRegistry()
    identity = _identity()
    registry.insert(
        identity,
        ThoughtTemplate(
            entry_id="bot-template:old",
            content="For 24, first make a small fraction, then multiply.",
            source_trial_id="trial-old",
        ),
    )
    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_thought_distill": "Use denominator complements before final division.",
                "bot_novelty_decide": "True - the denominator-complement strategy is distinct.",
            }
        }
    )

    event = maybe_update(
        registry,
        identity,
        _candidate(),
        None,
        client,
        "replay",
        {
            "sample_id": "sample-1",
            "verifier_result": {"is_correct": True},
            "embedding_provider": FakeEmbeddingProvider(vector_dimension=8),
        },
    )

    snapshot = registry.snapshot(identity)
    assert len(snapshot) == 2
    assert snapshot[-1].content == "Use denominator complements before final division."
    assert snapshot[-1].source_trial_id == "trial-1"
    assert snapshot[-1].source_entry_ids == ["seed-template"]
    assert event["status"] == "accepted"
    assert event["event_type"] == "bot_write"
    assert event["candidate_content"] == _candidate().content
    assert event["distilled_content"] == snapshot[-1].content
    assert event["source_trial_id"] == "trial-1"
    assert event["source_entry_ids"] == ["seed-template"]
    assert event["top_existing_entry_id"] == "bot-template:old"
    assert isinstance(event["top_similarity"], float)
    assert event["novelty_decision_response"].startswith("True")
    assert event["new_entry_id"] == snapshot[-1].entry_id


def test_failed_trial_does_not_update_buffer() -> None:
    registry = BotBufferRegistry()
    identity = _identity()
    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_thought_distill": "This response must not be consumed.",
                "bot_novelty_decide": "True",
            }
        }
    )

    event = maybe_update(
        registry,
        identity,
        _candidate(),
        None,
        client,
        "replay",
        {"sample_id": "sample-1", "verifier_result": {"is_correct": False}},
    )

    assert registry.snapshot(identity) == ()
    assert event["status"] == "rejected"
    assert event["event_type"] == "bot_write_rejected"
    assert event["reject_reason"] == "verifier_failed"
    assert event["candidate_content"] == _candidate().content
    assert client._stage_indices == {}


def test_duplicate_template_is_rejected() -> None:
    registry = BotBufferRegistry()
    identity = _identity()
    registry.insert(
        identity,
        ThoughtTemplate(
            entry_id="bot-template:old",
            content="Use denominator complements before final division.",
            source_trial_id="trial-old",
        ),
    )
    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_thought_distill": "Use denominator complements before final division.",
                "bot_novelty_decide": "False - same template.",
            }
        }
    )

    event = maybe_update(
        registry,
        identity,
        _candidate(),
        None,
        client,
        "replay",
        {
            "sample_id": "sample-1",
            "verifier_result": {"is_correct": True},
            "embedding_provider": FakeEmbeddingProvider(vector_dimension=8),
        },
    )

    assert len(registry.snapshot(identity)) == 1
    assert event["status"] == "rejected"
    assert event["event_type"] == "bot_write_rejected"
    assert event["reject_reason"] == "novelty_rejected"
    assert event["top_existing_entry_id"] == "bot-template:old"
    assert event["novelty_decision_response"].startswith("False")


def _templates_to_entries(templates: tuple[ThoughtTemplate, ...]) -> list[MemoryEntry]:
    return [
        MemoryEntry(
            entry_id=t.entry_id,
            content=t.content,
            memory_type="thought_template",
            clean_or_contaminated="clean",
            source_trial_id=t.source_trial_id,
            metadata={
                "source_entry_ids": list(t.source_entry_ids),
                "raw_response": t.metadata.get("raw_response", ""),
            },
        )
        for t in templates
    ]


def test_accepted_bot_update_normalizes_insert_mutation() -> None:
    registry = BotBufferRegistry()
    identity = _identity()
    registry.insert(
        identity,
        ThoughtTemplate(
            entry_id="bot-template:old",
            content="For 24, first make a small fraction, then multiply.",
            source_trial_id="trial-old",
        ),
    )
    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_thought_distill": "Use denominator complements before final division.",
                "bot_novelty_decide": "True - the denominator-complement strategy is distinct.",
            }
        }
    )
    before = _templates_to_entries(registry.snapshot(identity))

    event = maybe_update(
        registry,
        identity,
        _candidate(),
        None,
        client,
        "replay",
        {
            "sample_id": "sample-1",
            "verifier_result": {"is_correct": True},
            "embedding_provider": FakeEmbeddingProvider(vector_dimension=8),
        },
    )

    after = _templates_to_entries(registry.snapshot(identity))
    memory_event = normalize_memory_event(
        "bot_style",
        "trial-1",
        before,
        after,
        event,
    )

    assert memory_event is not None
    assert memory_event.status == "accepted"
    assert memory_event.operation == "insert"
    assert memory_event.baseline == "bot_style"
    assert memory_event.source_trial_id == "trial-1"
    assert memory_event.before_entry_ids == ["bot-template:old"]
    assert len(memory_event.after_entry_ids) == 2
    assert memory_event.new_entry_ids == [event["new_entry_id"]]
    assert memory_event.removed_entry_ids == []
    assert memory_event.before_snapshot_hash != memory_event.after_snapshot_hash
    assert memory_event.source_entry_ids == ["seed-template"]
    assert memory_event.parent_entry_ids == ["seed-template"]
    assert memory_event.contaminated_source_ids == []
    assert memory_event.creation_origin == "thought_template"


def test_rejected_bot_update_preserves_snapshot_and_reports_no_new_entries() -> None:
    registry = BotBufferRegistry()
    identity = _identity()
    registry.insert(
        identity,
        ThoughtTemplate(
            entry_id="bot-template:old",
            content="Use denominator complements before final division.",
            source_trial_id="trial-old",
        ),
    )
    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_thought_distill": "Use denominator complements before final division.",
                "bot_novelty_decide": "False - same template.",
            }
        }
    )
    before = _templates_to_entries(registry.snapshot(identity))

    event = maybe_update(
        registry,
        identity,
        _candidate(),
        None,
        client,
        "replay",
        {
            "sample_id": "sample-1",
            "verifier_result": {"is_correct": True},
            "embedding_provider": FakeEmbeddingProvider(vector_dimension=8),
        },
    )

    after = _templates_to_entries(registry.snapshot(identity))
    memory_event = normalize_memory_event(
        "bot_style",
        "trial-1",
        before,
        after,
        event,
    )

    assert memory_event is not None
    assert memory_event.status == "rejected"
    assert memory_event.operation == "insert"
    assert memory_event.before_entry_ids == ["bot-template:old"]
    assert memory_event.after_entry_ids == ["bot-template:old"]
    assert memory_event.new_entry_ids == []
    assert memory_event.updated_entry_ids == []
    assert memory_event.removed_entry_ids == []
    assert memory_event.before_snapshot_hash == memory_event.after_snapshot_hash
    assert memory_event.creation_origin is None
    assert memory_event.memory_version is None
