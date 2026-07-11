from __future__ import annotations

from memcontam.clients.replay import ReplayClient
from memcontam.memory.bot_buffer import (
    BotBufferIdentity,
    BotBufferRegistry,
    ThoughtTemplate,
    maybe_update,
)
from memcontam.memory.embeddings import FakeEmbeddingProvider


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
