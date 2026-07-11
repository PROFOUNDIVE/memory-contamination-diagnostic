from datetime import datetime, timezone

from memcontam.memory.bot_buffer import BotBufferIdentity, BotBufferRegistry, ThoughtTemplate


def test_same_identity_reuses_accepted_template():
    registry = BotBufferRegistry()
    identity = BotBufferIdentity(
        run_id="run_a",
        task_name="game24",
        baseline="bot_style",
        arm="clean",
        backbone="gpt-4o",
    )
    entry = ThoughtTemplate(
        entry_id="bot_template:abc123",
        content="Use the 24 target and combine largest numbers first.",
        source_trial_id="run_a:game24:s1:bot_style:clean:gpt-4o",
        source_entry_ids=["clean_strategy_01"],
        metadata={"distillation_source": "bot_writeback"},
    )

    registry.insert(identity, entry)

    snapshot = registry.snapshot(identity)
    assert len(snapshot) == 1
    stored = snapshot[0]
    assert stored.entry_id == "bot_template:abc123"
    assert stored.content == entry.content
    assert stored.source_trial_id == entry.source_trial_id
    assert stored.source_entry_ids == ["clean_strategy_01"]
    assert stored.accepted_at is not None
    assert stored.metadata == entry.metadata

    later = registry.snapshot(identity)
    assert len(later) == 1
    assert later[0].entry_id == "bot_template:abc123"
    assert later[0].source_trial_id == entry.source_trial_id

    assert snapshot == later

    clone = registry.clone(identity)
    assert len(clone) == 1
    clone[0].content = "mutated"
    clone[0].metadata["extra"] = True
    assert registry.snapshot(identity)[0].content == entry.content
    assert "extra" not in registry.snapshot(identity)[0].metadata


def test_buffer_isolates_run_task_arm_and_backbone():
    registry = BotBufferRegistry()
    base = BotBufferIdentity(
        run_id="run_a",
        task_name="game24",
        baseline="bot_style",
        arm="clean",
        backbone="gpt-4o",
    )
    registry.insert(
        base,
        ThoughtTemplate(
            entry_id="bot_template:xyz789",
            content="A game24 thought template.",
            source_trial_id="run_a:game24:s1:bot_style:clean:gpt-4o",
        ),
    )

    variations = [
        BotBufferIdentity("run_b", "game24", "bot_style", "clean", "gpt-4o"),
        BotBufferIdentity("run_a", "math_equation_balancer", "bot_style", "clean", "gpt-4o"),
        BotBufferIdentity("run_a", "game24", "retrieval_rag", "clean", "gpt-4o"),
        BotBufferIdentity("run_a", "game24", "bot_style", "contaminated", "gpt-4o"),
        BotBufferIdentity("run_a", "game24", "bot_style", "clean", "claude-3-5-sonnet"),
    ]

    for variant in variations:
        assert registry.snapshot(variant) == ()

    assert len(registry.snapshot(base)) == 1


def test_insert_preserves_order_and_sets_accepted_at():
    registry = BotBufferRegistry()
    identity = BotBufferIdentity(
        run_id="run_x",
        task_name="word_sorting",
        baseline="bot_style",
        arm="contaminated_filter",
        backbone="model-a",
    )
    now = datetime.now(timezone.utc)
    registry.insert(
        identity,
        ThoughtTemplate(
            entry_id="e1",
            content="first",
            source_trial_id="t1",
            accepted_at=now,
        ),
    )
    registry.insert(
        identity,
        ThoughtTemplate(
            entry_id="e2",
            content="second",
            source_trial_id="t2",
        ),
    )

    snapshot = registry.snapshot(identity)
    assert [e.entry_id for e in snapshot] == ["e1", "e2"]
    assert snapshot[0].accepted_at == now
    assert snapshot[1].accepted_at is not None
    assert snapshot[1].accepted_at >= now
