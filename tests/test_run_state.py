from importlib import import_module

import pytest

from memcontam.memory.bot_buffer import BotBufferIdentity, ThoughtTemplate

RunState = import_module("memcontam.memory.run_state").RunState


def _identity(arm: str = "clean") -> BotBufferIdentity:
    return BotBufferIdentity("run1", "game24", "bot_style", arm, "replay")


def _event(sample_id: str, entry_id: str = "warmup-template-1") -> dict:
    return {
        "sample_id": sample_id,
        "status": "accepted",
        "new_entry_id": entry_id,
        "distilled_content": "Use arithmetic pairings that preserve the target.",
        "source_trial_id": f"warmup:{sample_id}",
        "source_entry_ids": [f"warmup:{sample_id}"],
    }


def test_warmup_snapshot_clones_identical_arm_state():
    state = RunState("run1", config_hash="cfg-a", mode="fresh", evaluation_sample_ids=["eval-1"])
    clean = _identity()

    state.register_warmup_result(clean, _event("warmup-1"))

    snapshot = state.snapshot_clean_warmup(clean)
    contaminated = state.clone_for_arm(
        _identity("contaminated"),
        [
            ThoughtTemplate(
                entry_id="corrupt-1",
                content="Bad injected template",
                source_trial_id="catalog:corrupt-1",
                metadata={"tags": ["corrupted"]},
            )
        ],
    )
    filtered = state.clone_for_arm(_identity("contaminated_filter"), [])

    assert [entry.entry_id for entry in snapshot.entries] == ["warmup-template-1"]
    assert snapshot.metadata["warmup_sample_ids"] == ["warmup-1"]
    assert snapshot.metadata["accepted_template_ids"] == ["warmup-template-1"]
    assert snapshot.metadata["snapshot_hash"]

    assert [entry.entry_id for entry in contaminated] == ["warmup-template-1", "corrupt-1"]
    assert [entry.entry_id for entry in filtered] == ["warmup-template-1"]
    assert state.arm_metadata(_identity("contaminated"))["injection_ids"] == ["corrupt-1"]

    contaminated[0].source_entry_ids.append("mutated")
    contaminated[1].metadata["tags"].append("mutated")
    assert snapshot.entries[0].source_entry_ids == ["warmup:warmup-1"]
    assert filtered[0].source_entry_ids == ["warmup:warmup-1"]
    assert state.snapshot_clean_warmup(clean).entries[0].source_entry_ids == ["warmup:warmup-1"]


def test_warmup_rejects_evaluation_samples():
    state = RunState("run1", config_hash="cfg-a", mode="fresh", evaluation_sample_ids=["eval-1"])

    with pytest.raises(ValueError, match="evaluation sample"):
        state.register_warmup_result(_identity(), _event("eval-1"))


def test_resume_rejects_state_hash_mismatch():
    with pytest.raises(ValueError, match="state config hash mismatch"):
        RunState("run1", config_hash="cfg-new", mode="resume", state_config_hash="cfg-old")
