from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util

import pytest

from memcontam.baselines.contracts import StreamIdentity, stream_pair_key
from memcontam.memory.cards import MemoryCard, MemoryCardEnvelope


def _checkpoint_interfaces():
    assert importlib.util.find_spec("memcontam.memory.checkpoints"), (
        "BASELINE-FIDELITY-V1 requires checkpoint clone interfaces"
    )
    return importlib.import_module("memcontam.memory.checkpoints")


def _clean_checkpoint():
    checkpoints = _checkpoint_interfaces()
    return checkpoints.NativeCheckpoint(
        identity=StreamIdentity("run", "game24", "full_history", "clean", "replay"),
        cards=(
            MemoryCard(
                card_id="card-1",
                content="trusted record",
                card_type="history",
                metadata={"labels": ["original"]},
            ),
        ),
        envelopes=(
            MemoryCardEnvelope(
                entry_id="card-1",
                semantic_kind="history",
                writer_id="writer",
                writer_event_id="event-1",
                trial_log_support_ids=("trial-1",),
                memory_support_ids=(),
                declared_parent_ids=(),
                source_trial_id="trial-1",
                source_outcome=True,
                order_key=1,
            ),
        ),
        parameters={"window": 3, "nested": {"mode": "native"}},
    )


def _contamination_root():
    contracts = importlib.import_module("memcontam.memory.contamination_contracts")
    return contracts.ContaminationRoot(
        card=MemoryCard(
            card_id="root-1",
            content="misleading root",
            card_type="contamination",
            metadata={"labels": ["injected"]},
        ),
        envelope=MemoryCardEnvelope(
            entry_id="root-1",
            semantic_kind="contamination",
            writer_id="catalog",
            writer_event_id="root-event-1",
            trial_log_support_ids=(),
            memory_support_ids=(),
            declared_parent_ids=(),
            source_trial_id=None,
            source_outcome=None,
            order_key=2,
        ),
    )


def test_checkpoint_clone_replaces_only_arm_and_keeps_clean_state_immutable() -> None:
    checkpoints = _checkpoint_interfaces()

    assert getattr(checkpoints, "NativeCheckpoint", None) is not None
    assert callable(getattr(checkpoints, "clone_checkpoint_to_arm", None))

    clean = _clean_checkpoint()
    contaminated = checkpoints.clone_checkpoint_to_arm(clean, "contaminated")

    assert contaminated.identity == replace(clean.identity, arm="contaminated")
    assert contaminated.identity != clean.identity
    assert contaminated.cards == clean.cards
    assert contaminated.envelopes == clean.envelopes
    assert contaminated.parameters == clean.parameters
    assert contaminated.checkpoint_hash != clean.checkpoint_hash

    contaminated.cards[0].metadata["labels"].append("clone-only")
    contaminated.parameters["nested"]["mode"] = "clone"
    assert clean.cards[0].metadata == {"labels": ["original"]}
    assert clean.parameters == {"window": 3, "nested": {"mode": "native"}}


def test_matched_checkpoints_share_pair_key_but_reject_identity_mismatches() -> None:
    checkpoints = _checkpoint_interfaces()
    clean = _clean_checkpoint()
    contaminated = checkpoints.clone_checkpoint_to_arm(clean, "contaminated")

    assert checkpoints.validate_stream_pair(clean, contaminated) == stream_pair_key(clean.identity)
    assert stream_pair_key(clean.identity) == stream_pair_key(contaminated.identity)

    with pytest.raises(ValueError, match="different arms"):
        checkpoints.validate_stream_pair(clean, clean)

    mismatched = replace(
        contaminated,
        identity=replace(contaminated.identity, task_family="word_sorting"),
    )
    with pytest.raises(ValueError, match="non-arm"):
        checkpoints.validate_stream_pair(clean, mismatched)


def test_native_renderer_inserts_one_root_without_changing_existing_checkpoint_data() -> None:
    checkpoints = _checkpoint_interfaces()
    contracts = importlib.import_module("memcontam.memory.contamination_contracts")
    clean = _clean_checkpoint()
    contaminated = checkpoints.clone_checkpoint_to_arm(clean, "contaminated")

    rendered = contracts.NativeContaminationRenderer().render(
        contaminated, (_contamination_root(),)
    )

    assert rendered.identity == contaminated.identity
    assert rendered.cards[:-1] == contaminated.cards
    assert rendered.envelopes[:-1] == contaminated.envelopes
    assert rendered.parameters == contaminated.parameters
    assert rendered.cards[-1].card_id == "root-1"
    assert rendered.envelopes[-1].entry_id == "root-1"
    assert contaminated.cards == clean.cards
    assert contaminated.envelopes == clean.envelopes
    assert clean.cards == _clean_checkpoint().cards

    rendered.cards[0].metadata["labels"].append("rendered-only")
    assert contaminated.cards[0].metadata == {"labels": ["original"]}


def test_native_renderer_requires_exactly_one_root_and_rejects_clean_rendering() -> None:
    checkpoints = _checkpoint_interfaces()
    contracts = importlib.import_module("memcontam.memory.contamination_contracts")
    clean = _clean_checkpoint()
    contaminated = checkpoints.clone_checkpoint_to_arm(clean, "contaminated")
    renderer = contracts.NativeContaminationRenderer()
    root = _contamination_root()

    with pytest.raises(ValueError, match="exactly one"):
        renderer.render(contaminated, ())
    with pytest.raises(ValueError, match="exactly one"):
        renderer.render(contaminated, (root, root))
    with pytest.raises(ValueError, match="clean"):
        renderer.render(clean, (root,))
