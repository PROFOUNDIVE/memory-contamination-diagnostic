from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
import sys

import pytest

from memcontam.baselines.contracts import StreamIdentity
from memcontam.memory.admission import AdmissionDecision
from memcontam.memory.cards import MemoryCard, MemoryCardEnvelope


def _source_checkpoint(checkpoints):
    return checkpoints.NativeCheckpoint(
        identity=StreamIdentity("run", "game24", "full_history", "contaminated", "replay"),
        cards=(
            MemoryCard(
                card_id="first",
                content="first payload",
                card_type="history",
                metadata={"hidden_origin": "injected"},
            ),
            MemoryCard(
                card_id="second",
                content="second payload",
                card_type="history",
                metadata={"hidden_origin": "clean"},
            ),
            MemoryCard(
                card_id="third",
                content="third payload",
                card_type="history",
                metadata={"hidden_origin": "injected"},
            ),
        ),
        envelopes=(
            MemoryCardEnvelope(
                entry_id="first",
                semantic_kind="history",
                writer_id="writer",
                writer_event_id="event-first",
                trial_log_support_ids=("trial-1",),
                memory_support_ids=(),
                declared_parent_ids=(),
                source_trial_id="trial-1",
                source_outcome=True,
                order_key="01",
            ),
            MemoryCardEnvelope(
                entry_id="second",
                semantic_kind="history",
                writer_id="writer",
                writer_event_id="event-second",
                trial_log_support_ids=("trial-2",),
                memory_support_ids=(),
                declared_parent_ids=(),
                source_trial_id="trial-2",
                source_outcome=False,
                order_key="02",
            ),
            MemoryCardEnvelope(
                entry_id="third",
                semantic_kind="history",
                writer_id="writer",
                writer_event_id="event-third",
                trial_log_support_ids=("trial-3",),
                memory_support_ids=(),
                declared_parent_ids=(),
                source_trial_id="trial-3",
                source_outcome=True,
                order_key="03",
            ),
        ),
        parameters={"window": 3, "nested": {"mode": "native"}},
    )


def test_inactive_filtered_state_consumes_admission_decisions_without_recomputing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "memcontam.memory.filtered_state", raising=False)
    monkeypatch.delitem(sys.modules, "memcontam.cli", raising=False)
    assert importlib.util.find_spec("memcontam.memory.filtered_state"), (
        "Task 12 owns the inactive active/quarantine state"
    )
    filtered_state = importlib.import_module("memcontam.memory.filtered_state")

    assert getattr(filtered_state, "FilteredNativeState", None) is not None
    assert getattr(filtered_state, "PartitionDecision", None) is not None
    assert callable(getattr(filtered_state, "partition_native_state_for_fixture", None))
    assert callable(getattr(filtered_state, "serialize_filtered_state", None))
    assert callable(getattr(filtered_state, "validate_partition_preserves_native_contract", None))
    assert "memcontam.cli" not in sys.modules


def test_partition_preserves_source_entries_in_disjoint_ordered_clones() -> None:
    checkpoints = importlib.import_module("memcontam.memory.checkpoints")
    filtered_state = importlib.import_module("memcontam.memory.filtered_state")
    source = _source_checkpoint(checkpoints)
    decisions = (
        AdmissionDecision("third", True, "admitted"),
        AdmissionDecision("second", False, "unauthorized_writer"),
        AdmissionDecision("first", True, "admitted"),
    )

    partition = filtered_state.partition_native_state_for_fixture(source, decisions)

    filtered_identity = replace(source.identity, arm="contaminated_filter")
    assert partition.active.identity == filtered_identity
    assert partition.quarantine.identity == filtered_identity
    assert [card.card_id for card in partition.active.cards] == ["first", "third"]
    assert [card.card_id for card in partition.quarantine.cards] == ["second"]
    assert partition.active.envelopes == (source.envelopes[0], source.envelopes[2])
    assert partition.quarantine.envelopes == (source.envelopes[1],)
    assert partition.active.parameters == source.parameters
    assert partition.quarantine.parameters == source.parameters
    assert [(decision.entry_id, decision.state, decision.reason) for decision in partition.decisions] == [
        ("first", "active", "admitted"),
        ("second", "quarantine", "unauthorized_writer"),
        ("third", "active", "admitted"),
    ]
    assert filtered_state.validate_partition_preserves_native_contract(source, partition) is None

    partition.active.cards[0].metadata["copy_only"] = True
    partition.active.parameters["nested"]["mode"] = "filtered"
    assert source.cards[0].metadata == {"hidden_origin": "injected"}
    assert source.parameters == {"window": 3, "nested": {"mode": "native"}}


def test_partition_fails_closed_for_missing_duplicate_and_unknown_decisions() -> None:
    checkpoints = importlib.import_module("memcontam.memory.checkpoints")
    filtered_state = importlib.import_module("memcontam.memory.filtered_state")
    source = _source_checkpoint(checkpoints)

    with pytest.raises(ValueError, match="total"):
        filtered_state.partition_native_state_for_fixture(
            source,
            (AdmissionDecision("first", True, "admitted"),),
        )
    with pytest.raises(ValueError, match="duplicate"):
        filtered_state.partition_native_state_for_fixture(
            source,
            (
                AdmissionDecision("first", True, "admitted"),
                AdmissionDecision("first", False, "unauthorized_writer"),
                AdmissionDecision("second", True, "admitted"),
                AdmissionDecision("third", True, "admitted"),
            ),
        )
    with pytest.raises(ValueError, match="unknown"):
        filtered_state.partition_native_state_for_fixture(
            source,
            (
                AdmissionDecision("first", True, "admitted"),
                AdmissionDecision("second", True, "admitted"),
                AdmissionDecision("third", True, "admitted"),
                AdmissionDecision("outside", False, "invalid_schema"),
            ),
        )


def test_partition_serialization_is_deterministic_and_native_contract_validation_fails_closed() -> None:
    checkpoints = importlib.import_module("memcontam.memory.checkpoints")
    filtered_state = importlib.import_module("memcontam.memory.filtered_state")
    source = _source_checkpoint(checkpoints)
    decisions = tuple(
        AdmissionDecision(entry_id, admitted, reason)
        for entry_id, admitted, reason in (
            ("first", False, "unauthorized_writer"),
            ("second", True, "admitted"),
            ("third", False, "invalid_schema"),
        )
    )

    first = filtered_state.partition_native_state_for_fixture(source, decisions)
    second = filtered_state.partition_native_state_for_fixture(source, tuple(reversed(decisions)))

    serialized = filtered_state.serialize_filtered_state(first)
    assert serialized == filtered_state.serialize_filtered_state(second)
    assert json.loads(serialized)["active"]["cards"][0]["card_id"] == "second"

    overlapping = replace(first, quarantine=first.active)
    with pytest.raises(ValueError, match="disjoint"):
        filtered_state.validate_partition_preserves_native_contract(source, overlapping)
