from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import cast

import pytest

from memcontam.memory.cards import MemoryCard, MemoryCardEnvelope


def _entry(
    entry_id: str,
    *,
    writer_id: str = "writer",
    card_type: str = "reflection",
    semantic_kind: str = "reflection",
    parents: tuple[str, ...] = (),
    supports: tuple[str, ...] = (),
    trial_supports: tuple[str, ...] = (),
    order_key: int = 1,
) -> tuple[MemoryCard, MemoryCardEnvelope]:
    return (
        MemoryCard(
            card_id=entry_id,
            content=f"content for {entry_id}",
            card_type=card_type,
            metadata={"origin": "hidden", "contamination": "hidden"},
        ),
        MemoryCardEnvelope(
            entry_id=entry_id,
            semantic_kind=semantic_kind,
            writer_id=writer_id,
            writer_event_id=f"event-{entry_id}",
            trial_log_support_ids=trial_supports,
            memory_support_ids=supports,
            declared_parent_ids=parents,
            source_trial_id="trial-1" if trial_supports else None,
            source_outcome=True,
            order_key=order_key,
        ),
    )


def _context(admission, *, prior=(), trial_supports=("trial-1",)):
    return admission.AdmissionContext(
        authorized_writers=admission.AuthorizedWriterRegistry({"writer"}),
        trial_log_support_ids=frozenset(trial_supports),
        admitted_envelopes=prior,
    )


def test_inactive_admission_evaluator_is_pure_fail_closed_and_runner_independent() -> None:
    assert importlib.util.find_spec("memcontam.memory.admission"), (
        "Task 11 owns the inactive admission evaluator"
    )
    admission = importlib.import_module("memcontam.memory.admission")

    assert callable(getattr(admission, "evaluate_admission_graph", None))
    assert callable(getattr(admission, "evaluate_entry_admission", None))
    assert getattr(admission, "AdmissionContext", None) is not None
    assert getattr(admission, "AdmissionDecision", None) is not None
    assert getattr(admission, "AdmissionGraphError", None) is not None
    assert getattr(admission, "AuthorizedWriterRegistry", None) is not None
    assert callable(getattr(admission, "validate_support_reference", None))
    assert callable(getattr(admission, "validate_parent_graph", None))


def test_evaluator_admits_authorized_roots_and_recursively_admitted_parents() -> None:
    admission = importlib.import_module("memcontam.memory.admission")
    root = _entry("root", trial_supports=("trial-1",), order_key=1)
    child = _entry("child", parents=("root",), supports=("root",), order_key=2)
    context = _context(admission)

    decisions = admission.evaluate_admission_graph((root, child), context)

    assert [(decision.entry_id, decision.admitted, decision.reason) for decision in decisions] == [
        ("root", True, "admitted"),
        ("child", True, "admitted"),
    ]
    assert admission.evaluate_entry_admission(child[0], child[1], _context(admission, prior=(root[1],))).admitted


def test_evaluator_rejects_unauthorized_schema_support_missing_future_and_rejected_parents() -> None:
    admission = importlib.import_module("memcontam.memory.admission")
    context = _context(admission)
    unauthorized = _entry("unauthorized", writer_id="unknown")
    invalid_schema = _entry("invalid-schema", card_type="template", semantic_kind="reflection")
    invalid_support_card, invalid_support = _entry("invalid-support")
    object.__setattr__(invalid_support, "memory_support_ids", ("missing",))
    missing = _entry("missing", parents=("not-present",), order_key=2)
    future_child = _entry("future-child", parents=("future-parent",), order_key=1)
    future_parent = _entry("future-parent", order_key=2)
    rejected_parent = _entry("rejected-parent", writer_id="unknown", order_key=1)
    rejected_child = _entry("rejected-child", parents=("rejected-parent",), order_key=2)

    decisions = admission.evaluate_admission_graph(
        (
            unauthorized,
            invalid_schema,
            (invalid_support_card, invalid_support),
            missing,
            future_child,
            future_parent,
            rejected_parent,
            rejected_child,
        ),
        context,
    )

    assert {decision.entry_id: decision.reason for decision in decisions} == {
        "unauthorized": "unauthorized_writer",
        "invalid-schema": "invalid_schema",
        "invalid-support": "invalid_support",
        "missing": "missing_reference",
        "future-child": "future_reference",
        "future-parent": "admitted",
        "rejected-parent": "unauthorized_writer",
        "rejected-child": "rejected_parent",
    }


def test_evaluator_rejects_cycles_without_mutating_entries_or_reading_hidden_metadata() -> None:
    admission = importlib.import_module("memcontam.memory.admission")
    first = _entry("first", parents=("second",), order_key=1)
    second = _entry("second", parents=("first",), order_key=2)
    source_entries = (first, second)
    original_metadata = [dict(card.metadata) for card, _ in source_entries]
    original_envelopes = tuple(envelope for _, envelope in source_entries)

    first_pass = admission.evaluate_admission_graph(source_entries, _context(admission))
    second_pass = admission.evaluate_admission_graph(tuple(reversed(source_entries)), _context(admission))

    assert {decision.entry_id: decision.reason for decision in first_pass} == {
        "first": "cycle",
        "second": "cycle",
    }
    assert {(decision.entry_id, decision.admitted, decision.reason) for decision in first_pass} == {
        (decision.entry_id, decision.admitted, decision.reason) for decision in second_pass
    }
    assert [card.metadata for card, _ in source_entries] == original_metadata
    assert tuple(envelope for _, envelope in source_entries) == original_envelopes
    assert "memcontam.cli" not in sys.modules


def test_evaluator_fails_closed_for_a_malformed_envelope() -> None:
    admission = importlib.import_module("memcontam.memory.admission")
    card, _ = _entry("malformed")

    decisions = admission.evaluate_admission_graph(
        ((card, cast(MemoryCardEnvelope, object())),), _context(admission)
    )

    assert decisions == (admission.AdmissionDecision("malformed", False, "invalid_schema"),)


def test_evaluator_fails_closed_for_malformed_entry_shapes() -> None:
    admission = importlib.import_module("memcontam.memory.admission")
    malformed_pair = cast(tuple[MemoryCard, MemoryCardEnvelope], (object(),))

    decisions = admission.evaluate_admission_graph((malformed_pair,), _context(admission))

    assert decisions == (admission.AdmissionDecision("", False, "invalid_schema"),)


def test_evaluator_rejects_duplicate_ids_in_current_and_admitted_graphs() -> None:
    admission = importlib.import_module("memcontam.memory.admission")
    first = _entry("duplicate", order_key=1)
    second = _entry("duplicate", order_key=2)

    current_decisions = admission.evaluate_admission_graph((first, second), _context(admission))
    prior_decision = admission.evaluate_entry_admission(
        second[0], second[1], _context(admission, prior=(first[1],))
    )

    assert current_decisions == (
        admission.AdmissionDecision("duplicate", False, "invalid_schema"),
        admission.AdmissionDecision("duplicate", False, "invalid_schema"),
    )
    assert prior_decision == admission.AdmissionDecision("duplicate", False, "invalid_schema")
    with pytest.raises(admission.AdmissionGraphError, match="invalid_schema"):
        admission.validate_parent_graph((first[1], second[1]))
