from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pytest

from memcontam.evaluation.failure_classifier import FailureClassifier, classify_failure
from memcontam.evaluation.sequential import SequentialOutcomeError, compute_sequential_outcomes


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-SEQUENTIAL-001.json"


def _lineage() -> dict[str, dict[str, Any]]:
    return {
        "root-b": {"lineage_status": "exact", "injected_root_ids": ["root-b"]},
        "child-b1": {
            "lineage_status": "exact",
            "injected_root_ids": ["root-b"],
            "direct_parent_ids": ["root-b"],
            "kind": "derived",
        },
    }


def test_computes_bot_propagation_and_fh_storage_persistence() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    events = fixture["events"]
    trials = [
        {
            "trial_id": f"trial-{event['trial']}",
            "order_key": event["trial"],
            "baseline": "bot_style" if event["trial"] < 3 else "fh_bounded",
            "failure_class": event["failure_class"],
            "final_context_entry_ids": event["final_sources"],
            "retention": (
                None
                if event["trial"] < 3
                else {
                    "root_id": "root-b",
                    "root_persists_in_store": True,
                    "root_visible_in_prompt": False,
                    "first_eviction_trial_id": "trial-3",
                }
            ),
        }
        for event in events
    ]
    memory_events = [
        {
            "trial_id": "trial-2",
            "new_entry_ids": ["child-b1"],
            "after_entry_ids": ["root-b", "child-b1"],
        },
        {"trial_id": "trial-3", "after_entry_ids": ["root-b"]},
    ]

    outcomes = compute_sequential_outcomes(trials, memory_events, _lineage(), window=2)

    second, third = outcomes.trials[1:]
    assert second.generic_recurrence is True
    assert second.same_root_exact_lineage_recurrence is True
    assert second.propagation.status == "supported"
    assert second.propagation.value is True
    assert second.propagation.path == ("root-b", "child-b1")
    assert third.root_storage_persistence is True
    assert third.root_prompt_visibility is False
    assert third.retention.status == "supported"
    assert third.eviction.status == "supported"
    assert third.eviction.value is True
    assert outcomes.summary["generic_recurrence_at_2"] == 1
    assert outcomes.summary["exact_lineage_recurrence_at_2"] == 1
    assert outcomes.summary["propagation_transition_2"] == 1
    assert outcomes.summary["root_persistence_at_3"] == 1


def test_rejects_retrieval_as_propagation_and_generic_error_as_recurrence() -> None:
    trials = [
        {
            "trial_id": "trial-1",
            "order_key": 1,
            "baseline": "rag_frozen",
            "failure_class": "incorrect_answer",
            "final_context_entry_ids": ["root-b"],
            "retrieved_entry_ids": ["root-b"],
        },
        {
            "trial_id": "trial-2",
            "order_key": 2,
            "baseline": "rag_frozen",
            "failure_class": "incorrect_answer",
            "final_context_entry_ids": ["root-b"],
            "retrieved_entry_ids": ["root-b"],
        },
    ]

    outcomes = compute_sequential_outcomes(trials, [], _lineage(), window=2)

    second = outcomes.trials[1]
    assert second.generic_recurrence is False
    assert second.same_root_exact_lineage_recurrence is False
    assert second.propagation.status == "supported"
    assert second.propagation.value is False
    assert second.retention.status == "not_applicable"


def test_reports_reflexion_eviction_only_from_supported_telemetry() -> None:
    outcomes = compute_sequential_outcomes(
        [
            {
                "trial_id": "trial-1",
                "order_key": 1,
                "baseline": "reflexion_style",
                "eviction_events": [{"entry_id": "root-b"}],
            }
        ],
        [],
        _lineage(),
        window=1,
    )

    assert outcomes.trials[0].retention.status == "unavailable"
    assert outcomes.trials[0].eviction.status == "supported"
    assert outcomes.trials[0].eviction.value is True


def test_classifies_only_registered_task_specific_failures() -> None:
    classifier = FailureClassifier(
        class_id="fraction_pruned",
        classify=lambda _query, output, _verifier: output == "fraction-pruned",
    )

    assert classify_failure({"task_name": "game24"}, "fraction-pruned", None, {"game24": [classifier]}) == "fraction_pruned"
    assert classify_failure({"task_name": "game24"}, "generic error", None, {"game24": [classifier]}) is None


@pytest.mark.parametrize(
    ("trials", "memory_events", "lineage", "window", "code"),
    [
        ([], [], {}, None, "WINDOW_REQUIRED"),
        (
            [{"trial_id": "one", "order_key": 1, "baseline": "bot_style"}],
            [{"trial_id": "one", "new_entry_ids": ["fabricated"]}],
            {},
            1,
            "FABRICATED_LINEAGE",
        ),
        (
            [{"trial_id": "one", "order_key": 1, "baseline": "rag_frozen", "retention": {}}],
            [],
            {},
            1,
            "UNSUPPORTED_BASELINE_OPERATION",
        ),
    ],
)
def test_fails_closed_for_invalid_sequential_inputs(
    trials: list[dict[str, object]],
    memory_events: list[dict[str, object]],
    lineage: Mapping[str, Mapping[str, Any]],
    window: int | None,
    code: str,
) -> None:
    with pytest.raises(SequentialOutcomeError, match=code):
        compute_sequential_outcomes(trials, memory_events, lineage, window)
