from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from memcontam.logging.schema_v3 import (
    PreRouteRunMetadata,
    MemoryArmExecutionKey,
    parse_log_record_v3,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-AGG-001.json"
SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-SCHEMA-001.json"
ARMS = ("clean", "correct", "irrelevant", "contam", "filter")


def _metadata(seed: int, arm: str, **changes: object):
    payload = json.loads(SCHEMA_FIXTURE_PATH.read_text(encoding="utf-8"))["valid_run_metadata"][0]
    payload = copy.deepcopy(payload)
    payload.update(
        {
            "trajectory_seed": seed,
            "abstract_seed_slot_or_none": None,
            "execution_key": {"kind": "memory_arm", "arm": arm},
            "protocol_index_or_none": arm if arm in {"clean", "contam", "filter"} else None,
        }
    )
    payload.update(changes)
    metadata = parse_log_record_v3(payload)
    assert isinstance(metadata, PreRouteRunMetadata)
    return metadata


def _run(seed: int, arm: str, accuracy: float, **metadata_changes: object):
    from memcontam.evaluation.phase12_aggregate import AggregateTrial, ValidatedRun

    successes = int(accuracy * 4)
    trials = tuple(
        AggregateTrial(
            trial_id=f"seed-{seed}:{arm}:{index}",
            verified_score=1 if index < successes else 0,
            analysis_inclusion="included",
            execution_status="completed",
            failure_class="none" if index < successes else "model_behavior",
            eligible=True,
        )
        for index in range(4)
    )
    return ValidatedRun(_metadata(seed, arm, **metadata_changes), trials)


def _panel():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return tuple(
        _run(int(seed[1:]), arm, accuracy)
        for seed, accuracies in fixture["seed_accuracy"].items()
        for arm, accuracy in accuracies.items()
    )


def test_reconstructs_paired_five_arm_seed_aggregate() -> None:
    from memcontam.evaluation.phase12_aggregate import AggregateSpec, aggregate_phase12

    aggregate = aggregate_phase12(_panel(), AggregateSpec())

    assert len(aggregate.cells) == 1
    cell = aggregate.cells[0]
    assert cell.seed_count == 2
    assert cell.contrasts == {
        "clean_minus_contam": 0.5,
        "clean_minus_filter": 0.25,
        "correct_minus_contam": 0.375,
        "filter_minus_contam": 0.25,
        "irrelevant_minus_contam": 0.25,
    }
    assert cell.seed_scores[1]["clean"] == 1.0
    assert cell.seed_scores[2]["contam"] == 0.25


def test_rejects_incomplete_mixed_or_trial_resampled_inputs() -> None:
    from memcontam.evaluation.phase12_aggregate import (
        AggregateError,
        AggregateSpec,
        aggregate_phase12,
    )

    incomplete = tuple(
        run
        for run in _panel()
        if isinstance(run.metadata.execution_key, MemoryArmExecutionKey)
        and run.metadata.execution_key.arm != "filter"
    )
    with pytest.raises(AggregateError, match="INCOMPLETE_FIVE_ARM_PAIR"):
        aggregate_phase12(incomplete, AggregateSpec())

    mixed = list(_panel())
    mixed[-1] = _run(2, "filter", 0.5, embedding_contract_hash="sha256:mixed")
    with pytest.raises(AggregateError, match="COMPATIBILITY_MISMATCH"):
        aggregate_phase12(mixed, AggregateSpec())

    with pytest.raises(AggregateError, match="TRIAL_RESAMPLING_FORBIDDEN"):
        aggregate_phase12(_panel(), AggregateSpec(resampling_unit="trial"))

    with pytest.raises(AggregateError, match="COMPLETE_CASE_SELECTION_FORBIDDEN"):
        aggregate_phase12(_panel(), AggregateSpec(complete_case_policy="drop"))
