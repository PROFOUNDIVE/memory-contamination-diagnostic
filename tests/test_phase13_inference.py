from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from random import Random
from statistics import mean

import pytest

from memcontam.readiness.phase13_calibration_v2_runtime import execute_calibration_trajectory
from memcontam.readiness.phase13_calibration_v2_runtime_models import CompletedTrajectory
from memcontam.manifests.phase13 import PrefixDerivationArtifact
from memcontam.readiness.phase13_prefix_reuse import derive_prefix_windows
from test_phase13_calibration_v2_runtime import _fixture


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
POPULATIONS = (
    *(f"l1-{baseline}-structural-support" for baseline in BASELINES),
    *(f"l2-p0{index}-pairwise-structural-support" for index in range(1, 4)),
)


@pytest.fixture(autouse=True)
def _authenticated_fixture_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    original = __import__(
        "memcontam.evaluation.phase13_inference", fromlist=["authenticate_conformance"]
    ).authenticate_conformance

    def authenticate(prefix, request, source) -> None:  # noqa: ANN001
        if request.seed_id == 10000:
            original(prefix, request, source)

    monkeypatch.setattr(
        "memcontam.evaluation.phase13_inference.authenticate_conformance",
        authenticate,
    )


def _request():  # noqa: ANN202
    from memcontam.evaluation.phase13_inference import (
        ArmTrajectory,
        SeedBundle,
        SlotSupport,
        TaskInferenceInput,
    )

    effects = (
        (0.2, 0.4, 0.6, 0.8),
        (0.0, 0.2, 0.4, 0.6),
        (0.4, 0.6, 0.8, 1.0),
    )
    _, _, base_request = _fixture()
    source = execute_calibration_trajectory(base_request)
    assert isinstance(source, CompletedTrajectory)
    certificate = derive_prefix_windows(base_request, source)
    assert isinstance(certificate, PrefixDerivationArtifact)
    seeds = []
    for offset, seed_effects in enumerate(effects):
        seed_id = 10000 + offset
        suffix = next(
            row
            for stream in base_request.verified.execution.task_streams
            if stream.task == "game24"
            for row in stream.suffixes
            if row.seed_id == seed_id
        )
        task_ids = base_request.verified.ordered_suffixes[("game24", seed_id)]
        request = replace(
            base_request,
            stream_id=f"game24-seed-{seed_id}",
            seed_id=seed_id,
            source_ordered_stream_sha256=suffix.source_ordered_stream_sha256,
            session_id=f"session-{seed_id}",
            contexts=tuple(
                replace(context, task=context.task.model_copy(update={"sample_id": sample_id}))
                for context, sample_id in zip(base_request.contexts, task_ids, strict=True)
            ),
        )
        seeds.append(
            SeedBundle(
                request=request,
                source=source,
                prefix=certificate,
                arms=tuple(
                    ArmTrajectory(
                        baseline=baseline,
                        clean=(1, 1, 1, 1, 1),
                        contam=tuple(
                            0 if index < round(effect * 5) else 1 for index in range(5)
                        ),
                    )
                    for baseline, effect in zip(BASELINES, seed_effects, strict=True)
                ),
            )
        )
    return TaskInferenceInput(
        task="game24",
        seeds=tuple(seeds),
        support=tuple(SlotSupport(population, True) for population in POPULATIONS),
    )


def _expected_effects() -> tuple[tuple[float, ...], ...]:
    baseline = (
        (0.2, 0.0, 0.4),
        (0.4, 0.2, 0.6),
        (0.6, 0.4, 0.8),
        (0.8, 0.6, 1.0),
    )
    return (
        *baseline,
        *(tuple(left - right for left, right in zip(baseline[0], row, strict=True))
          for row in baseline[1:]),
    )


def _independent_statistics() -> tuple[tuple[float, float, float, float], ...]:
    effects = _expected_effects()
    generator = Random(13)
    boot = [[] for _ in effects]
    null = [[] for _ in effects]
    observed = tuple(mean(values) for values in effects)
    for _ in range(20_000):
        indices = tuple(generator.randrange(3) for _ in range(3))
        for slot, values in enumerate(effects):
            boot[slot].append(mean(values[index] for index in indices))
            null[slot].append(mean(values[index] - observed[slot] for index in indices))
    rows = []
    for slot, values in enumerate(effects):
        ordered = sorted(boot[slot])
        p_value = (1 + sum(abs(value) >= abs(observed[slot]) for value in null[slot])) / 20_001
        rows.append((observed[slot], ordered[500], ordered[19_499], p_value))
    return tuple(rows)


def test_primary_family_matches_independent_joint_seed_calculation() -> None:
    from memcontam.evaluation.phase13_inference import infer_primary

    result = infer_primary(_request(), ROOT)
    expected = _independent_statistics()

    assert len(result.slots) == 7
    assert tuple(row.order for row in result.slots) == tuple(range(1, 8))
    for row, expected_row in zip(result.slots, expected, strict=True):
        assert (row.estimate, row.lower, row.upper, row.p_value) == pytest.approx(expected_row)
    assert result.family_id == "game24-h5-primary-holm-v1"
    assert result.interval_id == "main-paired-seed-bootstrap95-v1"
    assert result.replicates == 20_000
    assert result.rng_seed == 13


def test_primary_output_is_byte_deterministic_and_holm_is_hand_computed() -> None:
    from memcontam.evaluation.phase13_inference import infer_primary

    first = infer_primary(_request(), ROOT)
    second = infer_primary(_request(), ROOT)
    encoded = lambda result: json.dumps(  # noqa: E731
        [tuple(getattr(row, field.name) for field in fields(row)) for row in result.slots],
        separators=(",", ":"),
    ).encode()

    assert encoded(first) == encoded(second)
    ordered = sorted(enumerate(row.p_value for row in first.slots), key=lambda item: (item[1], item[0]))
    running = 0.0
    expected_adjusted = [0.0] * 7
    expected_reject = [False] * 7
    active = True
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (7 - rank) * p_value))
        expected_adjusted[index] = running
        active = active and p_value <= 0.05 / (7 - rank)
        expected_reject[index] = active
    assert [row.holm_adjusted_p for row in first.slots] == expected_adjusted
    assert [row.reject_null for row in first.slots] == expected_reject


def test_unsupported_slot_retains_fixed_family_membership() -> None:
    from memcontam.evaluation.phase13_inference import SlotSupport, infer_primary

    request = _request()
    support = list(request.support)
    support[5] = SlotSupport(POPULATIONS[5], False, "pair support absent")

    result = infer_primary(replace(request, support=tuple(support)), ROOT)

    row = result.slots[5]
    assert len(result.slots) == 7
    assert row.status == "NOT_ESTIMABLE"
    assert row.reason == "pair support absent"
    assert row.p_value == 1.0
    assert row.reject_null is False
    assert row.decision == "NOT_REJECTED"
    assert row.holm_adjusted_p == 1.0


def test_non_primary_window_has_estimation_fields_only_and_separate_interval() -> None:
    from memcontam.evaluation.phase13_inference import estimate_non_primary

    row = estimate_non_primary("accuracy-h10-sensitivity", (0.0, 0.5, 1.0), ROOT)

    assert row.analysis_window_id == "accuracy-h10-sensitivity"
    assert row.inference_status == "estimation_only"
    assert row.interval_id == "main-paired-seed-bootstrap95-v1"
    assert row.interval_id != "support-planning-cp95-one-sided-v1"
    assert not {"p_value", "reject_null", "holm_adjusted_p"} & {field.name for field in fields(row)}


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda request: replace(request, task="word_sorting"), "INFERENCE_TASK_SOURCE_MISMATCH"),
        (lambda request: replace(request, seeds=(request.seeds[0], request.seeds[0])), "DUPLICATE_TRAJECTORY_SEED"),
        (lambda request: replace(request, seeds=request.seeds[:1]), "INCOMPLETE_SEED_DENOMINATOR"),
        (lambda request: replace(request, support=request.support[:-1]), "INFERENCE_SLOT_INVENTORY_INVALID"),
        (lambda request: replace(request, support=tuple(reversed(request.support))), "INFERENCE_SLOT_ORDER_INVALID"),
    ],
)
def test_adversarial_input_mutations_raise_named_errors(mutate, code: str) -> None:  # noqa: ANN001
    from memcontam.evaluation.phase13_inference import InferenceError, infer_primary

    with pytest.raises(InferenceError) as caught:
        infer_primary(mutate(_request()), ROOT)

    assert caught.value.code == code


def test_incomplete_arm_or_horizon_cannot_be_resampled() -> None:
    from memcontam.evaluation.phase13_inference import InferenceError, infer_primary

    request = _request()
    bad_arm = replace(request.seeds[0].arms[0], clean=(1, 1, 1, 1))
    bad_seed = replace(request.seeds[0], arms=(bad_arm, *request.seeds[0].arms[1:]))
    with pytest.raises(InferenceError) as caught:
        infer_primary(replace(request, seeds=(bad_seed, *request.seeds[1:])), ROOT)
    assert caught.value.code == "INCOMPLETE_H5_TRAJECTORY"


def test_source_authority_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from memcontam.evaluation.phase13_inference import InferenceError, infer_primary
    from memcontam.readiness.phase13_support_authority import authenticate_conformance

    monkeypatch.setattr(
        "memcontam.evaluation.phase13_inference.authenticate_conformance",
        authenticate_conformance,
    )
    request = _request()
    mutated = request.seeds[0].prefix.model_copy(update={"source_raw_sha256": "0" * 64})
    seed = replace(request.seeds[0], prefix=mutated)

    with pytest.raises(InferenceError) as caught:
        infer_primary(replace(request, seeds=(seed, *request.seeds[1:])), ROOT)

    assert caught.value.code == "INFERENCE_SOURCE_AUTHORITY_INVALID"
