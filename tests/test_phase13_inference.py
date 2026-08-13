from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from random import Random
from statistics import mean

import pytest

from memcontam.manifests.phase13 import PrefixDerivationArtifact
from memcontam.readiness.phase13_calibration_v2_runtime import execute_calibration_trajectory
from memcontam.readiness.phase13_calibration_v2_runtime_models import CompletedTrajectory
from memcontam.readiness.phase13_prefix_reuse import derive_prefix_windows
from test_phase13_calibration_v2_runtime import _fixture


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(10000, 10012))
BASELINES = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
POPULATIONS = (
    *(f"l1-{baseline}-structural-support" for baseline in BASELINES),
    *(f"l2-p0{index}-pairwise-structural-support" for index in range(1, 4)),
)


def _seed_bundle(seed_id: int, template):  # noqa: ANN001, ANN202
    from memcontam.evaluation.phase13_inference import SeedBundle

    request, source, prefix = template
    suffix = next(
        row
        for stream in request.verified.execution.task_streams
        if stream.task == request.task
        for row in stream.suffixes
        if row.seed_id == seed_id
    )
    task_ids = request.verified.ordered_suffixes[(request.task, seed_id)]
    bound_request = replace(
        request,
        stream_id=f"{request.task}-seed-{seed_id}",
        seed_id=seed_id,
        source_ordered_stream_sha256=suffix.source_ordered_stream_sha256,
        session_id=f"session-{seed_id}",
        contexts=tuple(
            replace(context, task=context.task.model_copy(update={"sample_id": sample_id}))
            for context, sample_id in zip(request.contexts, task_ids, strict=True)
        ),
    )
    bound_source = replace(
        source,
        stream_id=bound_request.stream_id,
        events=tuple(replace(event, session_id=bound_request.session_id) for event in source.events),
        source_manifest_id=bound_request.stream_id,
        source_seal=replace(source.source_seal, source_manifest_id=bound_request.stream_id),
    )
    bound_prefix = prefix.model_copy(
        update={
            "checks": tuple(
                check.model_copy(update={"source_run_id": bound_request.stream_id,
                                         "source_manifest_id": bound_request.stream_id})
                for check in prefix.checks
            )
        }
    )
    return SeedBundle(bound_request, bound_source, bound_prefix)


@pytest.fixture(autouse=True)
def _test_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "memcontam.evaluation.phase13_inference.authenticate_conformance",
        lambda _prefix, _request, _source: None,
    )


def _request():  # noqa: ANN202
    from memcontam.evaluation.phase13_inference import SlotSupport, TaskInferenceInput

    _, _, request = _fixture()
    source = execute_calibration_trajectory(request)
    assert isinstance(source, CompletedTrajectory)
    prefix = derive_prefix_windows(request, source)
    assert isinstance(prefix, PrefixDerivationArtifact)
    template = request, source, prefix
    return TaskInferenceInput(
        "game24",
        tuple(_seed_bundle(seed, template) for seed in SEEDS),
        tuple(SlotSupport(population, True) for population in POPULATIONS),
    )


def test_requires_exact_registered_twelve_seed_inventory() -> None:
    from memcontam.evaluation.phase13_inference import InferenceError, infer_primary

    request = _request()
    for seeds, code in (
        (request.seeds[:-1], "INCOMPLETE_SEED_DENOMINATOR"),
        ((*request.seeds[:-1], request.seeds[0]), "DUPLICATE_TRAJECTORY_SEED"),
        ((replace(request.seeds[0], request=replace(request.seeds[0].request, seed_id=99999)),
          *request.seeds[1:]), "TRAJECTORY_SEED_INVENTORY_INVALID"),
    ):
        with pytest.raises(InferenceError) as caught:
            infer_primary(replace(request, seeds=tuple(seeds)), ROOT)
        assert caught.value.code == code


def test_rejects_cross_task_and_permuted_source_rows() -> None:
    from memcontam.evaluation.phase13_inference import InferenceError, infer_primary

    request = _request()
    cross_task = replace(request.seeds[0].source.events[0], task="word_sorting")
    permuted = replace(request.seeds[0].source, events=(request.seeds[0].source.events[1], cross_task,
                                                        *request.seeds[0].source.events[2:]))
    with pytest.raises(InferenceError) as caught:
        infer_primary(replace(request, seeds=(replace(request.seeds[0], source=permuted),
                                             *request.seeds[1:])), ROOT)
    assert caught.value.code == "INFERENCE_SOURCE_ROWS_INVALID"


def test_rejects_fabricated_verified_outcome() -> None:
    from memcontam.evaluation.phase13_inference import InferenceError, infer_primary

    request = _request()
    fabricated = replace(request.seeds[0].source.events[0], verified_score=2)
    source = replace(request.seeds[0].source,
                     events=(fabricated, *request.seeds[0].source.events[1:]))
    with pytest.raises(InferenceError) as caught:
        infer_primary(replace(request, seeds=(replace(request.seeds[0], source=source),
                                             *request.seeds[1:])), ROOT)
    assert caught.value.code == "INFERENCE_OUTCOME_INVALID"


def test_input_model_has_no_caller_authored_arm_rows() -> None:
    from dataclasses import fields
    from memcontam.evaluation.phase13_inference import SeedBundle

    assert tuple(field.name for field in fields(SeedBundle)) == ("request", "source", "prefix")


def test_primary_family_uses_source_rows_and_fixed_slots() -> None:
    from memcontam.evaluation.phase13_inference import infer_primary

    result = infer_primary(_request(), ROOT)

    assert len(result.slots) == 7
    assert tuple(row.order for row in result.slots) == tuple(range(1, 8))
    assert all(row.estimate == 0.0 for row in result.slots)
    assert result.family_id == "game24-h5-primary-holm-v1"
    assert result.replicates == 20_000
    assert result.rng_seed == 13


def test_percentile_uses_declared_upper_order_statistic() -> None:
    from memcontam.evaluation.phase13_inference import _statistics

    values = (
        0.8444218515250481, 0.7579544029403025, 0.420571580830845,
        0.25891675029296335, 0.5112747213686085, 0.4049341374504143,
        0.7837985890347726, 0.30331272607892745, 0.4765969541523558,
        0.5833820394550312, 0.9081128851953352, 0.5046868558173903,
    )
    generator = Random(13)
    ordered = sorted(
        mean(values[generator.randrange(12)] for _ in range(12)) for _ in range(20_000)
    )

    assert ordered[19_499] != ordered[19_500]
    assert _statistics((values,))[0][2] == ordered[19_500]


def test_unsupported_slot_remains_in_seven_member_holm_family() -> None:
    from memcontam.evaluation.phase13_inference import SlotSupport, infer_primary

    request = _request()
    support = list(request.support)
    support[5] = SlotSupport(POPULATIONS[5], False, "pair support absent")
    row = infer_primary(replace(request, support=tuple(support)), ROOT).slots[5]

    assert row.status == "NOT_ESTIMABLE"
    assert row.p_value == row.holm_adjusted_p == 1.0
    assert row.reject_null is False
    assert row.reason == "pair support absent"


def test_non_primary_window_has_no_rejection_fields() -> None:
    from dataclasses import fields
    from memcontam.evaluation.phase13_inference import estimate_non_primary

    row = estimate_non_primary("accuracy-h10-sensitivity", tuple(float(x) for x in range(12)), ROOT)

    assert row.interval_id == "main-paired-seed-bootstrap95-v1"
    assert row.interval_id != "support-planning-cp95-one-sided-v1"
    assert not {"p_value", "reject_null", "holm_adjusted_p"} & {field.name for field in fields(row)}
