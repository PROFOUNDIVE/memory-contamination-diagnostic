from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from memcontam.manifests.phase13 import ConformanceCheck, PrefixDerivationArtifact
from memcontam.readiness.phase13_calibration_v2_runtime import execute_calibration_trajectory
from memcontam.readiness.phase13_calibration_v2_runtime_models import (
    CompletedTrajectory,
    TrajectoryRequest,
)
from memcontam.readiness.phase13_prefix_reuse import CHECKER_VERSION, CHECK_IDS, derive_prefix_windows
from memcontam.readiness.phase13_support_planning import (
    AbsentSeed,
    DeterministicSupportInput,
    PlanningError,
    ResolvedSeed,
    RoutePlanningRequest,
    StochasticSupportInput,
    UnresolvedSeed,
    clopper_pearson_lower,
    plan_route,
    plan_support,
)
from .test_phase13_calibration_v2_runtime import _fixture


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("game24", "math_equation_balancer", "word_sorting")
L1 = (
    "l1-fh_bounded-structural-support",
    "l1-rag_frozen-structural-support",
    "l1-bot_style-structural-support",
    "l1-reflexion_style-structural-support",
)
REQUIRED_L2 = (
    "l2-p01-pairwise-structural-support",
    "l2-p02-pairwise-structural-support",
    "l2-p03-pairwise-structural-support",
)


def _certificate() -> PrefixDerivationArtifact:
    import hashlib

    execution_hash = "acb769e1e1adbc3eb69e4302322c8eac81829dc836611519caea2ba960900c38"
    source_hash = "1" * 64
    checks = tuple(
        ConformanceCheck(
            check_id=check_id,
            verdict="pass",
            evidence_sha256=hashlib.sha256(
                f"{check_id}:True:{execution_hash}:{source_hash}".encode()
            ).hexdigest(),
            checker_version=CHECKER_VERSION,
            source_run_id="fixture-stream",
            source_manifest_id="fixture-stream",
            source_raw_sha256=source_hash,
        )
        for check_id in CHECK_IDS
    )
    return PrefixDerivationArtifact(
        schema_version="phase13_prefix_derivation_v2",
        conformance_id="phase13-ten-condition-prefix-v1",
        execution_registry_hash=execution_hash,
        source_raw_sha256=source_hash,
        checks=checks,
        rows=(),
    )


def _trusted_certificate() -> tuple[
    PrefixDerivationArtifact, TrajectoryRequest, CompletedTrajectory
]:
    _, _, request = _fixture()
    source = execute_calibration_trajectory(request)
    assert isinstance(source, CompletedTrajectory)
    certificate = derive_prefix_windows(request, source)
    assert isinstance(certificate, PrefixDerivationArtifact)
    return certificate, request, source


def _rows(successes: int) -> tuple[ResolvedSeed, ...]:
    return tuple(
        ResolvedSeed(seed_id=seed, passed=index < successes)
        for index, seed in enumerate(range(10000, 10012))
    )


def _required_inputs(successes: int = 12) -> tuple[StochasticSupportInput, ...]:
    return tuple(
        StochasticSupportInput(task=task, support_population_id=population, seeds=_rows(successes))
        for task in TASKS
        for population in (*L1, *REQUIRED_L2)
    )


@pytest.mark.parametrize(
    ("successes", "expected"),
    [(0, "0.000"), (1, "0.004"), (6, "0.245"), (10, "0.561"), (12, "0.779")],
)
def test_cp95_matches_independent_exact_binomial_table(successes: int, expected: str) -> None:
    assert clopper_pearson_lower(successes, 12) == Decimal(expected)


def test_cp95_floors_a_value_that_round_to_nearest_would_increase() -> None:
    assert clopper_pearson_lower(10, 12) == Decimal("0.561")


@pytest.mark.parametrize(
    ("evidence", "code"),
    [
        (StochasticSupportInput("game24", L1[0], _rows(12), method="point_estimate"), "POINT_ESTIMATE_FORBIDDEN"),
        (StochasticSupportInput("game24", L1[0], _rows(12), rounding="round_half_even"), "ROUNDING_MODE_INVALID"),
        (StochasticSupportInput("game24", L1[0], _rows(12), denominator=11), "DENOMINATOR_INVALID"),
    ],
)
def test_registered_stochastic_method_rounding_and_denominator_are_mandatory(
    evidence: StochasticSupportInput, code: str
) -> None:
    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == code


def test_self_signed_all_pass_certificate_is_rejected_as_untrusted() -> None:
    evidence = DeterministicSupportInput(
        task="game24",
        support_population_id=L1[0],
        certificate=_certificate(),
    )

    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == "CONFORMANCE_AUTHORITY_REQUIRED"


def test_trusted_todo10_recomputation_yields_deterministic_support() -> None:
    certificate, request, source = _trusted_certificate()
    evidence = DeterministicSupportInput(
        task="game24",
        support_population_id=L1[0],
        certificate=certificate,
        authority_request=request,
        authority_source=source,
    )

    result = plan_support(evidence, ROOT)

    assert result.planning_value == Decimal("1.000")
    assert result.method == "deterministic_structural"


def test_trusted_source_manifest_drift_is_rejected() -> None:
    certificate, request, source = _trusted_certificate()
    evidence = DeterministicSupportInput(
        task="game24",
        support_population_id=L1[0],
        certificate=certificate,
        authority_request=request,
        authority_source=replace(source, source_manifest_id="forged-manifest"),
    )

    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == "CONFORMANCE_SOURCE_IDENTITY_MISMATCH"


def test_trusted_source_raw_hash_drift_is_rejected() -> None:
    certificate, request, source = _trusted_certificate()
    evidence = DeterministicSupportInput(
        task="game24",
        support_population_id=L1[0],
        certificate=certificate,
        authority_request=request,
        authority_source=replace(source, source_raw_sha256="0" * 64),
    )

    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == "CONFORMANCE_SOURCE_HASH_MISMATCH"


@pytest.mark.parametrize(
    ("source_update", "code"),
    [
        ({"sealed": False}, "CONFORMANCE_SOURCE_NOT_SEALED"),
        ({"status": "invalidated"}, "CONFORMANCE_SOURCE_NOT_COMPLETED"),
    ],
)
def test_nonfinal_source_status_cannot_authorize_deterministic_support(
    source_update: dict[str, bool | str], code: str
) -> None:
    certificate, request, source = _trusted_certificate()
    evidence = DeterministicSupportInput(
        task="game24",
        support_population_id=L1[0],
        certificate=certificate,
        authority_request=request,
        authority_source=replace(source, **source_update),
    )

    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == code


def test_coordinated_analysis_hash_resigning_is_rejected() -> None:
    certificate, request, source = _trusted_certificate()
    forged_hash = "3" * 64
    verified = replace(
        request.verified,
        analysis=request.verified.analysis.model_copy(update={"registry_hash": forged_hash}),
    )
    evidence = DeterministicSupportInput(
        task="game24",
        support_population_id=L1[0],
        certificate=certificate,
        authority_request=replace(request, verified=verified),
        authority_source=replace(
            source,
            source_seal=replace(source.source_seal, analysis_registry_hash=forged_hash),
        ),
    )

    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == "CONFORMANCE_ANALYSIS_AUTHORITY_MISMATCH"


def test_resigned_conformance_mutation_is_rejected() -> None:
    certificate, request, source = _trusted_certificate()
    changed = certificate.checks[0].model_copy(
        update={"evidence_sha256": "0" * 64, "verdict": "pass"}
    )
    evidence = DeterministicSupportInput(
        task="game24",
        support_population_id=L1[0],
        certificate=certificate.model_copy(update={"checks": (changed, *certificate.checks[1:])}),
        authority_request=request,
        authority_source=source,
    )

    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == "CONFORMANCE_ARTIFACT_MISMATCH"


def test_failed_conformance_never_upgrades_deterministic_support() -> None:
    certificate, request, source = _trusted_certificate()
    check = certificate.checks[0]
    import hashlib

    failed = check.model_copy(
        update={
            "verdict": "fail",
            "evidence_sha256": hashlib.sha256(
                f"{check.check_id}:False:{certificate.execution_registry_hash}:"
                f"{certificate.source_raw_sha256}".encode()
            ).hexdigest(),
        }
    )
    evidence = DeterministicSupportInput(
        task="game24",
        support_population_id=L1[0],
        certificate=certificate.model_copy(update={"checks": (failed, *certificate.checks[1:])}),
        authority_request=request,
        authority_source=source,
    )

    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == "CONFORMANCE_ARTIFACT_MISMATCH"


@pytest.mark.parametrize(
    ("seeds", "code"),
    [
        (_rows(12)[:-1], "FINAL_SEED_ROW_COUNT_INVALID"),
        ((*_rows(12)[:-1], UnresolvedSeed(10011, "provider_outcome_unresolved")), "SEED_UNRESOLVED"),
        ((*_rows(12)[:-1], AbsentSeed(10011)), "SEED_ABSENT"),
    ],
)
def test_seed_denominator_stays_twelve_and_blocks_incomplete_states(
    seeds: tuple[ResolvedSeed | UnresolvedSeed | AbsentSeed, ...], code: str
) -> None:
    evidence = StochasticSupportInput("game24", L1[0], seeds)

    with pytest.raises(PlanningError) as caught:
        plan_support(evidence, ROOT)

    assert caught.value.code == code


def test_three_week_route_uses_maximum_ceiling_ratio_across_required_components() -> None:
    inputs = list(_required_inputs(12))
    inputs[4] = replace(inputs[4], seeds=_rows(10))

    route = plan_route(RoutePlanningRequest(route="3w", support_inputs=tuple(inputs)), ROOT)

    assert route.attempted_seeds == tuple((task, 18 if task == "game24" else 13) for task in TASKS)
    assert route.capacity.nominal_semantic_calls == (18 + 13 + 13) * 256
    assert route.capacity.raw_maximum_semantic_calls == (18 + 13 + 13) * 379
    assert route.capacity.reserved_semantic_calls == 17_510
    assert route.capacity.reserved_transport_attempts == 70_040
    assert route.capacity.maximum_input_tokens == 286_883_840
    assert route.capacity.maximum_output_tokens == 143_441_920


def test_p04_to_p06_and_l3_are_informational_for_three_week_route() -> None:
    optional = (
        StochasticSupportInput("game24", "l2-p04-pairwise-structural-support", _rows(0)),
        StochasticSupportInput("game24", "l2-p05-pairwise-structural-support", _rows(0)),
        StochasticSupportInput("game24", "l2-p06-pairwise-structural-support", _rows(0)),
        StochasticSupportInput("game24", "l3-all-primary-baselines-structural-support", _rows(0)),
    )

    route = plan_route(
        RoutePlanningRequest(route="3w", support_inputs=(*_required_inputs(), *optional)), ROOT
    )

    assert route.attempted_seeds == tuple((task, 13) for task in TASKS)
    assert set(route.informational_population_ids) == {
        item.support_population_id for item in optional
    }


@pytest.mark.parametrize(
    ("route_request", "code"),
    [
        (RoutePlanningRequest("3w", _required_inputs(), historical_route_calls=15_509), "HISTORICAL_ROUTE_MERGE_FORBIDDEN"),
        (RoutePlanningRequest("3w", _required_inputs(), budget_layer="phase12"), "BUDGET_LAYER_INVALID"),
    ],
)
def test_old_route_figures_and_budget_layers_are_rejected(
    route_request: RoutePlanningRequest, code: str
) -> None:
    with pytest.raises(PlanningError) as caught:
        plan_route(route_request, ROOT)

    assert caught.value.code == code


def test_zero_support_blocks_required_route_instead_of_using_point_estimate() -> None:
    inputs = list(_required_inputs())
    inputs[0] = replace(inputs[0], seeds=_rows(0))

    with pytest.raises(PlanningError) as caught:
        plan_route(RoutePlanningRequest("3w", tuple(inputs)), ROOT)

    assert caught.value.code == "REQUIRED_SUPPORT_ZERO"
