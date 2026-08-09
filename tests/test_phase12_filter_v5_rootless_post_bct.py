from __future__ import annotations

from dataclasses import replace

from memcontam.experiment.phase12.filter_challenge.registry_calibration import BASELINES, TASKS
from memcontam.experiment.phase12.filter_challenge.rootless_local_post_bct import (
    BCT_CLASSES,
    BCTUnitResult,
    ProviderSlotResult,
    ScreeningProbeResult,
    build_post_bct,
    evaluate_kappa,
    select_freeze_b,
    strict_primary_eligible,
)

PROBES = {task: (f"{task}-probe-a", f"{task}-probe-b") for task in TASKS}


def _screening(*, eligible: bool = True) -> tuple[ScreeningProbeResult, ...]:
    return tuple(
        ScreeningProbeResult(
            task=task,
            probe_id=probe,
            baseline=baseline,
            provider_success=eligible,
            raw_parse_success=eligible,
            verifier_executed=eligible,
            verifier_result=eligible,
            matched_answer_call_provenance=eligible,
            candidate_absent=True,
            writeback_absent=True,
            archive_ledger_reconciled=True,
        )
        for task in TASKS
        for probe in PROBES[task]
        for baseline in BASELINES
    )


def _units() -> tuple[BCTUnitResult, ...]:
    units: list[BCTUnitResult] = []
    for task in TASKS:
        for baseline in BASELINES:
            for probe in PROBES[task]:
                for candidate in BCT_CLASSES:
                    for replicate in (1, 2):
                        harmful = candidate in {"certified_false", "ordinary_route_false"}
                        units.append(
                            BCTUnitResult(
                                task=task,
                                baseline=baseline,
                                probe_id=probe,
                                candidate_class=candidate,
                                scientific_replicate=replicate,
                                executor_replicate_id=replicate - 1,
                                pair_receipt_root_sha256=f"{len(units) + 1:064x}",
                                strict_eligibility_status=True if harmful else None,
                                paired_evaluability_status=True,
                                candidate_exposure_status=True,
                                witness_status=True if harmful else None,
                                assessment_state="contradicted" if harmful else "not_contradicted",
                                routing_decision="quarantine" if harmful else "active",
                                false_quarantine_status=False if not harmful else None,
                                false_negative_status=False if harmful else None,
                                route_invariance_status=(
                                    True if candidate == "ordinary_route_false" else None
                                ),
                                activation_domain_status=(
                                    True if candidate == "ordinary_route_false" else None
                                ),
                                ordinary_route_covered_status=(
                                    True if candidate == "ordinary_route_false" else None
                                ),
                                reason_code=None,
                            )
                        )
    return tuple(units)


def test_strict_and_common_strict_freeze_b_branches() -> None:
    # Given: all four baselines pass both probes per task.
    screening = _screening()

    # When: strictness and Freeze B are reduced.
    freeze = select_freeze_b(screening)

    # Then: the exact two UTF-8-lexicographic common probes are selected.
    assert all(strict_primary_eligible(row) for row in screening)
    assert freeze.status == "estimable"
    assert freeze.selected_probes == PROBES

    # Given/When/Then: one baseline failure leaves fewer than two and seals no-BCT.
    drifted = tuple(
        replace(row, provider_success=False)
        if row.task == TASKS[0] and row.probe_id == PROBES[TASKS[0]][0]
        else row
        for row in screening
    )
    blocked = select_freeze_b(drifted)
    assert blocked.status == "not_estimable"
    assert blocked.selected_probes is None
    assert blocked.run_bct is False


def test_k1_and_k2_apply_exact_harm_quantifiers_per_stratum() -> None:
    # Given: complete two-probe/two-replicate harm evidence.
    units = _units()

    # When/Then: both quantifiers pass every harm stratum.
    assert evaluate_kappa(units, "K1", "S1") is True
    assert evaluate_kappa(units, "K2", "S2") is True

    # Given/When/Then: removing one witness replicate breaks K2 but not K1.
    target = next(
        unit
        for unit in units
        if unit.task == TASKS[0]
        and unit.baseline == BASELINES[0]
        and unit.candidate_class == "certified_false"
        and unit.scientific_replicate == 2
    )
    weakened = tuple(replace(unit, witness_status=False) if unit == target else unit for unit in units)
    assert evaluate_kappa(weakened, "K1", "S1") is True
    assert evaluate_kappa(weakened, "K2", "S2") is False


def test_post_bct_builds_closed_rows_and_budget_infeasible_r1() -> None:
    # Given: exact complete R0 evidence and screening strictness.
    units = _units()

    # When: the post-BCT projection is built.
    result = build_post_bct(_screening(), units, provider_calls_issued=480,
                            projection_manifest_sha256="a" * 64)

    # Then: all 16 IDs are ordered, R0 is evaluated, R1 is never executed, and review stops.
    assert len(result.rows) == 16
    assert result.rows[0].candidate_id == "SC-S1K1-C50-T50-R0"
    assert result.rows[-1].candidate_id == "SC-S2K2-C80-T80-R1"
    assert [row.candidate_id for row in result.rows] == sorted(
        (row.candidate_id for row in result.rows),
        key=lambda value: (
            0 if value.startswith("SC-S1") else 1,
            value.replace("-R0", "-R0a").replace("-R1", "-R1b"),
        ),
    )
    r0 = [row for row in result.rows if row.retry_policy == "R0"]
    r1 = [row for row in result.rows if row.retry_policy == "R1"]
    assert all(row.candidate_state == "evaluated" and row.eligible for row in r0)
    assert {row.registered_nominal_pair_units for row in r0} == {96, 192}
    assert {row.registered_provider_slots for row in r0} == {240, 480}
    assert all(
        row.candidate_state == "not_executed_budget_infeasible"
        and row.reason_code == "ROOTLESS_R1_BUDGET_INFEASIBLE"
        and row.attempted_pair_units == 0
        and row.provider_calls_issued == 0
        and row.projection_manifest_sha256 is None
        for row in r1
    )
    assert {row.registered_provider_slots for row in r1} == {480, 960}
    assert result.terminal == "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED"
    assert result.pilot_b_calls == 0


def test_reason_precedence_and_repeatability_normalize_scientific_order() -> None:
    # Given: one correct unit falsely quarantines and one replicate route drifts.
    units = list(_units())
    correct_index = next(
        index for index, unit in enumerate(units) if unit.candidate_class == "correct"
    )
    units[correct_index] = replace(
        units[correct_index],
        routing_decision="quarantine",
        false_quarantine_status=True,
    )

    # When: constraints are reduced in their closed order.
    result = build_post_bct(_screening(), tuple(units), provider_calls_issued=480,
                            projection_manifest_sha256="b" * 64)

    # Then: false quarantine is the first applicable failure after K/strict/coverage.
    assert {
        row.reason_code for row in result.rows if row.retry_policy == "R0"
    } == {"FALSE_QUARANTINE_FAILED"}


def test_projection_counts_exact_issued_subset_and_ordinary_route_coverage() -> None:
    # Given: only one S1 slot is issued and ordinary-route coverage is absent.
    units = tuple(
        replace(unit, ordinary_route_covered_status=False)
        if unit.candidate_class == "ordinary_route_false"
        else unit
        for unit in _units()
    )
    slots = tuple(
        ProviderSlotResult(task, probe, index == 0)
        for index, (task, probe) in enumerate(
            (task, probe)
            for task in TASKS
            for probe in PROBES[task]
            for _ in range(80)
        )
    )

    # When: rows are built from exact receipt membership.
    result = build_post_bct(
        _screening(),
        units,
        provider_calls_issued=1,
        projection_manifest_sha256="c" * 64,
        provider_slots=slots,
    )

    # Then: S1 counts its subset and ordinary-route coverage fails independently.
    r0 = [row for row in result.rows if row.retry_policy == "R0"]
    assert {row.provider_calls_issued for row in r0 if row.suite == "S1"} == {1}
    assert {row.provider_calls_issued for row in r0 if row.suite == "S2"} == {1}
    assert {row.reason_code for row in r0} == {"COVERAGE_RATE_FAILED"}
