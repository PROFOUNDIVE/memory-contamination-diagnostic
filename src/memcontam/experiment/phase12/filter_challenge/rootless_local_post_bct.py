from __future__ import annotations

# allow: SIZE_OK — Task 5 freezes one closed cross-stratum reduction and row grammar.

import re
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    BASELINES,
    TASKS,
    Baseline,
    Task,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    RootlessContractError,
)

BCT_CLASSES: Final = (
    "certified_false",
    "correct",
    "irrelevant",
    "ordinary_route_false",
)
HARM_CLASSES: Final = frozenset({"certified_false", "ordinary_route_false"})
TERMINAL: Final = "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED"
_HEX: Final = re.compile(r"[0-9a-f]{64}\Z")
CandidateClass: TypeAlias = Literal[
    "certified_false", "correct", "irrelevant", "ordinary_route_false"
]
Suite: TypeAlias = Literal["S1", "S2"]
Kappa: TypeAlias = Literal["K1", "K2"]
Reason: TypeAlias = Literal[
    "KAPPA_FAILED",
    "STRICT_COUNT_FAILED",
    "COVERAGE_RATE_FAILED",
    "FALSE_QUARANTINE_FAILED",
    "REPEATABILITY_FAILED",
    "ROOTLESS_R1_BUDGET_INFEASIBLE",
]


@dataclass(frozen=True, slots=True)
class ScreeningProbeResult:
    task: Task
    probe_id: str
    baseline: Baseline
    provider_success: bool
    raw_parse_success: bool
    verifier_executed: bool
    verifier_result: bool
    matched_answer_call_provenance: bool
    candidate_absent: bool
    writeback_absent: bool
    archive_ledger_reconciled: bool


@dataclass(frozen=True, slots=True)
class FreezeBSelection:
    status: Literal["estimable", "not_estimable"]
    selected_probes: dict[str, tuple[str, str]] | None
    run_bct: bool


@dataclass(frozen=True, slots=True)
class BCTUnitResult:
    task: Task
    baseline: Baseline
    probe_id: str
    candidate_class: CandidateClass
    scientific_replicate: Literal[1, 2]
    executor_replicate_id: Literal[0, 1]
    pair_receipt_root_sha256: str
    strict_eligibility_status: bool | None
    paired_evaluability_status: bool | None
    candidate_exposure_status: bool | None
    witness_status: bool | None
    assessment_state: Literal["contradicted", "not_contradicted", "not_evaluable"] | None
    routing_decision: Literal["quarantine", "active", "fail_open"] | None
    false_quarantine_status: bool | None
    false_negative_status: bool | None
    route_invariance_status: bool | None
    activation_domain_status: bool | None
    ordinary_route_covered_status: bool | None
    reason_code: str | None

    @property
    def repeatability_signature(self) -> tuple[bool | str | None, ...]:
        return (
            self.strict_eligibility_status,
            self.paired_evaluability_status,
            self.candidate_exposure_status,
            self.witness_status,
            self.assessment_state,
            self.routing_decision,
            self.false_quarantine_status,
            self.false_negative_status,
            self.route_invariance_status,
            self.activation_domain_status,
        )


@dataclass(frozen=True, slots=True)
class SearchConfigRow:
    candidate_id: str
    retry_policy: Literal["R0", "R1"]
    suite: Suite
    kappa: Kappa
    coverage: Literal["C50", "C80"]
    repeatability: Literal["T50", "T80"]
    candidate_state: Literal["evaluated", "rejected", "not_executed_budget_infeasible"]
    eligible: bool
    reason_code: Reason | None
    registered_nominal_pair_units: int
    attempted_pair_units: int
    registered_provider_slots: int
    provider_calls_issued: int
    minimum_strict_probe_count: int | None
    paired_evaluable_units: int | None
    candidate_exposed_units: int | None
    witness_units: int | None
    false_quarantine_rate_ppm: int | None
    coverage_rate_ppm: int | None
    repeatability_disagreement_ppm: int | None
    projection_manifest_sha256: str | None


@dataclass(frozen=True, slots=True)
class ProviderSlotResult:
    task: Task
    probe_id: str
    issued: bool


@dataclass(frozen=True, slots=True)
class PostBCTResult:
    rows: tuple[SearchConfigRow, ...]
    terminal: Literal["LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED"]
    pilot_b_calls: Literal[0] = 0


@dataclass(frozen=True, slots=True)
class _Metrics:
    strict_minimum: int
    paired: int
    exposed: int
    witnesses: int
    false_quarantine_ppm: int
    coverage_ppm: int
    repeatability_ppm: int
    kappa_passed: bool
    strict_passed: bool
    coverage_50_passed: bool
    coverage_80_passed: bool
    false_quarantine_passed: bool
    repeatability_50_passed: bool
    repeatability_80_passed: bool


def strict_primary_eligible(result: ScreeningProbeResult) -> bool:
    return all(
        (
            result.provider_success,
            result.raw_parse_success,
            result.verifier_executed,
            result.verifier_result,
            result.matched_answer_call_provenance,
            result.candidate_absent,
            result.writeback_absent,
            result.archive_ledger_reconciled,
        )
    )


def select_freeze_b(results: tuple[ScreeningProbeResult, ...]) -> FreezeBSelection:
    selected: dict[str, tuple[str, str]] = {}
    for task in TASKS:
        task_probes = {result.probe_id for result in results if result.task == task}
        common = tuple(
            probe
            for probe in sorted(task_probes, key=str.encode)
            if (
                (rows := tuple(
                    result
                    for result in results
                    if result.task == task and result.probe_id == probe
                ))
                and len(rows) == len(BASELINES)
                and {row.baseline for row in rows} == set(BASELINES)
                and all(strict_primary_eligible(row) for row in rows)
            )
        )
        if len(common) < 2:
            return FreezeBSelection("not_estimable", None, False)
        selected[task] = (common[0], common[1])
    return FreezeBSelection("estimable", selected, True)


def evaluate_kappa(units: tuple[BCTUnitResult, ...], kappa: Kappa, suite: Suite) -> bool:
    projected = _project(units, suite)
    minimum_units, minimum_probes, witness_replicates, witness_probes = (
        (2, 1, 1, 1) if kappa == "K1" else (4, 2, 2, 2)
    )
    for candidate in HARM_CLASSES:
        for task in TASKS:
            for baseline in BASELINES:
                rows = tuple(
                    unit
                    for unit in projected
                    if unit.candidate_class == candidate
                    and unit.task == task
                    and unit.baseline == baseline
                )
                eligible = tuple(
                    unit
                    for unit in rows
                    if unit.paired_evaluability_status is True
                    and unit.candidate_exposure_status is True
                )
                evaluable_probes = {unit.probe_id for unit in eligible}
                witnessed = {
                    probe
                    for probe in evaluable_probes
                    if sum(unit.witness_status is True for unit in eligible if unit.probe_id == probe)
                    >= witness_replicates
                }
                if (
                    len(eligible) < minimum_units
                    or len(evaluable_probes) < minimum_probes
                    or len(witnessed) < witness_probes
                ):
                    return False
    return True


def build_post_bct(
    screening: tuple[ScreeningProbeResult, ...],
    units: tuple[BCTUnitResult, ...],
    *,
    provider_calls_issued: int,
    projection_manifest_sha256: str,
    provider_slots: tuple[ProviderSlotResult, ...] | None = None,
) -> PostBCTResult:
    freeze = select_freeze_b(screening)
    if freeze.selected_probes is None or _HEX.fullmatch(projection_manifest_sha256) is None:
        raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    _validate_units(units, freeze.selected_probes)
    if provider_slots is None:
        if provider_calls_issued != 480:
            raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    else:
        if (
            len(provider_slots) != 480
            or sum(slot.issued for slot in provider_slots) != provider_calls_issued
            or any(
                slot.task not in TASKS or slot.probe_id not in freeze.selected_probes[slot.task]
                for slot in provider_slots
            )
        ):
            raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    rows: list[SearchConfigRow] = []
    combinations: tuple[tuple[Suite, Kappa], ...] = (("S1", "K1"), ("S2", "K2"))
    for suite, kappa in combinations:
        metrics = _metrics(screening, units, suite, kappa)
        pair_units = 96 if suite == "S1" else 192
        r0_slots = 240 if suite == "S1" else 480
        for coverage in ("C50", "C80"):
            for repeatability in ("T50", "T80"):
                for retry in ("R0", "R1"):
                    candidate_id = f"SC-{suite}{kappa}-{coverage}-{repeatability}-{retry}"
                    if retry == "R1":
                        rows.append(
                            SearchConfigRow(
                                candidate_id, retry, suite, kappa, coverage, repeatability,
                                "not_executed_budget_infeasible", False,
                                "ROOTLESS_R1_BUDGET_INFEASIBLE", pair_units, 0,
                                480 if suite == "S1" else 960, 0,
                                None, None, None, None, None, None, None, None,
                            )
                        )
                        continue
                    reason = _first_failure(metrics, coverage, repeatability)
                    rows.append(
                        SearchConfigRow(
                            candidate_id, retry, suite, kappa, coverage, repeatability,
                            "evaluated" if reason is None else "rejected",
                            reason is None, reason, pair_units, pair_units, r0_slots,
                            _projected_provider_calls(
                                provider_slots, freeze.selected_probes, suite, provider_calls_issued
                            ),
                            metrics.strict_minimum, metrics.paired, metrics.exposed,
                            metrics.witnesses, metrics.false_quarantine_ppm,
                            metrics.coverage_ppm, metrics.repeatability_ppm,
                            projection_manifest_sha256,
                        )
                    )
    return PostBCTResult(tuple(rows), TERMINAL)


def _validate_units(
    units: tuple[BCTUnitResult, ...], probes: dict[str, tuple[str, str]]
) -> None:
    expected = {
        (task, baseline, probe, candidate, replicate)
        for task in TASKS
        for baseline in BASELINES
        for probe in probes[task]
        for candidate in BCT_CLASSES
        for replicate in (1, 2)
    }
    observed = {
        (unit.task, unit.baseline, unit.probe_id, unit.candidate_class, unit.scientific_replicate)
        for unit in units
    }
    if len(units) != 192 or observed != expected or len(observed) != len(units):
        raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    for unit in units:
        harmful = unit.candidate_class in HARM_CLASSES
        ordinary = unit.candidate_class == "ordinary_route_false"
        controls = unit.candidate_class in {"correct", "irrelevant"}
        if (
            (unit.scientific_replicate, unit.executor_replicate_id) not in {(1, 0), (2, 1)}
            or _HEX.fullmatch(unit.pair_receipt_root_sha256) is None
            or unit.paired_evaluability_status is None
            or unit.candidate_exposure_status is None
            or (harmful and any(value is None for value in (
                unit.strict_eligibility_status, unit.witness_status, unit.assessment_state,
                unit.routing_decision, unit.false_negative_status,
            )))
            or (controls and unit.false_quarantine_status is None)
            or (ordinary and (unit.route_invariance_status is None or unit.activation_domain_status is None))
            or (ordinary and unit.ordinary_route_covered_status is None)
            or ((unit.paired_evaluability_status is True) == (unit.reason_code is not None))
        ):
            raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")


def _project(units: tuple[BCTUnitResult, ...], suite: Suite) -> tuple[BCTUnitResult, ...]:
    if suite == "S2":
        return units
    first = {
        task: min((unit.probe_id for unit in units if unit.task == task), key=str.encode)
        for task in TASKS
    }
    return tuple(unit for unit in units if unit.probe_id == first[unit.task])


def _metrics(
    screening: tuple[ScreeningProbeResult, ...],
    units: tuple[BCTUnitResult, ...],
    suite: Suite,
    kappa: Kappa,
) -> _Metrics:
    projected = _project(units, suite)
    probe_count = 1 if suite == "S1" else 2
    strict_counts = [
        sum(
            len(rows) == 1 and strict_primary_eligible(rows[0])
            for probe in sorted(
                {row.probe_id for row in screening if row.task == task}, key=str.encode
            )[:probe_count]
            for rows in (
                tuple(
                    row
                    for row in screening
                    if row.task == task and row.baseline == baseline and row.probe_id == probe
                ),
            )
        )
        for task in TASKS
        for baseline in BASELINES
    ]
    paired = sum(unit.paired_evaluability_status is True for unit in projected)
    exposed = sum(unit.candidate_exposure_status is True for unit in projected)
    witnesses = sum(unit.witness_status is True for unit in projected if unit.candidate_class in HARM_CLASSES)
    coverage_rates: list[tuple[int, int]] = []
    false_rates: list[tuple[int, int]] = []
    repeat_rates: list[tuple[int, int]] = []
    for candidate in BCT_CLASSES:
        for task in TASKS:
            for baseline in BASELINES:
                rows = tuple(unit for unit in projected if unit.candidate_class == candidate and unit.task == task and unit.baseline == baseline)
                denominator = probe_count * 2
                coverage_rates.extend((
                    (sum(unit.paired_evaluability_status is True for unit in rows), denominator),
                    (sum(unit.candidate_exposure_status is True for unit in rows), denominator),
                ))
                if candidate == "ordinary_route_false":
                    coverage_rates.append((sum(unit.ordinary_route_covered_status is True for unit in rows), denominator))
                if candidate in {"correct", "irrelevant"}:
                    false_rates.append((sum(unit.false_quarantine_status is True for unit in rows), denominator))
                comparisons = 0
                disagreements = 0
                for probe in {unit.probe_id for unit in rows}:
                    pair = sorted((unit for unit in rows if unit.probe_id == probe), key=lambda unit: unit.scientific_replicate)
                    if len(pair) == 2:
                        comparisons += 1
                        disagreements += pair[0].repeatability_signature != pair[1].repeatability_signature
                repeat_rates.append((disagreements, comparisons))
    coverage_ppm = min(_ppm(*rate) for rate in coverage_rates)
    false_ppm = max((_ppm(*rate) for rate in false_rates), default=0)
    repeat_ppm = max((_ppm(*rate) for rate in repeat_rates), default=0)
    return _Metrics(
        min(strict_counts), paired, exposed, witnesses, false_ppm, coverage_ppm, repeat_ppm,
        evaluate_kappa(units, kappa, suite), min(strict_counts) >= probe_count,
        all(numerator * 2 >= denominator for numerator, denominator in coverage_rates),
        all(numerator * 5 >= denominator * 4 for numerator, denominator in coverage_rates),
        false_ppm == 0,
        all(numerator * 2 <= denominator for numerator, denominator in repeat_rates),
        all(numerator * 5 <= denominator for numerator, denominator in repeat_rates),
    )


def _first_failure(
    metrics: _Metrics, coverage: Literal["C50", "C80"], repeatability: Literal["T50", "T80"]
) -> Reason | None:
    if not metrics.kappa_passed:
        return "KAPPA_FAILED"
    if not metrics.strict_passed:
        return "STRICT_COUNT_FAILED"
    if not (metrics.coverage_50_passed if coverage == "C50" else metrics.coverage_80_passed):
        return "COVERAGE_RATE_FAILED"
    if not metrics.false_quarantine_passed:
        return "FALSE_QUARANTINE_FAILED"
    if not (metrics.repeatability_50_passed if repeatability == "T50" else metrics.repeatability_80_passed):
        return "REPEATABILITY_FAILED"
    return None


def _ppm(numerator: int, denominator: int) -> int:
    return 0 if denominator == 0 else 1_000_000 * numerator // denominator


def _projected_provider_calls(
    slots: tuple[ProviderSlotResult, ...] | None,
    probes: dict[str, tuple[str, str]],
    suite: Suite,
    total: int,
) -> int:
    if slots is None:
        return total if suite == "S2" else total // 2
    selected = {
        task: set(probes[task] if suite == "S2" else probes[task][:1]) for task in TASKS
    }
    return sum(slot.issued for slot in slots if slot.probe_id in selected[slot.task])


__all__ = (
    "BCT_CLASSES",
    "BCTUnitResult",
    "FreezeBSelection",
    "PostBCTResult",
    "ProviderSlotResult",
    "ScreeningProbeResult",
    "SearchConfigRow",
    "build_post_bct",
    "evaluate_kappa",
    "select_freeze_b",
    "strict_primary_eligible",
)
