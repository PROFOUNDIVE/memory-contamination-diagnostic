from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Self

from pydantic import model_validator

from memcontam.experiment.phase12.filter_challenge.mft_safety import (
    MFT_SAFETY_IDS,
    build_mft_safety_report,
)
from memcontam.experiment.phase12.filter_challenge.mft_safety_types import (
    MftExecutionCount,
    MftSafetyReport,
)
from memcontam.experiment.phase12.filter_challenge.mft_state import (
    MFT_STATE_IDS,
    MftStateContext,
    MftStateReport,
    run_mft_state_gates,
)
from memcontam.experiment.phase12.filter_challenge.registry import (
    validate_registry_closure,
)
from memcontam.experiment.phase12.filter_challenge.registry_common import StrictRegistry
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


MFT_IDS: Final = (
    "MFT-FV5-01-PAIR-MATCH",
    "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE",
    "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE",
    "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT",
    "MFT-FV5-08-NO-WRITEBACK",
    "MFT-FV5-09-CONTAM-SHADOW-SHARE",
    "MFT-FV5-10-PARSER-BOUNDARY",
    "MFT-FV5-11-CONTROL-CACHE",
    "MFT-FV5-12-PROBE-KEY-INVARIANCE",
    "MFT-FV5-13-ANSWER-CALL-PROVENANCE",
    "MFT-FV5-14-ACTIVATION-DOMAIN",
    "MFT-FV5-15-ELIGIBILITY-STATES",
    "MFT-FV5-16-COVERAGE-NOT-ESTIMABLE",
)


@dataclass(frozen=True, slots=True)
class MftReportError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


class MergedMftReport(StrictRegistry):
    schema_version: Literal["filter_challenge_mft_v1"] = "filter_challenge_mft_v1"
    evidence_layer: Literal["build"] = "build"
    scientific_result: Literal[False] = False
    fixture_only: Literal[True] = True
    search_config_id: str
    search_config_hash: str
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str
    operational_probe_suite_manifest_id: str
    operational_probe_suite_manifest_hash: str
    ordered_test_ids: tuple[str, ...]
    execution_counts: tuple[MftExecutionCount, ...]
    state_report: MftStateReport
    safety_report: MftSafetyReport
    all_passed: bool
    provider_calls_issued: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_merged_registry(self) -> Self:
        if (
            self.ordered_test_ids != MFT_IDS
            or MFT_STATE_IDS + MFT_SAFETY_IDS != MFT_IDS
            or tuple(item.test_id for item in self.execution_counts) != MFT_IDS
            or any(item.count != 1 for item in self.execution_counts)
            or self.safety_report.test_ids != MFT_SAFETY_IDS
            or tuple(case.test_id for case in self.safety_report.cases) != MFT_SAFETY_IDS
            or tuple(
                (item.test_id, item.count) for item in self.safety_report.execution_counts
            )
            != tuple((test_id, 1) for test_id in MFT_SAFETY_IDS)
        ):
            raise MftReportError("MFT_REGISTRY_MISMATCH")
        passed = all(result.status == "pass" for result in self.state_report.results)
        passed = passed and self.safety_report.all_passed and all(
            case.status == "pass" and case.reason_code is None
            for case in self.safety_report.cases
        )
        if self.all_passed != passed:
            raise MftReportError("MFT_STATUS_MISMATCH")
        return self


def build_mft_report(
    search: SearchConfig,
    inventory: ProbeInventoryRegistry,
    suite: OperationalSuiteRegistry,
) -> MergedMftReport:
    closure = validate_registry_closure(search, inventory, suite)
    context = MftStateContext(
        search_config_id=closure.search_config_id,
        search_config_hash=search.search_config_hash,
        calibration_probe_inventory_id=closure.calibration_probe_inventory_id,
        calibration_probe_inventory_manifest_hash=(
            closure.calibration_probe_inventory_manifest_hash
        ),
        operational_probe_suite_manifest_id=closure.operational_probe_suite_manifest_id,
        operational_probe_suite_manifest_hash=closure.operational_probe_suite_manifest_hash,
        suite_candidate=search.suite_candidates[0],
        kappa_candidate=search.kappa_candidates[0],
    )
    state = run_mft_state_gates(context)
    safety = build_mft_safety_report()
    report = MergedMftReport(
        search_config_id=search.registry_id,
        search_config_hash=search.search_config_hash,
        calibration_probe_inventory_id=inventory.registry_id,
        calibration_probe_inventory_manifest_hash=inventory.calibration_probe_inventory_manifest_hash,
        operational_probe_suite_manifest_id=suite.registry_id,
        operational_probe_suite_manifest_hash=suite.operational_probe_suite_manifest_hash,
        ordered_test_ids=MFT_IDS,
        execution_counts=tuple(
            MftExecutionCount(test_id=test_id, count=1) for test_id in MFT_IDS
        ),
        state_report=state,
        safety_report=safety,
        all_passed=(
            all(result.status == "pass" for result in state.results) and safety.all_passed
        ),
    )
    if not report.all_passed:
        raise MftReportError("MFT_GATE_FAILED")
    return report


__all__ = ("MFT_IDS", "MergedMftReport", "MftReportError", "build_mft_report")
