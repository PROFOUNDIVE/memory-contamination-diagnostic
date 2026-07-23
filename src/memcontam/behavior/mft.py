from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from memcontam.behavior.registry import BehaviorRegistryBundle
from memcontam.experiment.phase12.contracts import MftManifest, canonical_json_hash


__all__ = [
    "MftBaselineCondition",
    "MftCase",
    "MftCaseExecution",
    "MftCaseResult",
    "MftError",
    "MftEventLedger",
    "MftExecutionContext",
    "MftSuiteResult",
    "freeze_mft_manifest",
    "run_mft_suite",
]


class MftError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class MftBaselineCondition:
    condition_id: str
    mode: Literal["nomem", "memory"]

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise MftError("INVALID_MFT_CONDITION")


@dataclass(frozen=True)
class MftCase:
    case_id: str
    test_id: str
    task_family: str
    baseline_condition_id: str
    baseline_mode: Literal["nomem", "memory"]


@dataclass(frozen=True)
class MftCaseExecution:
    status: Literal["pass", "model_failure", "contract_failure"]
    raw_log_ids: tuple[str, ...]
    pilot_allowance_units: int
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.raw_log_ids
            or any(not raw_log_id for raw_log_id in self.raw_log_ids)
            or self.pilot_allowance_units < 0
        ):
            raise MftError("INVALID_MFT_CASE_EXECUTION")


@dataclass(frozen=True)
class MftCaseResult:
    case: MftCase
    status: Literal["pass", "model_behavior", "contract_failure"]
    raw_log_ids: tuple[str, ...]
    pilot_allowance_units: int
    failure_code: str | None
    attempted: Literal[True] = True

    @property
    def case_id(self) -> str:
        return self.case.case_id

    @property
    def test_id(self) -> str:
        return self.case.test_id


@dataclass(frozen=True)
class MftExecutionContext:
    task_families: tuple[str, ...]
    baseline_conditions: tuple[MftBaselineCondition, ...]
    execute_case: Callable[[MftCase], MftCaseExecution]

    def __post_init__(self) -> None:
        if (
            not self.task_families
            or any(not task_family for task_family in self.task_families)
            or len(set(self.task_families)) != len(self.task_families)
            or not self.baseline_conditions
            or len({condition.condition_id for condition in self.baseline_conditions})
            != len(self.baseline_conditions)
            or not callable(self.execute_case)
        ):
            raise MftError("INVALID_MFT_EXECUTION_CONTEXT")


@dataclass(frozen=True)
class MftSuiteResult:
    expected_case_ids: tuple[str, ...]
    cases: tuple[MftCaseResult, ...]

    def __post_init__(self) -> None:
        if len(set(self.expected_case_ids)) != len(self.expected_case_ids) or len(
            {case.case_id for case in self.cases}
        ) != len(self.cases):
            raise MftError("DUPLICATE_MFT_CASE")

    @property
    def all_registered_cases_attempted(self) -> bool:
        return set(self.expected_case_ids) == {case.case_id for case in self.cases} and all(
            case.attempted for case in self.cases
        )

    @property
    def mft04_status(self) -> Literal["pass", "fail"]:
        return (
            "fail"
            if any(case.test_id == "MFT-04" and case.status == "contract_failure" for case in self.cases)
            else "pass"
        )

    @property
    def route_gate_status(self) -> Literal["pass", "blocked"]:
        return "blocked" if self.mft04_status == "fail" else "pass"

    @property
    def pilot_allowance_units(self) -> int:
        return sum(case.pilot_allowance_units for case in self.cases)


class MftEventLedger:
    def __init__(self) -> None:
        self.cases: list[MftCaseResult] = []

    def append_mft_case(self, case: MftCaseResult) -> None:
        self.cases.append(case)


def run_mft_suite(
    bundle: BehaviorRegistryBundle,
    execution_context: MftExecutionContext,
    writer: MftEventLedger | object,
) -> MftSuiteResult:
    if not isinstance(bundle, BehaviorRegistryBundle) or not isinstance(
        execution_context, MftExecutionContext
    ):
        raise MftError("INVALID_MFT_EXECUTION_CONTEXT")

    cases = _generate_cases(bundle, execution_context)
    results: list[MftCaseResult] = []
    for case in cases:
        try:
            execution = execution_context.execute_case(case)
        except Exception as error:
            raise MftError("MFT_CASE_EXECUTION_FAILED") from error
        if not isinstance(execution, MftCaseExecution):
            raise MftError("INVALID_MFT_CASE_EXECUTION")
        result = _classify_case(case, execution)
        _append_case(writer, result)
        results.append(result)
    return MftSuiteResult(tuple(case.case_id for case in cases), tuple(results))


def freeze_mft_manifest(
    result: MftSuiteResult,
    case_status_ledger_hash: str,
    pilot_allowance_ledger_hash: str,
    frozen_at: str,
) -> MftManifest:
    if not isinstance(result, MftSuiteResult) or not result.all_registered_cases_attempted:
        raise MftError("MFT_CASE_COVERAGE_INCOMPLETE")
    if not all((case_status_ledger_hash, pilot_allowance_ledger_hash, frozen_at)):
        raise MftError("INVALID_MFT_MANIFEST_INPUT")
    payload = {
        "all_registered_cases_attempted": True,
        "case_status_ledger_hash": case_status_ledger_hash,
        "frozen_at": frozen_at,
        "mft04_status": result.mft04_status,
        "pilot_allowance_ledger_hash": pilot_allowance_ledger_hash,
        "route_gate_status": result.route_gate_status,
    }
    artifact_hash = canonical_json_hash(payload)
    return MftManifest(
        manifest_id=f"mft-{artifact_hash[:12]}",
        artifact_hash=artifact_hash,
        all_registered_cases_attempted=True,
        mft04_status=result.mft04_status,
        route_gate_status=result.route_gate_status,
        case_status_ledger_hash=case_status_ledger_hash,
        pilot_allowance_ledger_hash=pilot_allowance_ledger_hash,
        frozen_at=frozen_at,
    )


def _generate_cases(
    bundle: BehaviorRegistryBundle, execution_context: MftExecutionContext
) -> tuple[MftCase, ...]:
    cases: list[MftCase] = []
    for row in bundle.behavior_tests.rows:
        if row.test_class != "MFT":
            continue
        task_families = (
            execution_context.task_families
            if row.task_families == ("all",)
            else tuple(task for task in execution_context.task_families if task in row.task_families)
        )
        conditions = tuple(
            condition
            for condition in execution_context.baseline_conditions
            if condition.mode in row.applicable_baseline_mode_conditions
        )
        if not task_families or not conditions:
            raise MftError("MFT_CASE_COVERAGE_INCOMPLETE")
        cases.extend(
            MftCase(
                case_id=f"{row.test_id}:{task_family}:{condition.condition_id}",
                test_id=row.test_id,
                task_family=task_family,
                baseline_condition_id=condition.condition_id,
                baseline_mode=condition.mode,
            )
            for task_family in task_families
            for condition in conditions
        )
    if {case.test_id for case in cases} != {"MFT-01", "MFT-02", "MFT-03", "MFT-04"}:
        raise MftError("MFT_CASE_COVERAGE_INCOMPLETE")
    return tuple(cases)


def _classify_case(case: MftCase, execution: MftCaseExecution) -> MftCaseResult:
    status: Literal["pass", "model_behavior", "contract_failure"]
    failure_code = execution.failure_code
    if execution.status == "pass":
        status = "pass"
    elif execution.status == "model_failure" and case.test_id != "MFT-04":
        status = "model_behavior"
    else:
        status = "contract_failure"
        if case.test_id == "MFT-04" and failure_code is None:
            failure_code = "MFT04_CONTRACT_MISMATCH"
    return MftCaseResult(
        case=case,
        status=status,
        raw_log_ids=execution.raw_log_ids,
        pilot_allowance_units=execution.pilot_allowance_units,
        failure_code=failure_code,
    )


def _append_case(writer: MftEventLedger | object, result: MftCaseResult) -> None:
    append_case = getattr(writer, "append_mft_case", None)
    if callable(append_case):
        append_case(result)
        return
    append_audit_label = getattr(writer, "append_audit_label", None)
    if callable(append_audit_label):
        append_audit_label(
            {
                "baseline_condition_id": result.case.baseline_condition_id,
                "case_id": result.case_id,
                "failure_code": result.failure_code,
                "pilot_allowance_units": result.pilot_allowance_units,
                "raw_log_ids": list(result.raw_log_ids),
                "record_type": "mft_case_result",
                "status": result.status,
                "task_family": result.case.task_family,
                "test_id": result.test_id,
            }
        )
        return
    raise MftError("MFT_WRITER_UNSUPPORTED")
