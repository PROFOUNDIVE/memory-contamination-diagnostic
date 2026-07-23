from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memcontam.behavior.mft import (
    MftBaselineCondition,
    MftCaseExecution,
    MftEventLedger,
    MftExecutionContext,
    freeze_mft_manifest,
    run_mft_suite,
)
from memcontam.behavior.registry import load_behavior_registry_bundle


ROOT = Path(__file__).parents[1]
REGISTRIES = ROOT / "data" / "phase12" / "registries"
FIXTURES = ROOT / "tests" / "fixtures" / "phase12"
TASKS = ("game24", "math_equation_balancer", "word_sorting")
CONDITIONS = (
    MftBaselineCondition("nomem", "nomem"),
    MftBaselineCondition("fh_bounded", "memory"),
    MftBaselineCondition("rag_frozen", "memory"),
    MftBaselineCondition("bot_style", "memory"),
    MftBaselineCondition("reflexion_style", "memory"),
)


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _context(execute_case):
    return MftExecutionContext(
        task_families=TASKS,
        baseline_conditions=CONDITIONS,
        execute_case=execute_case,
    )


def _passed(case):
    return MftCaseExecution(
        status="pass",
        raw_log_ids=(f"raw:{case.case_id}",),
        pilot_allowance_units=1,
    )


def test_executes_all_four_frozen_mft_families() -> None:
    behavior_fixture = _fixture("FX-BEHAVIOR-001.json")
    bundle = load_behavior_registry_bundle(REGISTRIES)
    writer = MftEventLedger()

    result = run_mft_suite(bundle, _context(_passed), writer)
    manifest = freeze_mft_manifest(
        result,
        case_status_ledger_hash="case-ledger-hash",
        pilot_allowance_ledger_hash="pilot-allowance-hash",
        frozen_at="2026-07-23T00:00:00Z",
    )

    assert behavior_fixture["expected"]["mft04_required"] is True
    assert {case.test_id for case in result.cases} == {"MFT-01", "MFT-02", "MFT-03", "MFT-04"}
    assert len(result.cases) == 39
    assert result.all_registered_cases_attempted is True
    assert all(case.status == "pass" and case.raw_log_ids for case in result.cases)
    assert writer.cases == list(result.cases)
    assert manifest.all_registered_cases_attempted is True
    assert (manifest.mft04_status, manifest.route_gate_status) == ("pass", "pass")


def test_distinguishes_model_failure_from_mft04_contract_failure() -> None:
    behavior_fixture = _fixture("FX-BEHAVIOR-001.json")
    outcome_fixture = _fixture("FX-OUTCOME-001.json")
    route_fixture = _fixture("FX-ROUTE-001.json")
    bundle = load_behavior_registry_bundle(REGISTRIES)

    def execute_case(case):
        if case.test_id == "MFT-01":
            return MftCaseExecution(
                status="model_failure",
                raw_log_ids=(f"raw:{case.case_id}",),
                pilot_allowance_units=1,
            )
        if case.test_id == "MFT-04":
            return MftCaseExecution(
                status="contract_failure",
                raw_log_ids=(f"raw:{case.case_id}",),
                pilot_allowance_units=0,
                failure_code="MFT04_CONTRACT_MISMATCH",
            )
        return _passed(case)

    result = run_mft_suite(bundle, _context(execute_case), MftEventLedger())
    manifest = freeze_mft_manifest(
        result,
        case_status_ledger_hash="case-ledger-fail-hash",
        pilot_allowance_ledger_hash="mft-pilot-allowance-hash",
        frozen_at="2026-07-23T00:00:00Z",
    )

    assert behavior_fixture["expected"]["valid_model_failure_blocks_route"] is False
    assert any(case.status == "model_behavior" for case in result.cases if case.test_id == "MFT-01")
    assert all(
        case.status == "contract_failure" for case in result.cases if case.test_id == "MFT-04"
    )
    assert outcome_fixture["cases"][2]["expected"][1] == "model_behavior"
    assert route_fixture["completed_failed_mft_manifest"]["mft04_status"] == "fail"
    assert (manifest.mft04_status, manifest.route_gate_status) == ("fail", "blocked")
