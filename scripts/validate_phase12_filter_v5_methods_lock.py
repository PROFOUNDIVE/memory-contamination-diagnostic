from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Final

import yaml


TASKS: Final = ["game24", "math_equation_balancer", "word_sorting"]
BASELINES: Final = ["full_history", "rag_frozen", "bot_style", "reflexion_style"]
SEARCH_IDS: Final = [f"SC-{suite}-{coverage}-{repeatability}-{retry}" for suite in ("S1K1", "S2K2") for coverage in ("C50", "C80") for repeatability in ("T50", "T80") for retry in ("R0", "R1")]
REPORTS: Final = {
    "authority-transition": "phase12_fv5_authority_transition_report_v1", "methods-lock": "phase12_fv5_methods_lock_report_v1", "freeze-a": "phase12_fv5_freeze_a_report_v1", "screening": "phase12_fv5_screening_report_v1", "freeze-b-search-config": "phase12_fv5_freeze_b_search_config_report_v1", "bct-execution": "phase12_fv5_bct_execution_report_v1", "archive-validation": "phase12_fv5_archive_validation_report_v1", "claim-scope": "phase12_fv5_claim_scope_report_v1", "pilot-b-readiness": "phase12_fv5_pilot_b_readiness_report_v1",
}


class MethodsError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise MethodsError(code)


def _load(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "METHODS_CONFIG_INVALID")
    return value


def validate(document: Path, config: Path, plan: Path) -> None:
    data = _load(config)
    document_text = document.read_text(encoding="utf-8")
    descriptor = plan.parents[1] / "approvals" / "phase12-post-filter-v5-calibration-readiness.plan.sha256"
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    _require(descriptor.read_bytes() == digest.encode("ascii") + b"\n", "METHODS_PLAN_BINDING_MISMATCH")
    _require(data.get("schema_version") == "phase12_fv5_bct_calibration_methods_v1" and data.get("approved_plan_sha256") == digest, "METHODS_PLAN_BINDING_MISMATCH")
    scope = data.get("scope")
    if not isinstance(scope, dict):
        raise MethodsError("METHODS_STRATA_MISMATCH")
    _require(scope.get("tasks") == TASKS and scope.get("baselines") == BASELINES and scope.get("excluded_baselines") == ["nomem"], "METHODS_STRATA_MISMATCH")
    _require(scope.get("strata") == [f"{task}:{baseline}" for task in TASKS for baseline in BASELINES], "METHODS_STRATA_MISMATCH")
    search = data.get("search_config")
    _require(isinstance(search, dict) and search.get("ids") == SEARCH_IDS, "METHODS_SEARCH_CONFIG_MISMATCH")
    laws = data.get("laws")
    _require(isinstance(laws, dict) and laws.get("generators") == ["game24_exact_fraction", "meb_standard_precedence", "word_sorting_first_difference"] and laws.get("certificates") == ["phase12_fv5_game24_certificate_v1", "phase12_fv5_meb_certificate_v1", "phase12_fv5_word_sorting_certificate_v1"] and laws.get("duplicate_exclusions") == ["pilot_a", "candidate_examples", "future_main", "reserved_extension", "exact", "canonical", "near_duplicate"], "METHODS_GENERATOR_LAW_MISMATCH")
    budget = data.get("budget")
    if not isinstance(budget, dict):
        raise MethodsError("METHODS_BUDGET_MISMATCH")
    screening, bct, shared = budget.get("screening"), budget.get("bct"), budget.get("shared")
    if not isinstance(screening, dict) or not isinstance(bct, dict) or not isinstance(shared, dict):
        raise MethodsError("METHODS_BUDGET_MISMATCH")
    _require(screening.get("maximum_calls") == 90 and bct.get("maximum_calls") == 480 and shared.get("maximum_calls") == 570 and shared.get("hard_ceiling_usd") == 10 and shared.get("wall_seconds") == 10800, "METHODS_BUDGET_MISMATCH")
    _require(screening.get("native_call_multiplier") == 5 and bct.get("native_call_multiplier") == 5, "METHODS_CALL_RESERVATION_MISMATCH")
    graph = data.get("native_stage_graph")
    _require(isinstance(graph, dict) and graph.get("bot_style") == ["bot_problem_distill:issued", "bot_instantiate_solve:issued_if_distill_parses_else_not_issued"] and set(graph) == set(BASELINES), "METHODS_STAGE_GRAPH_MISMATCH")
    _require(data.get("terminal_table") == ["AWAITING_SCREENING_AUTHORIZATION", "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE", "FILTER_V5_PILOT_B_NOT_ESTIMABLE", "AWAITING_BCT_AUTHORIZATION", "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_BCT_EVIDENCE", "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION"], "METHODS_TERMINAL_TABLE_MISMATCH")
    reports = data.get("report_matrix")
    _require(isinstance(reports, list) and {row.get("report_id"): row.get("schema_id") for row in reports if isinstance(row, dict)} == REPORTS and all(isinstance(row, dict) and row.get("bindings") for row in reports), "METHODS_REPORT_MATRIX_MISMATCH")
    _require("not_contradicted -> active" in document_text and "not_contradicted -> safe" not in document_text, "METHODS_DECISION_LAW_MISMATCH")
    _require("No pooling across challenge, Main, or code" in document_text, "METHODS_NO_POOLING_LAW_MISSING")
    _require("USD 10" in document_text and not re.search(r"USD\s+11\b", document_text), "METHODS_BUDGET_MISMATCH")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        validate(arguments.document, arguments.config, arguments.plan)
    except (MethodsError, OSError, yaml.YAMLError) as error:
        print(error.code if isinstance(error, MethodsError) else "METHODS_INPUT_INVALID")
        return 2
    print("METHODS_LOCK_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
