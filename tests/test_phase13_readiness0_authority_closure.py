from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.readiness.phase13_readiness0_evidence_models import (
    ProviderCallEvidence,
    RuntimeJoinEvidence,
)


ROOT = Path(__file__).resolve().parents[1]
MR_P4 = ROOT / "data/phase13/main/mr_p4"


def test_live_request_binds_non_circular_implementation_manifest() -> None:
    request = json.loads((MR_P4 / "readiness0_live_request_v1.json").read_text())
    implementation = json.loads(
        (MR_P4 / "readiness0_live_implementation_manifest_v1.json").read_text()
    )

    assert request["implementation_manifest"]["path"].endswith(
        "readiness0_live_implementation_manifest_v1.json"
    )
    assert "readiness0_live_implementation_manifest_v1.json" not in {
        Path(row["path"]).name for row in implementation["artifacts"].values()
    }
    assert implementation["status"] == "PASS"


def test_f1c_report_has_one_runtime_and_exact_active_arm_rows() -> None:
    registry = json.loads((MR_P4 / "readiness0_f1c_registry_v1.json").read_text())
    report = json.loads((MR_P4 / "readiness0_f1c_report_v1.json").read_text())

    assert registry["report"]["path"].endswith("readiness0_f1c_report_v1.json")
    assert report["status"] == "PASS"
    assert report["runtime"]["network_attempts"] == 0
    assert len(report["rows"]) == 52
    assert len({row["row_id"] for row in report["rows"]}) == 52
    assert sum(row["baseline"] == "rag_frozen" for row in report["rows"]) == 12
    assert sum(row["baseline"] == "bot_style" for row in report["rows"]) == 20
    assert sum(row["baseline"] == "dc_rs" for row in report["rows"]) == 20


def test_window_proof_resolves_all_frozen_contexts_and_roles() -> None:
    proof = json.loads((MR_P4 / "readiness0_window_proof_v1.json").read_text())

    assert proof["H_run"] == 50
    assert len(proof["windows"]) == 50
    assert [row["window_id"] for row in proof["windows"]] == [
        f"core_prefix_{index:02d}" for index in range(1, 51)
    ]
    assert proof["windows"][49]["role"] == "confirmatory_primary"
    assert {
        row["end"] for row in proof["windows"] if row["role"] == "prespecified_sensitivity"
    } == {5, 10, 20}
    assert proof["provider_dispatch_suffix_positions"] == [1]
    assert proof["resolved_context_count"] == 2500


def test_evidence_models_require_registered_provider_and_runtime_joins() -> None:
    assert {
        "requested_model",
        "returned_model",
        "response_status",
        "reasoning_mode",
        "reasoning_effort",
        "reasoning_context",
        "previous_response_id",
        "store",
        "tools",
        "maximum_input_tokens",
        "maximum_output_tokens",
        "execution_envelope_id",
        "execution_envelope_sha256",
        "failure_contract_id",
        "failure_contract_sha256",
        "cost_source",
        "rate_card_sha256",
    } <= ProviderCallEvidence.model_fields.keys()
    assert {
        "retrieval_query_sha256",
        "retrieval_candidates_sha256",
        "retrieval_source_span_sha256",
        "capacity_artifact_sha256",
        "task_order_sha256",
        "analysis_window_id",
        "analysis_window_registry_sha256",
    } <= RuntimeJoinEvidence.model_fields.keys()


@pytest.mark.parametrize(
    ("model", "field"),
    [
        *(
            (ProviderCallEvidence, field)
            for field in (
                "requested_model", "returned_model", "response_status",
                "reasoning_mode", "reasoning_effort", "reasoning_context",
                "previous_response_id", "store", "tools", "maximum_input_tokens",
                "maximum_output_tokens", "execution_envelope_id",
                "execution_envelope_sha256", "failure_contract_id",
                "failure_contract_sha256", "terminal_failure_contract_id",
                "terminal_failure_contract_sha256", "raw_usage", "normalized_usage",
                "authoritative_provider_cost_usd", "derived_cost_usd", "cost_source",
                "rate_card_sha256",
            )
        ),
        *(
            (RuntimeJoinEvidence, field)
            for field in (
                "retrieval_query_sha256", "retrieval_candidates_sha256",
                "retrieval_source_span_sha256", "capacity_artifact_sha256",
                "task_order_sha256", "analysis_window_id",
                "analysis_window_registry_sha256",
            )
        ),
    ],
)
def test_authority_evidence_fields_are_mandatory(model: type, field: str) -> None:
    assert model.model_fields[field].is_required()


def test_current_status_reports_missing_credential_and_stale_authorization() -> None:
    status = json.loads((MR_P4 / "readiness0_current_status_v1.json").read_text())

    assert status["f1c_status"] == "PASS"
    assert status["external_blockers"] == [
        "OPENAI_API_KEY_MISSING",
        "READINESS0_REAUTHORIZATION_REQUIRED",
    ]
    assert status["mr_p4_status"] == "OPEN"
    assert status["mr_p5_status"] == "NOT_STARTED"
    assert status["mr_p6_status"] == "NOT_AUTHORIZED"
    assert status["main_a_status"] == "NOT_STARTED"
