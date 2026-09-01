from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import memcontam.readiness.phase13_main_runner_store as store_module
from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_live_dispatch import (
    MainUnitDispatchOutput,
    persist_reconciliation_evidence,
    persist_unit_dispatch,
)
from memcontam.readiness.phase13_main_runner import (
    DispatchCompleted,
    DispatchTechnicalFailure,
    InFlightEvidence,
    MainRunBinding,
    MainRunError,
    MainRunLedger,
    enumerate_execution_units,
)


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"


def _units():
    package = MainExecutionFreeze.model_validate_json(P5.read_bytes())
    return enumerate_execution_units(package, ROOT)


def _binding() -> MainRunBinding:
    return MainRunBinding(
        "phase13-main-a-execution-freeze-v1",
        "1" * 64,
        "2" * 64,
        "phase13-main-a-authorized-execution-v1",
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )


def _ledger(tmp_path: Path) -> MainRunLedger:
    return MainRunLedger.create(tmp_path / "main-run.sqlite3", _binding(), _units())


def _prefix_output(unit, *, cost_usd: float = 0.01) -> MainUnitDispatchOutput:
    messages = [{"role": "user", "content": "frozen request"}]
    return MainUnitDispatchOutput(
        evidence={
            "evidence_kind": "CLEAN_PREFIX",
            "prefix_unit_id": unit.unit_id,
            "checkpoint": {
                "schema_version": "phase13_main_prefix_checkpoint_v1",
                "baseline": unit.memory_baseline,
                "checkpoint_id": "checkpoint-1",
                "checkpoint_identity_sha256": "b" * 64,
                "canonical_sha256": "c" * 64,
                "canonical_state_utf8": "{}",
            },
            "runtime_evidence": {
                "unit_id": unit.unit_id,
                "task": unit.task,
                "seed": unit.seed,
                "memory_baseline": unit.memory_baseline,
                "arm": unit.arm,
                "production_identity": {
                    "execution_template_id": "game24|fh_bounded|prefix",
                    "trajectory_seed": unit.seed,
                    "concrete_seed_id": str(unit.seed),
                    "ordered_sample_ids_sha256": unit.ordered_sample_ids_sha256,
                    "registration_packet_sha256": unit.registration_packet_sha256,
                    "scientific_result": False,
                    "checkpoint_registry_sha256": unit.checkpoint_registry_sha256,
                },
                "observability_registration_packet_sha256": unit.registration_packet_sha256,
                "request": {
                    "api": "OpenAI Responses API",
                    "model": "gpt-5.6-luna",
                    "service_tier": "default",
                    "reasoning_mode": "standard",
                    "reasoning_effort": "none",
                    "reasoning_context": "current_turn",
                    "previous_response_id": None,
                    "store": False,
                    "timeout_seconds": 180,
                    "retries_after_initial_attempt": 0,
                    "semantic_invalid_generic_retry": False,
                },
            },
        },
        provider_calls=(
            MethodCall(
                call_id="prefix-call",
                stage="full_history_generate",
                messages=messages,
                raw_response="offline",
                model="gpt-5.6-luna",
                temperature=0.0,
                top_p=1.0,
                token_usage={"prompt_tokens": 3, "completion_tokens": 2},
                transport_attempts=1,
                provider_status="completed",
                provider_response_status="completed",
                provider_response_id="response-prefix-call",
                provider_usage={"input_tokens": 3, "output_tokens": 2},
                provider_service_tier="default",
                provider_returned_model="gpt-5.6-luna",
                provider_request_contract={
                    "model": "gpt-5.6-luna",
                    "input_sha256": hashlib.sha256(
                        json.dumps(
                            messages,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest(),
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "reasoning": {
                        "mode": "standard",
                        "effort": "none",
                        "context": "current_turn",
                    },
                    "previous_response_id": None,
                    "service_tier": "default",
                    "store": False,
                    "tools": [],
                    "max_output_tokens": 512,
                },
                provider_authority_contract={
                    "maximum_input_tokens": 9330,
                    "maximum_output_tokens": 512,
                    "execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V2",
                    "execution_envelope_sha256": (
                        "41cd7e7310a961d0856e2020b05a3ae455811fb0660455b4c7dfbcb0a9aafd93"
                    ),
                    "failure_contract_id": "CORE_TRANSPORT_ATTEMPT_CONTRACT_V2",
                    "failure_contract_sha256": (
                        "1ee66fcb795f97d483c2ef976133ee61dbd5108c9dae851c2c2786ff496d788f"
                    ),
                    "terminal_failure_contract_id": (
                        "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"
                    ),
                    "terminal_failure_contract_sha256": (
                        "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
                    ),
                    "rate_card_sha256": (
                        "50975b67dce4c59ba9267c3234a873076137ded5078aa3e8b5c9a2fad4ff3e06"
                    ),
                },
                provider_cost_usd=cost_usd,
                authoritative_provider_cost_usd=cost_usd,
                derived_cost_usd=cost_usd,
                provider_cost_source="AUTHORITATIVE_PROVIDER",
            ),
        ),
        realized_cost_krw=int(cost_usd * 1600),
    )


def _tamper(path: Path, action: str) -> None:
    if action == "delete":
        path.unlink()
    else:
        path.write_bytes(b"{}")


def test_dispatch_outcomes_reject_invalid_digest_and_negative_cost() -> None:
    with pytest.raises(MainRunError, match="MAIN_RUN_EVIDENCE_INVALID"):
        DispatchCompleted("not-a-sha256", 0)
    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        DispatchCompleted("6" * 64, -1)
    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        DispatchTechnicalFailure("PROVIDER_QUOTA", "7" * 64, -1)


def test_negative_projected_cost_fails_before_intent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None

    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        ledger.claim_dispatch(unit.unit_id, -1, 0)

    assert ledger.status().in_flight_count == 0


def test_tranche_cannot_exceed_frozen_core_gate(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None

    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        ledger.claim_dispatch(unit.unit_id, unit.projected_cost_krw, 450001)

    assert ledger.status().in_flight_count == 0


def test_second_runner_cannot_claim_while_unit_is_inflight(tmp_path: Path) -> None:
    first = _ledger(tmp_path)
    unit = first.next_pending()
    assert unit is not None
    first.persist_dispatch_intent(unit.unit_id)
    second = MainRunLedger.open(first.path, _binding(), _units())
    next_unit = second.next_pending()
    assert next_unit is not None

    with pytest.raises(MainRunError, match="MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"):
        second.persist_dispatch_intent(next_unit.unit_id)


def test_reconciliation_evidence_must_bind_current_intent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    context = ledger.in_flight_context(unit.unit_id)
    next_unit = _units()[1]
    wrong_context = replace(context, unit_id=next_unit.unit_id)
    evidence = InFlightEvidence.no_provider_request(wrong_context, "9" * 64)

    with pytest.raises(MainRunError, match="MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID"):
        ledger.reconcile(unit.unit_id, evidence)


def test_completed_reconciliation_rejects_caller_digest_without_durable_unit_evidence(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    evidence = InFlightEvidence.completed(
        ledger.in_flight_context(unit.unit_id),
        "a" * 64,
        16,
    )

    with pytest.raises(MainRunError, match="MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID"):
        ledger.reconcile(unit.unit_id, evidence)

    assert ledger.status().in_flight_count == 1


def test_completed_reconciliation_accepts_exact_durable_unit_evidence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    assert unit.kind == "CLEAN_PREFIX"
    assert unit.memory_baseline == "fh_bounded"
    ledger.persist_dispatch_intent(unit.unit_id)
    completed = persist_unit_dispatch(tmp_path, unit, _prefix_output(unit))

    ledger.reconcile(
        unit.unit_id,
        InFlightEvidence.completed(
            ledger.in_flight_context(unit.unit_id),
            completed.evidence_sha256,
            completed.realized_cost_krw,
        ),
    )

    assert ledger.status().completed_count == 1


@pytest.mark.parametrize("action", ["delete", "replace"])
def test_completed_evidence_remains_part_of_ledger_integrity(
    tmp_path: Path,
    action: str,
) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    completed = persist_unit_dispatch(tmp_path, unit, _prefix_output(unit))
    ledger.persist_completed(unit.unit_id, completed)
    _tamper(tmp_path / "units" / f"{unit.sequence:06d}-{unit.unit_id}.json", action)

    with pytest.raises(MainRunError, match="MAIN_RUN_COMPLETION_EVIDENCE_INVALID"):
        ledger.status()


@pytest.mark.parametrize("action", ["delete", "replace"])
def test_reconciliation_evidence_remains_part_of_ledger_integrity(
    tmp_path: Path,
    action: str,
) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    context = ledger.in_flight_context(unit.unit_id)
    evidence = persist_reconciliation_evidence(tmp_path, "NO_PROVIDER_REQUEST", context)
    ledger.reconcile(unit.unit_id, evidence)
    _tamper(tmp_path / "reconciliation" / f"{context.intent_event_hash}.json", action)

    with pytest.raises(MainRunError, match="MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID"):
        ledger.status()


def test_realized_overrun_is_preserved_and_pauses_before_next_dispatch(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    assert ledger.claim_dispatch(unit.unit_id, unit.projected_cost_krw, 20)
    completed = persist_unit_dispatch(tmp_path, unit, _prefix_output(unit, cost_usd=0.015625))
    ledger.persist_completed(unit.unit_id, completed)

    assert ledger.status().realized_cost_krw == 25
    next_unit = ledger.next_pending()
    assert next_unit is not None
    assert not ledger.claim_dispatch(next_unit.unit_id, next_unit.projected_cost_krw, 20)
    assert ledger.status().session_state == "PAUSED_BEFORE_DISPATCH"


def test_event_genesis_is_bound_and_nonempty(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)

    with sqlite3.connect(ledger.path) as connection:
        previous_hash = connection.execute(
            "SELECT previous_hash FROM events WHERE event_sequence = 0"
        ).fetchone()[0]

    assert len(previous_hash) == 64
    assert previous_hash != "0" * 64


def test_crash_during_creation_never_publishes_partial_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "main-run.sqlite3"

    def crash(*_args) -> None:
        raise MainRunError("TEST_CREATION_CRASH")

    monkeypatch.setattr(store_module, "_initialize_ledger", crash)

    with pytest.raises(MainRunError, match="TEST_CREATION_CRASH"):
        MainRunLedger.create(path, _binding(), _units())

    assert not path.exists()
