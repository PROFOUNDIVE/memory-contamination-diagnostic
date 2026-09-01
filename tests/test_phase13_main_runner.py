from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue

from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_cost_policy import load_cost_policy_bundle
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
    run_pending,
)


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"
_COST_POLICY = load_cost_policy_bundle(ROOT)
_RATE_CARD_SHA256 = hashlib.sha256(
    json.dumps(
        _COST_POLICY.proof.rate_card.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _package() -> MainExecutionFreeze:
    return MainExecutionFreeze.model_validate_json(P5.read_bytes())


def _binding() -> MainRunBinding:
    return MainRunBinding(
        package_id="phase13-main-a-execution-freeze-v1",
        package_sha256="1" * 64,
        package_hash="2" * 64,
        authorization_id="phase13-main-a-authorized-execution-v1",
        authorization_sha256="3" * 64,
        authorization_hash="4" * 64,
        runner_sha256="5" * 64,
    )


def _ledger(tmp_path: Path) -> MainRunLedger:
    return MainRunLedger.create(
        tmp_path / "main-run.sqlite3",
        _binding(),
        enumerate_execution_units(_package(), ROOT),
    )


def _completed_call(call_id: str, stage: str) -> MethodCall:
    messages = [{"role": "user", "content": "frozen request"}]
    authority = next(
        row for row in _COST_POLICY.registry.stages if row.semantic_stage_id == stage
    )
    return MethodCall(
        call_id=call_id,
        stage=stage,
        messages=messages,
        raw_response="offline",
        model="gpt-5.6-luna",
        temperature=0.0,
        top_p=1.0,
        token_usage={"prompt_tokens": 3, "completion_tokens": 2},
        transport_attempts=1,
        provider_status="completed",
        provider_response_status="completed",
        provider_response_id=f"response-{call_id}",
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
            "reasoning": {"mode": "standard", "effort": "none", "context": "current_turn"},
            "previous_response_id": None,
            "service_tier": "default",
            "store": False,
            "tools": [],
            "max_output_tokens": authority.maximum_output_tokens,
        },
        provider_authority_contract={
            "maximum_input_tokens": authority.maximum_input_tokens,
            "maximum_output_tokens": authority.maximum_output_tokens,
            "execution_envelope_id": _COST_POLICY.registry.registry_id,
            "execution_envelope_sha256": _COST_POLICY.registry.registry_hash,
            "failure_contract_id": _COST_POLICY.retry.contract_id,
            "failure_contract_sha256": _COST_POLICY.retry.contract_hash,
            "terminal_failure_contract_id": _COST_POLICY.retry.terminal_failure_contract_id,
            "terminal_failure_contract_sha256": (
                _COST_POLICY.retry.terminal_failure_contract_sha256
            ),
            "rate_card_sha256": _RATE_CARD_SHA256,
        },
        provider_cost_usd=0.001,
        authoritative_provider_cost_usd=0.001,
        derived_cost_usd=0.001,
        provider_cost_source="AUTHORITATIVE_PROVIDER",
    )


def _success(tmp_path: Path, unit) -> DispatchCompleted:
    stages = {
        "fh_bounded": ("full_history_generate",),
        "rag_frozen": ("rag_generate",),
        "bot_style": (
            "bot_problem_distill",
            "bot_instantiate_solve",
            "bot_thought_distill",
        ),
        "reflexion_style": ("reflexion_generate", "reflexion_reflect"),
        "dc_rs": ("dc_rs_generate", "dc_rs_synthesize"),
    }
    if unit.kind == "NO_MEMORY_SINGLETON":
        expected_stages = ("no_memory_generate",) * 50
        evidence: dict[str, JsonValue] = {
            "evidence_kind": "NO_MEMORY_SINGLETON",
            "internal_baseline": "nomem",
            "internal_arm": "clean",
            "scientific_arm": "NOT_APPLICABLE",
        }
    elif unit.kind == "CLEAN_PREFIX":
        assert unit.memory_baseline is not None
        expected_stages = stages[unit.memory_baseline]
        evidence = {
            "evidence_kind": "CLEAN_PREFIX",
            "prefix_unit_id": unit.unit_id,
            "checkpoint": {
                "schema_version": "phase13_main_prefix_checkpoint_v1",
                "baseline": unit.memory_baseline,
                "checkpoint_id": f"checkpoint-{unit.unit_id}",
                "checkpoint_identity_sha256": "6" * 64,
                "canonical_sha256": "7" * 64,
                "canonical_state_utf8": "{}",
            },
        }
    else:
        assert unit.memory_baseline is not None
        repetitions = 100 if unit.memory_baseline == "reflexion_style" else 50
        expected_stages = tuple(
            stage for stage in stages[unit.memory_baseline] for _ in range(repetitions)
        )
        evidence = {
            "evidence_kind": "MEMORY_BEARING",
            "prefix_unit_id": unit.prefix_unit_id,
            "consumed_checkpoint_id": "checkpoint",
            "consumed_checkpoint_identity_sha256": "6" * 64,
            "consumed_checkpoint_canonical_sha256": "7" * 64,
        }
    evidence["runtime_evidence"] = {
        "unit_id": unit.unit_id,
        "task": unit.task,
        "seed": unit.seed,
        "memory_baseline": unit.memory_baseline,
        "arm": unit.arm,
        "production_identity": {
            "execution_template_id": unit.execution_template_id,
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
    }
    calls = tuple(
        _completed_call(f"{unit.unit_id}-{index}", stage)
        for index, stage in enumerate(expected_stages)
    )
    return persist_unit_dispatch(
        tmp_path,
        unit,
        MainUnitDispatchOutput(
            evidence=evidence,
            provider_calls=calls,
            realized_cost_krw=2 * len(calls),
        ),
    )


def test_frozen_domain_enumerates_1200_injective_objects_once() -> None:
    units = enumerate_execution_units(_package())

    assert len(units) == 1200
    assert len({unit.unit_id for unit in units}) == 1200
    assert tuple(unit.sequence for unit in units) == tuple(range(1200))


def test_nomem_is_one_disjoint_singleton_per_task_and_seed() -> None:
    units = enumerate_execution_units(_package())
    nomem = tuple(unit for unit in units if unit.kind == "NO_MEMORY_SINGLETON")

    assert len(nomem) == 50
    assert {(unit.seed, unit.task) for unit in nomem} == {
        (seed, task) for seed in range(10) for task in _package().active_cells.tasks
    }
    assert {unit.arm for unit in nomem} == {"NOT_APPLICABLE"}
    assert {unit.memory_baseline for unit in nomem} == {None}


@pytest.mark.parametrize("terminal", ["COMPLETED", "TERMINAL_TECHNICAL_MISSING"])
def test_terminal_unit_is_never_redispatched(tmp_path: Path, terminal: str) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    if terminal == "COMPLETED":
        ledger.persist_completed(unit.unit_id, _success(tmp_path, unit))
    else:
        reconciliation = persist_reconciliation_evidence(
            tmp_path,
            "TERMINAL_FAILURE",
            ledger.in_flight_context(unit.unit_id),
            failure_code="PROVIDER_QUOTA",
        )
        ledger.persist_terminal_missing(
            unit.unit_id,
            DispatchTechnicalFailure("PROVIDER_QUOTA", reconciliation.evidence_sha256),
        )
    calls: list[str] = []

    run_pending(
        ledger,
        lambda row: calls.append(row.unit_id) or _success(tmp_path, row),
        tranche_ceiling_krw=444126,
        max_units=1,
    )

    next_index = 1 if terminal == "COMPLETED" else 5
    assert calls == [enumerate_execution_units(_package())[next_index].unit_id]


def test_clean_tranche_pause_precedes_dispatch_intent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    calls: list[str] = []
    pending = ledger.next_pending()
    assert pending is not None

    report = run_pending(
        ledger,
        lambda unit: calls.append(unit.unit_id) or _success(tmp_path, unit),
        tranche_ceiling_krw=pending.projected_cost_krw - 1,
    )

    assert report.session_state == "PAUSED_BEFORE_DISPATCH"
    assert report.attempted_count == 0
    assert calls == []
    assert ledger.status().in_flight_count == 0


def test_quota_failure_terminalizes_current_unit_and_stops(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    calls: list[str] = []

    def dispatch(unit) -> DispatchCompleted:
        calls.append(unit.unit_id)
        reconciliation = persist_reconciliation_evidence(
            tmp_path,
            "TERMINAL_FAILURE",
            ledger.in_flight_context(unit.unit_id),
            failure_code="PROVIDER_QUOTA",
        )
        raise DispatchTechnicalFailure("PROVIDER_QUOTA", reconciliation.evidence_sha256)

    report = run_pending(
        ledger,
        dispatch,
        tranche_ceiling_krw=444126,
    )

    assert report.session_state == "READY"
    assert report.terminal_technical_missing_count == 5
    assert len(calls) == 1
    assert ledger.status().pending_count == 1195


def test_repeated_resume_does_not_duplicate_calls(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    calls: list[str] = []

    def dispatch(unit) -> DispatchCompleted:
        calls.append(unit.unit_id)
        return _success(tmp_path, unit)

    for _ in range(3):
        run_pending(
            ledger,
            dispatch,
            tranche_ceiling_krw=444126,
            max_units=1,
        )

    assert len(calls) == len(set(calls)) == 3
    assert ledger.status().completed_count == 3


def test_ledger_tampering_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE execution_units SET state = 'COMPLETED' WHERE unit_id = ?",
            (unit.unit_id,),
        )

    with pytest.raises(MainRunError, match="MAIN_RUN_LEDGER_INTEGRITY_INVALID"):
        MainRunLedger.open(ledger.path, _binding(), enumerate_execution_units(_package()))


def test_invalid_frozen_projection_leaves_unit_safely_pending(tmp_path: Path) -> None:
    units = enumerate_execution_units(_package())
    invalid = (replace(units[0], projected_cost_krw=-1), *units[1:])
    ledger = MainRunLedger.create(tmp_path / "main-run.sqlite3", _binding(), invalid)

    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        run_pending(
            ledger,
            lambda unit: _success(tmp_path, unit),
            tranche_ceiling_krw=444126,
        )

    assert ledger.status().pending_count == 1200
    assert ledger.status().in_flight_count == 0


def test_inflight_with_proven_no_request_can_return_to_pending(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)

    context = ledger.in_flight_context(unit.unit_id)
    ledger.reconcile(
        unit.unit_id,
        persist_reconciliation_evidence(tmp_path, "NO_PROVIDER_REQUEST", context),
    )

    assert ledger.next_pending() == unit
    assert ledger.status().in_flight_count == 0


def test_inflight_with_terminal_evidence_is_never_retried(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)

    context = ledger.in_flight_context(unit.unit_id)
    ledger.reconcile(
        unit.unit_id,
        persist_reconciliation_evidence(
            tmp_path,
            "TERMINAL_FAILURE",
            context,
            failure_code="PROVIDER_QUOTA",
        ),
    )

    assert ledger.status().terminal_technical_missing_count == 5
    assert ledger.next_pending() != unit


def test_accepted_but_unpersisted_request_blocks_blind_resume(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    context = ledger.in_flight_context(unit.unit_id)

    with pytest.raises(MainRunError, match="MAIN_RUN_IN_FLIGHT_AMBIGUOUS"):
        ledger.reconcile(unit.unit_id, InFlightEvidence.ambiguous(context, "a" * 64))

    assert ledger.status().in_flight_count == 1


def test_crash_after_terminal_persist_skips_unit_on_resume(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    ledger.persist_completed(unit.unit_id, _success(tmp_path, unit))

    reopened = MainRunLedger.open(
        ledger.path,
        _binding(),
        enumerate_execution_units(_package(), ROOT),
    )

    assert reopened.next_pending() != unit
    assert reopened.status().completed_count == 1
