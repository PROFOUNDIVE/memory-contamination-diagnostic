from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

import memcontam.readiness.phase13_main_runner as runner_module
from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_cost_policy import load_cost_policy_bundle
from memcontam.readiness.phase13_main_live_dispatch import (
    MainUnitDispatchOutput,
    persist_unit_dispatch,
)
from memcontam.readiness.phase13_main_runner import (
    DispatchCompleted,
    MainRunError,
    MainRunRequest,
    open_main_run,
    prepare_main_run,
    resume_main,
    run_main,
)


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"
P6 = ROOT / "data/phase13/main/mr_p6/authorized_execution_v1.json"
_COST_POLICY = load_cost_policy_bundle(ROOT)
_RATE_CARD_SHA256 = hashlib.sha256(
    json.dumps(
        _COST_POLICY.proof.rate_card.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _authorization_sha256() -> str:
    return hashlib.sha256(P6.read_bytes()).hexdigest()


def _request(tmp_path: Path, expected_sha256: str | None = None) -> MainRunRequest:
    return MainRunRequest(
        repository_root=ROOT,
        package_path=P5,
        authorization_path=P6,
        expected_authorization_sha256=expected_sha256 or _authorization_sha256(),
        run_root=tmp_path,
        run_id="offline-qa",
    )


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-m", "memcontam.readiness.phase13_main_runner_cli", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _common(tmp_path: Path) -> tuple[str, ...]:
    return (
        "--repository-root",
        str(ROOT),
        "--package",
        str(P5),
        "--authorization",
        str(P6),
        "--expected-authorization-sha256",
        _authorization_sha256(),
        "--run-root",
        str(tmp_path),
        "--run-id",
        "offline-qa",
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


def test_authorized_run_creation_binds_exact_frozen_inputs(tmp_path: Path) -> None:
    ledger = prepare_main_run(_request(tmp_path))

    status = ledger.status()
    assert status.session_state == "NOT_STARTED"
    assert status.total_count == 1200
    assert status.pending_count == 1200


def test_authorized_run_reopen_revalidates_package_and_authorization(tmp_path: Path) -> None:
    prepare_main_run(_request(tmp_path))

    reopened = open_main_run(_request(tmp_path))

    assert reopened.status().total_count == 1200


def test_authorized_run_and_resume_dispatch_distinct_pending_units(tmp_path: Path) -> None:
    calls: list[str] = []

    def dispatch(unit) -> DispatchCompleted:
        calls.append(unit.unit_id)
        if unit.kind == "CLEAN_PREFIX":
            stages = ("full_history_generate",)
            evidence: dict[str, JsonValue] = {
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
            stages = ("full_history_generate",) * 50
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
        provider_calls = tuple(
            _completed_call(f"{unit.unit_id}-{index}", stage)
            for index, stage in enumerate(stages)
        )
        return persist_unit_dispatch(
            tmp_path / "offline-qa",
            unit,
            MainUnitDispatchOutput(
                evidence=evidence,
                provider_calls=provider_calls,
                realized_cost_krw=2 * len(provider_calls),
            ),
        )

    run_main(
        _request(tmp_path),
        dispatch,
        tranche_ceiling_krw=500,
        max_units=1,
    )
    resume_main(
        _request(tmp_path),
        dispatch,
        tranche_ceiling_krw=500,
        max_units=1,
    )

    assert len(calls) == len(set(calls)) == 2


def test_authorized_run_rejects_authorization_hash_tampering(tmp_path: Path) -> None:
    with pytest.raises(MainRunError, match="MAIN_AUTHORIZATION_FILE_HASH_MISMATCH"):
        prepare_main_run(_request(tmp_path, "0" * 64))


def test_authorized_run_rejects_package_changed_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = json.loads(P5.read_text())
    changed["dispatch"]["task_order"] = list(reversed(changed["dispatch"]["task_order"]))
    changed_raw = json.dumps(changed).encode()
    monkeypatch.setattr(runner_module, "read_regular_nofollow", lambda _path: changed_raw)

    with pytest.raises(MainRunError, match="MAIN_RUN_PACKAGE_BYTES_CHANGED"):
        prepare_main_run(_request(tmp_path))


def test_frozen_runner_binds_direct_authorization_trust_base() -> None:
    package = json.loads(P5.read_text())
    roles = {binding["role"] for binding in package["artifacts"]}

    assert {
        "main_execution",
        "main_execution_models",
        "main_execution_bindings",
    } <= roles


def test_main_run_id_must_be_one_component(tmp_path: Path) -> None:
    request = _request(tmp_path)
    changed = MainRunRequest(
        request.repository_root,
        request.package_path,
        request.authorization_path,
        request.expected_authorization_sha256,
        request.run_root,
        "../escape",
    )

    with pytest.raises(MainRunError, match="MAIN_RUN_ID_INVALID"):
        prepare_main_run(changed)


def test_phase13_help_exposes_main_execution_control_surface() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "run" in result.stdout
    assert "status" in result.stdout
    assert "resume" in result.stdout


def test_main_cli_run_status_resume_are_offline_and_stable(tmp_path: Path) -> None:
    started = _run("run", *_common(tmp_path))
    status = _run("status", *_common(tmp_path))
    resumed = _run("resume", *_common(tmp_path))

    assert started.returncode == status.returncode == resumed.returncode == 0
    reports = tuple(json.loads(result.stdout) for result in (started, status, resumed))
    assert {report["session_state"] for report in reports} == {"NOT_STARTED"}
    assert {report["total_count"] for report in reports} == {1200}
    assert {report["pending_count"] for report in reports} == {1200}


def test_main_cli_reports_bad_authorization_without_traceback(tmp_path: Path) -> None:
    arguments = list(_common(tmp_path))
    hash_index = arguments.index("--expected-authorization-sha256") + 1
    arguments[hash_index] = "0" * 64

    result = _run("run", *arguments)

    assert result.returncode != 0
    assert "MAIN_AUTHORIZATION_FILE_HASH_MISMATCH" in result.stderr
    assert "Traceback" not in result.stderr
