from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_main_live_contract import (
    MainLiveContractError,
    load_main_live_contract,
)
from memcontam.readiness.phase13_main_live_dispatch import (
    DurableMainDispatch,
    MainLiveDispatchError,
    MainUnitDispatchOutput,
    load_live_environment,
    persist_unit_dispatch,
    summarize_telemetry,
)
from memcontam.readiness.phase13_main_runner_models import ExecutionUnit


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v2.json"
P6 = ROOT / "data/phase13/main/mr_p6/authorized_execution_v2.json"


def _contract_payload() -> dict[str, JsonValue]:
    return json.loads(
        (ROOT / "data/phase13/main/main_live_contract_v1.json").read_text(encoding="utf-8")
    )


def _unit() -> ExecutionUnit:
    return ExecutionUnit(
        0,
        "c" * 64,
        "CLEAN_PREFIX",
        0,
        "game24",
        "fh_bounded",
        "NOT_APPLICABLE",
        None,
        1,
        "game24|fh_bounded|prefix",
        "f" * 64,
        "a" * 64,
        "b" * 64,
    )


def _request_contract(messages: list[dict[str, str]], stage: str) -> dict[str, JsonValue]:
    limits = {
        "full_history_generate": 512,
        "bot_problem_distill": 384,
        "bot_instantiate_solve": 512,
        "bot_thought_distill": 384,
        "reflexion_generate": 512,
        "reflexion_reflect": 384,
    }
    return {
        "model": "gpt-5.6-luna",
        "input_sha256": hashlib.sha256(
            json.dumps(
                messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest(),
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning": {"mode": "standard", "effort": "none", "context": "current_turn"},
        "previous_response_id": None,
        "service_tier": "default",
        "store": False,
        "tools": [],
        "max_output_tokens": limits[stage],
    }


def _authority_contract(stage: str) -> dict[str, JsonValue]:
    limits = {
        "full_history_generate": (9330, 512),
        "bot_problem_distill": (1177, 384),
        "bot_instantiate_solve": (1949, 512),
        "bot_thought_distill": (2545, 384),
        "reflexion_generate": (2282, 512),
        "reflexion_reflect": (3349, 384),
    }
    maximum_input_tokens, maximum_output_tokens = limits[stage]
    return {
        "maximum_input_tokens": maximum_input_tokens,
        "maximum_output_tokens": maximum_output_tokens,
        "execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V2",
        "execution_envelope_sha256": (
            "58e1ebda33a63fba4cb5289d21531298a7803a765b3525214d45700bc993cc22"
        ),
        "failure_contract_id": "CORE_TRANSPORT_ATTEMPT_CONTRACT_V2",
        "failure_contract_sha256": (
            "1ee66fcb795f97d483c2ef976133ee61dbd5108c9dae851c2c2786ff496d788f"
        ),
        "terminal_failure_contract_id": "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1",
        "terminal_failure_contract_sha256": (
            "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
        ),
        "rate_card_sha256": (
            "50975b67dce4c59ba9267c3234a873076137ded5078aa3e8b5c9a2fad4ff3e06"
        ),
    }


def _call(
    stage: str = "full_history_generate",
    call_id: str = "call-1",
) -> MethodCall:
    messages = [{"role": "user", "content": "frozen request"}]
    return MethodCall(
        call_id=call_id,
        stage=stage,
        messages=messages,
        raw_response="answer",
        model="gpt-5.6-luna",
        temperature=0.0,
        top_p=1.0,
        latency_ms=7,
        token_usage={"prompt_tokens": 3, "completion_tokens": 2},
        transport_attempts=1,
        provider_cost_usd=0.01,
        provider_status="completed",
        provider_response_id=f"response-{call_id}",
        provider_usage={"input_tokens": 3, "output_tokens": 2},
        provider_service_tier="default",
        provider_returned_model="gpt-5.6-luna",
        provider_response_status="completed",
        provider_request_contract=_request_contract(messages, stage),
        provider_authority_contract=_authority_contract(stage),
        authoritative_provider_cost_usd=0.01,
        derived_cost_usd=0.01,
        provider_cost_source="AUTHORITATIVE_PROVIDER",
    )


def _evidence() -> dict[str, JsonValue]:
    return _evidence_for(_unit())


def _evidence_for(
    unit: ExecutionUnit,
    *,
    verifier_result: bool | None = True,
) -> dict[str, JsonValue]:
    return {
        "evidence_kind": "CLEAN_PREFIX",
        "prefix_unit_id": unit.unit_id,
        "checkpoint": {
            "schema_version": "phase13_main_prefix_checkpoint_v1",
            "baseline": unit.memory_baseline,
            "checkpoint_id": "checkpoint-1",
            "checkpoint_identity_sha256": "d" * 64,
            "canonical_sha256": "e" * 64,
            "canonical_state_utf8": "{}",
        },
        "runtime_evidence": {
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
            "verifier_result": verifier_result,
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
    }


def _bot_unit() -> ExecutionUnit:
    return ExecutionUnit(
        0,
        "1" * 64,
        "CLEAN_PREFIX",
        0,
        "game24",
        "bot_style",
        "NOT_APPLICABLE",
        None,
        3,
        "game24|bot_style|prefix",
        "f" * 64,
        "a" * 64,
        "b" * 64,
    )


def _reflexion_unit() -> ExecutionUnit:
    return ExecutionUnit(
        0,
        "2" * 64,
        "CLEAN_PREFIX",
        0,
        "game24",
        "reflexion_style",
        "NOT_APPLICABLE",
        None,
        2,
        "game24|reflexion_style|prefix",
        "f" * 64,
        "a" * 64,
        "b" * 64,
    )


def test_live_contract_rejects_missing_prefix_ownership(tmp_path: Path) -> None:
    payload = _contract_payload()
    prefix = payload["prefix"]
    assert isinstance(prefix, dict)
    del prefix["owner_law_id"]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MainLiveContractError, match="MAIN_LIVE_CONTRACT_INVALID"):
        load_main_live_contract(path)


def test_live_contract_derives_prefix_and_stage_multiplicity(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_contract_payload()), encoding="utf-8")

    contract = load_main_live_contract(path)

    assert contract.prefix.realization_count == 230
    assert sum(contract.prefix.stage_call_counts.values()) == 430


def test_unit_completion_is_returned_only_after_durable_evidence(tmp_path: Path) -> None:
    completed = persist_unit_dispatch(
        tmp_path,
        _unit(),
        MainUnitDispatchOutput(
            evidence=_evidence(),
            provider_calls=(_call(),),
            realized_cost_krw=16,
        ),
    )

    evidence_path = tmp_path / "units" / f"000000-{_unit().unit_id}.json"
    raw = evidence_path.read_bytes()
    assert completed.evidence_sha256 == hashlib.sha256(raw).hexdigest()
    assert completed.realized_cost_krw == 16
    assert json.loads(raw)["unit_id"] == _unit().unit_id


def test_durable_dispatch_adapts_backend_output_to_runner_completion(tmp_path: Path) -> None:
    dispatched: list[str] = []

    def backend(unit: ExecutionUnit) -> MainUnitDispatchOutput:
        dispatched.append(unit.unit_id)
        return MainUnitDispatchOutput(
            evidence=_evidence(),
            provider_calls=(_call(),),
            realized_cost_krw=16,
        )

    completed = DurableMainDispatch(tmp_path, backend)(_unit())

    assert dispatched == [_unit().unit_id]
    assert completed.realized_cost_krw == 16
    assert (tmp_path / "units" / f"000000-{_unit().unit_id}.json").is_file()


def test_durable_dispatch_rejects_backend_cost_mismatch(tmp_path: Path) -> None:
    def backend(_unit: ExecutionUnit) -> MainUnitDispatchOutput:
        return MainUnitDispatchOutput(
            evidence=_evidence(),
            provider_calls=(_call(),),
            realized_cost_krw=20,
        )

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_REALIZED_COST_MISMATCH"):
        DurableMainDispatch(tmp_path, backend)(_unit())


def test_durable_dispatch_rejects_zero_call_success(tmp_path: Path) -> None:
    def backend(_unit: ExecutionUnit) -> MainUnitDispatchOutput:
        return MainUnitDispatchOutput(
            evidence=_evidence(),
            provider_calls=(),
            realized_cost_krw=0,
        )

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_PROVIDER_CALLS_INVALID"):
        DurableMainDispatch(tmp_path, backend)(_unit())


def test_durable_dispatch_rejects_incorrect_stage(tmp_path: Path) -> None:
    call = _call().model_copy(update={"stage": "rag_generate"})

    def backend(_unit: ExecutionUnit) -> MainUnitDispatchOutput:
        return MainUnitDispatchOutput(
            evidence=_evidence(),
            provider_calls=(call,),
            realized_cost_krw=16,
        )

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_PROVIDER_CALLS_INVALID"):
        DurableMainDispatch(tmp_path, backend)(_unit())


@pytest.mark.parametrize(
    "stages",
    (
        ("reflexion_generate",),
        ("reflexion_generate", "reflexion_reflect"),
    ),
)
def test_reflexion_prefix_accepts_failure_dependent_reflection_stage(
    tmp_path: Path,
    stages: tuple[str, ...],
) -> None:
    unit = _reflexion_unit()
    calls = tuple(_call(stage, f"call-{index}") for index, stage in enumerate(stages, start=1))

    completed = persist_unit_dispatch(
        tmp_path,
        unit,
        MainUnitDispatchOutput(
            evidence=_evidence_for(unit, verifier_result=len(stages) == 1),
            provider_calls=calls,
            realized_cost_krw=16 * len(calls),
        ),
    )

    assert completed.realized_cost_krw == 16 * len(calls)


def test_reflexion_prefix_rejects_unknown_verifier_result(tmp_path: Path) -> None:
    unit = _reflexion_unit()
    calls = (
        _call("reflexion_generate", "call-1"),
        _call("reflexion_reflect", "call-2"),
    )

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_PROVIDER_CALLS_INVALID"):
        persist_unit_dispatch(
            tmp_path,
            unit,
            MainUnitDispatchOutput(
                evidence=_evidence_for(unit, verifier_result=None),
                provider_calls=calls,
                realized_cost_krw=32,
            ),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"call_id": None},
        {"model": "substituted-model"},
        {"provider_returned_model": "substituted-model"},
        {"provider_service_tier": "flex"},
        {"transport_attempts": 2},
        {"retry_count": 1},
        {"provider_response_id": None},
        {"provider_usage": None},
        {"provider_request_contract": None},
        {"provider_authority_contract": None},
        {
            "provider_request_contract": {
                **_request_contract(
                    [{"role": "user", "content": "frozen request"}],
                    "full_history_generate",
                ),
                "model": "substituted-model",
            }
        },
        {
            "provider_authority_contract": {
                **_authority_contract("full_history_generate"),
                "failure_contract_sha256": "0" * 64,
            }
        },
    ],
)
def test_completed_call_must_match_frozen_provider_contract(
    tmp_path: Path,
    updates: dict[str, JsonValue],
) -> None:
    call = _call().model_copy(update=updates)

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_PROVIDER_CALLS_INVALID"):
        persist_unit_dispatch(
            tmp_path,
            _unit(),
            MainUnitDispatchOutput(
                evidence=_evidence(),
                provider_calls=(call,),
                realized_cost_krw=16,
            ),
        )


def test_completed_call_rejects_consistently_mutated_top_p(tmp_path: Path) -> None:
    call = _call()
    request = dict(call.provider_request_contract or {})
    request["top_p"] = 0.5
    mutated = call.model_copy(
        update={"top_p": 0.5, "provider_request_contract": request}
    )

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_PROVIDER_CALLS_INVALID"):
        persist_unit_dispatch(
            tmp_path,
            _unit(),
            MainUnitDispatchOutput(
                evidence=_evidence(),
                provider_calls=(mutated,),
                realized_cost_krw=16,
            ),
        )


def test_provider_call_ids_must_be_unique_within_unit(tmp_path: Path) -> None:
    unit = _bot_unit()
    calls = (
        _call("bot_problem_distill", "duplicate"),
        _call("bot_instantiate_solve", "duplicate"),
        _call("bot_thought_distill", "call-3"),
    )

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_PROVIDER_CALLS_INVALID"):
        persist_unit_dispatch(
            tmp_path,
            unit,
            MainUnitDispatchOutput(
                evidence=_evidence_for(unit),
                provider_calls=calls,
                realized_cost_krw=48,
            ),
        )


def test_durable_dispatch_rejects_runtime_identity_for_different_task(tmp_path: Path) -> None:
    evidence = _evidence()
    runtime = evidence["runtime_evidence"]
    assert isinstance(runtime, dict)
    runtime["task"] = "word_sorting"

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_RUNTIME_IDENTITY_INVALID"):
        persist_unit_dispatch(
            tmp_path,
            _unit(),
            MainUnitDispatchOutput(
                evidence=evidence,
                provider_calls=(_call(),),
                realized_cost_krw=16,
            ),
        )


def test_unit_evidence_cannot_be_overwritten(tmp_path: Path) -> None:
    output = MainUnitDispatchOutput(
        evidence=_evidence(),
        provider_calls=(_call(),),
        realized_cost_krw=16,
    )
    persist_unit_dispatch(tmp_path, _unit(), output)

    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_EVIDENCE_ALREADY_EXISTS"):
        persist_unit_dispatch(tmp_path, _unit(), output)


def test_telemetry_summary_joins_units_calls_tokens_latency_and_cost(tmp_path: Path) -> None:
    persist_unit_dispatch(
        tmp_path,
        _unit(),
        MainUnitDispatchOutput(
            evidence=_evidence(),
            provider_calls=(_call(),),
            realized_cost_krw=16,
        ),
    )

    summary = summarize_telemetry(tmp_path)

    assert summary.unit_count == 1
    assert summary.provider_call_count == 1
    assert summary.transport_attempt_count == 1
    assert summary.latency_ms == 7
    assert summary.token_usage == {"completion_tokens": 2, "prompt_tokens": 3}
    assert summary.provider_cost_usd == "0.01"
    assert summary.realized_cost_krw == 16


def test_live_environment_loads_approved_dotenv_without_shell_sourcing(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# comment\nexport OPENAI_API_KEY='secret-value'\n", encoding="utf-8")

    environment = load_live_environment(path, required_keys=("OPENAI_API_KEY",))

    assert environment["OPENAI_API_KEY"] == "secret-value"


def test_live_cli_validates_bound_contract_without_ledger(tmp_path: Path) -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "memcontam.readiness.phase13_main_live_cli",
            "validate",
            "--repository-root",
            str(ROOT),
            "--package",
            str(P5),
            "--authorization",
            str(P6),
            "--expected-authorization-sha256",
            hashlib.sha256(P6.read_bytes()).hexdigest(),
            "--cache-root",
            str(Path.home() / ".cache/huggingface/hub"),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "authorization_id": "phase13-main-a-authorized-execution-v1",
        "main_a_status": "NOT_STARTED",
        "prefix_count": 230,
        "provider_calls_issued": 0,
        "status": "READY_NO_CALLS",
        "unit_count": 1200,
    }
    assert not (tmp_path / "offline-qa" / "main-run-v1.sqlite3").exists()
