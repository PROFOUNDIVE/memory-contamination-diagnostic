from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.logging.schema import MethodCall
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.readiness.phase13_cost_policy import load_cost_policy_bundle
from memcontam.readiness.phase13_main_live_dispatch import (
    DurableMainDispatch,
    MainUnitDispatchOutput,
    summarize_telemetry,
)
from memcontam.readiness.phase13_main_production import ProductionObject, UnitKind
from memcontam.readiness.phase13_main_production_backend import (
    MainProductionBackend,
    MainProductionBackendError,
    OrdinaryRuntimeRequest,
    PrefixRuntimeOutput,
)


ROOT = Path(__file__).resolve().parents[1]
_COST_POLICY = load_cost_policy_bundle(ROOT)
_RATE_CARD_SHA256 = hashlib.sha256(
    json.dumps(
        _COST_POLICY.proof.rate_card.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _unit(
    sequence: int,
    kind: UnitKind,
    *,
    baseline: str | None,
    arm: str,
    prefix_unit_id: str | None,
) -> ProductionObject:
    return ProductionObject(
        sequence=sequence,
        unit_id=f"{sequence + 1:064x}",
        kind=kind,
        seed=0,
        task="game24",
        memory_baseline=baseline,
        arm=arm,
        prefix_unit_id=prefix_unit_id,
        projected_cost_krw=1,
        execution_template_id=(
            f"game24|{baseline}|prefix"
            if kind == "CLEAN_PREFIX"
            else "game24|nomem"
            if kind == "NO_MEMORY_SINGLETON"
            else f"game24|{baseline}|{arm}"
        ),
        ordered_sample_ids_sha256="f" * 64,
        registration_packet_sha256="a" * 64,
        checkpoint_registry_sha256="b" * 64,
    )


def _checkpoint():
    entry = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"]
    snapshot = entry.serialize_state(FullHistoryStateV3(records=[]))
    assert isinstance(snapshot, NativeState)
    return serialize_checkpoint(snapshot)


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


def _output(
    unit: ProductionObject,
    label: str,
    stages: tuple[str, ...] = (),
) -> MainUnitDispatchOutput:
    return MainUnitDispatchOutput(
        evidence={
            "unit_id": unit.unit_id,
            "task": unit.task,
            "seed": unit.seed,
            "memory_baseline": unit.memory_baseline,
            "arm": unit.arm,
            "production_identity": {
                "execution_template_id": unit.execution_template_id,
                "trajectory_seed": unit.seed,
                "concrete_seed_id": str(unit.seed),
                "ordered_sample_ids_sha256": "f" * 64,
                "registration_packet_sha256": "a" * 64,
                "scientific_result": False,
                "checkpoint_registry_sha256": "b" * 64,
            },
            "observability_registration_packet_sha256": "a" * 64,
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
        provider_calls=tuple(
            _completed_call(f"{label}-{index}", stage)
            for index, stage in enumerate(stages)
        ),
        realized_cost_krw=2 * len(stages),
    )


def test_prefix_checkpoint_is_durable_and_all_four_consumers_load_exact_identity(
    tmp_path: Path,
) -> None:
    prefix = _unit(
        0,
        "CLEAN_PREFIX",
        baseline="fh_bounded",
        arm="NOT_APPLICABLE",
        prefix_unit_id=None,
    )
    checkpoint = _checkpoint()
    requests: list[OrdinaryRuntimeRequest] = []

    def execute_prefix(unit: ProductionObject) -> PrefixRuntimeOutput:
        assert unit == prefix
        return PrefixRuntimeOutput(
            checkpoint,
            _output(unit, "prefix", ("full_history_generate",)),
        )

    def execute_ordinary(request: OrdinaryRuntimeRequest) -> MainUnitDispatchOutput:
        requests.append(request)
        return _output(request.unit, request.arm, ("full_history_generate",) * 50)

    completed: dict[str, str] = {}
    dispatch = DurableMainDispatch(
        tmp_path,
        MainProductionBackend(
            tmp_path,
            execute_prefix,
            execute_ordinary,
            completed.__getitem__,
        ),
    )
    prefix_completion = dispatch(prefix)
    completed[prefix.unit_id] = prefix_completion.evidence_sha256
    consumers = tuple(
        _unit(
            index,
            "MEMORY_BEARING",
            baseline="fh_bounded",
            arm=arm,
            prefix_unit_id=prefix.unit_id,
        )
        for index, arm in enumerate(("clean", "correct", "irrelevant", "contam"), start=1)
    )
    for consumer in consumers:
        dispatch(consumer)

    assert not (tmp_path / "checkpoints").exists()
    assert (tmp_path / "units" / f"000000-{prefix.unit_id}.json").is_file()
    assert tuple(request.arm for request in requests) == (
        "clean",
        "correct",
        "irrelevant",
        "contam",
    )
    assert all(request.checkpoint is not None for request in requests)
    assert {request.checkpoint.identity for request in requests if request.checkpoint} == {
        checkpoint.identity
    }
    assert {request.prefix_unit_id for request in requests} == {prefix.unit_id}
    memory_record = json.loads(
        (tmp_path / "units" / f"000001-{consumers[0].unit_id}.json").read_text()
    )
    prefix_record = json.loads(
        (tmp_path / "units" / f"000000-{prefix.unit_id}.json").read_text()
    )
    assert prefix_record["evidence"]["checkpoint"]["canonical_state_utf8"]
    assert memory_record["evidence"]["prefix_unit_id"] == prefix.unit_id
    assert memory_record["evidence"]["consumed_checkpoint_id"] == checkpoint.identity.checkpoint_id
    assert (
        memory_record["evidence"]["consumed_checkpoint_identity_sha256"]
        == checkpoint.identity.sha256
    )
    assert (
        memory_record["evidence"]["consumed_checkpoint_canonical_sha256"]
        == checkpoint.canonical_sha256
    )


def test_nomem_uses_explicit_clean_adapter_without_prefix(tmp_path: Path) -> None:
    requests: list[OrdinaryRuntimeRequest] = []

    def reject_prefix(_unit: ProductionObject) -> PrefixRuntimeOutput:
        pytest.fail("NoMem must not execute a prefix")

    def execute_ordinary(request: OrdinaryRuntimeRequest) -> MainUnitDispatchOutput:
        requests.append(request)
        return _output(request.unit, "nomem", ("no_memory_generate",) * 50)

    unit = _unit(
        0,
        "NO_MEMORY_SINGLETON",
        baseline=None,
        arm="NOT_APPLICABLE",
        prefix_unit_id=None,
    )
    DurableMainDispatch(
        tmp_path,
        MainProductionBackend(tmp_path, reject_prefix, execute_ordinary),
    )(unit)

    assert len(requests) == 1
    assert requests[0].baseline == "nomem"
    assert requests[0].arm == "clean"
    assert requests[0].scientific_arm == "NOT_APPLICABLE"
    assert requests[0].prefix_unit_id is None
    assert requests[0].checkpoint is None
    assert not (tmp_path / "checkpoints").exists()
    record = json.loads((tmp_path / "units" / f"000000-{unit.unit_id}.json").read_text())
    assert record["evidence"]["internal_baseline"] == "nomem"
    assert record["evidence"]["internal_arm"] == "clean"
    assert record["evidence"]["scientific_arm"] == "NOT_APPLICABLE"
    summary = summarize_telemetry(tmp_path)
    assert summary.provider_call_count == 50
    assert summary.provider_cost_usd == "0.05"
    assert summary.realized_cost_krw == 100


@pytest.mark.parametrize("failure", ["absent", "mismatched"])
def test_memory_consumer_fails_closed_for_invalid_checkpoint(
    tmp_path: Path,
    failure: str,
) -> None:
    prefix = _unit(
        0,
        "CLEAN_PREFIX",
        baseline="fh_bounded",
        arm="NOT_APPLICABLE",
        prefix_unit_id=None,
    )
    ordinary_calls = 0

    def execute_prefix(_unit: ProductionObject) -> PrefixRuntimeOutput:
        return PrefixRuntimeOutput(
            _checkpoint(),
            _output(_unit, "prefix", ("full_history_generate",)),
        )

    def execute_ordinary(_request: OrdinaryRuntimeRequest) -> MainUnitDispatchOutput:
        nonlocal ordinary_calls
        ordinary_calls += 1
        return _output(_request.unit, "ordinary")

    completed: dict[str, str] = {}
    backend = MainProductionBackend(
        tmp_path,
        execute_prefix,
        execute_ordinary,
        completed.__getitem__,
    )
    if failure == "mismatched":
        prefix_completion = DurableMainDispatch(tmp_path, backend)(prefix)
        completed[prefix.unit_id] = prefix_completion.evidence_sha256
    consumer = _unit(
        1,
        "MEMORY_BEARING",
        baseline="bot_style" if failure == "mismatched" else "fh_bounded",
        arm="clean",
        prefix_unit_id=prefix.unit_id,
    )

    with pytest.raises(MainProductionBackendError, match="MAIN_PREFIX_CHECKPOINT_INVALID"):
        backend(consumer)

    assert ordinary_calls == 0
