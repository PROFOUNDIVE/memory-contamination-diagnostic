from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from memcontam.readiness.phase13_main_request_dispatch import (
    DispatchTechnicalFailureV3, ProductionRequestDispatcherV3,
)
from memcontam.readiness.phase13_v3_terminal_models import TerminalEvidenceError
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
from memcontam.readiness.phase13_v3_count import count_projection
from memcontam.readiness.phase13_v3_request import compile_request_bytes

from .test_phase13_v3_provider_count import counted as counted


def test_unknown_exception_cannot_self_register_retry_entitlement(counted, monkeypatch):
    dispatcher = ProductionRequestDispatcherV3(
        counted.ledger, counted.dispatcher.binding, counted.dispatcher.parents,
        provider_factory=lambda _: counted.provider,
        retry_entitlements=frozenset({counted.key.dispatch_id}),
    )

    def failed(compiled, before_request):
        before_request()
        error = RuntimeError("untrusted fault")
        error.phase13_retry_class = "TIMEOUT_BEFORE_SEMANTIC_PAYLOAD"
        error.provider_failure_acknowledged = True
        raise error

    monkeypatch.setattr(counted.provider, "send_compiled_v3", failed)
    with pytest.raises(RuntimeError, match="untrusted fault"):
        dispatcher.receive(counted.key, lambda: counted.material)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPT_STARTED"
    assert sum(b'"kind":"ATTEMPT_STARTED"' in row for row in counted.ledger.rows()) == 1


def test_native_verifier_failure_outcome_is_not_semantic_rejection(counted):
    from memcontam.baselines.contracts import BaselineExecutionOutcome
    from memcontam.experiment.phase12.runtime_registry import NOMEM_SINGLETON, RuntimeTrialResult
    from memcontam.readiness.phase13_main_request_client import MainRequestClientV3

    client = MainRequestClientV3(counted.dispatcher, counted.key.parent_id, lambda: None)

    def native():
        client.chat([message.model_dump() for message in counted.material.messages],
                    "gpt-5.6-luna", {"method_stage": counted.key.stage})
        return RuntimeTrialResult(BaselineExecutionOutcome(
            "failed", error_type="VerifierContractError", failure_disposition="verifier_contract_failed",
            scientific_ineligibility_reason="verifier_contract_failed"), NOMEM_SINGLETON)

    with pytest.raises(TerminalEvidenceError, match="MAIN_RUN_POST_INTENT_RUNTIME_FAILURE"):
        client.trial(native, lambda: counted.material.native_state)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPT_STARTED"


@pytest.mark.parametrize("fault", [ValueError, KeyError, TypeError, RuntimeError])
def test_native_caller_fault_does_not_acknowledge_semantic_rejection(counted, fault):
    from memcontam.readiness.phase13_main_request_client import MainRequestClientV3

    client = MainRequestClientV3(counted.dispatcher, counted.key.parent_id, lambda: None)

    def native():
        client.chat([message.model_dump() for message in counted.material.messages],
                    "gpt-5.6-luna", {"method_stage": counted.key.stage})
        raise fault("native source defect")

    with pytest.raises(fault):
        client.trial(native, lambda: counted.material.native_state)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPT_STARTED"
    counted.dispatcher.recover()
    assert counted.ledger.state(counted.key.dispatch_id).kind == "AMBIGUOUS_ATTEMPT"
    assert counted.seen.creates == 1


@pytest.mark.parametrize("point", ["_usage_dict", "_output_text", "record_usage"])
def test_post_sdk_response_source_fault_is_not_provider_missingness(counted, monkeypatch, point):
    from memcontam.clients import openai_responses

    def fault(*_args, **_kwargs):
        raise RuntimeError("secret normalization source defect")

    owner = counted.provider.cost_guard if point == "record_usage" else openai_responses
    monkeypatch.setattr(owner, point, fault)
    with pytest.raises(RuntimeError):
        counted.dispatcher.receive(counted.key, lambda: counted.material)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPT_STARTED"
    raw = counted.ledger.read_record(f"{counted.key.dispatch_id}.implementation-observation.json")
    observed = json.loads(raw)
    assert observed["response_id"] == "fake-response"
    assert observed["failure_code"] == "MAIN_RESPONSE_NORMALIZATION_FAILED"
    assert b"secret" not in raw
    counted.dispatcher.recover()
    assert counted.ledger.state(counted.key.dispatch_id).kind == "AMBIGUOUS_ATTEMPT"
    assert counted.seen.creates == 1


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("override", [
    {"authoritative_provider_cost_usd": "NaN", "currency": "USD"},
    {"authoritative_provider_cost_usd": "0.001", "currency": "EUR"},
    {"authoritative_provider_cost_usd": "0.001", "currency": "USD"},
    {"usage": {"input_tokens": 1, "output_tokens": 1,
               "input_tokens_details": {"cached_tokens": 2}}},
    {"usage": {"input_tokens": 1, "output_tokens": 1,
               "input_tokens_details": None}},
    {"usage": []},
])
def test_bad_cost_terminalizes_before_semantic_use(counted, monkeypatch, deferred, override):
    send = counted.provider.send_compiled_v3

    def altered(compiled, before_request):
        response = send(compiled, before_request)
        return replace(response, raw={**response.raw, **override})

    monkeypatch.setattr(counted.provider, "send_compiled_v3", altered)
    parsed = []
    with pytest.raises(DispatchTechnicalFailureV3):
        if deferred:
            counted.dispatcher.receive(counted.key, lambda: counted.material)
        else:
            counted.dispatcher.dispatch(counted.key, lambda: counted.material,
                                        lambda response: parsed.append(response.content))
    assert parsed == []
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    assert counted.seen.creates == 1
    reopened = TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)
    try:
        assert reopened.state(counted.key.dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    finally:
        reopened.close()


def test_failed_ack_preserves_hash_without_raw_response(counted):
    response = counted.dispatcher.receive(counted.key, lambda: counted.material)
    with pytest.raises(DispatchTechnicalFailureV3):
        counted.dispatcher.acknowledge(counted.key, response, semantic_success=False)
    raw = counted.ledger.read_record(f"{counted.key.dispatch_id}.observation.json")
    observation = json.loads(raw)
    assert "response" not in observation
    assert observation["response_sha256"] == hashlib.sha256(response.content.encode()).hexdigest()
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"


def test_ack_rejects_response_contract_mutation(counted):
    response = counted.dispatcher.receive(counted.key, lambda: counted.material)
    changed = replace(response, raw={**response.raw, "model": "wrong-model"})
    with pytest.raises(DispatchTechnicalFailureV3):
        counted.dispatcher.acknowledge(counted.key, changed, semantic_success=True)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"


@pytest.mark.parametrize("untrusted_code", ["secret-prompt-response", {"raw": "secret"},
                                          "MAIN_SEMANTIC_RESULT_UNAVAILABLE"])
def test_arbitrary_exception_code_is_not_durable_telemetry(counted, monkeypatch, untrusted_code):
    send = counted.provider.send_compiled_v3

    def failed(compiled, before_request):
        send(compiled, before_request)
        error = RuntimeError("untrusted provider text")
        setattr(error, "code", untrusted_code)
        raise error

    monkeypatch.setattr(counted.provider, "send_compiled_v3", failed)
    with pytest.raises(RuntimeError, match="untrusted provider text"):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    event = json.loads(counted.ledger.rows()[-1])
    assert event["kind"] == "ATTEMPT_STARTED"
    assert b"untrusted provider text" not in b"".join(counted.ledger.rows())


@pytest.mark.parametrize("fault", [RuntimeError, TimeoutError, TerminalEvidenceError])
def test_parser_source_fault_stops_without_terminal_missingness_or_redispatch(counted, fault):
    def broken_parser(response):
        raise fault("source fault, not a semantic parse rejection")

    with pytest.raises(fault, match="source fault"):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, broken_parser)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPT_STARTED"
    reopened = TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)
    try:
        dispatcher = ProductionRequestDispatcherV3(
            reopened, counted.dispatcher.binding, counted.dispatcher.parents,
            provider_factory=lambda _: counted.provider,
        )
        dispatcher.recover()
        assert reopened.state(counted.key.dispatch_id).kind == "AMBIGUOUS_ATTEMPT"
        with pytest.raises((ValueError, RuntimeError)):
            dispatcher.dispatch(counted.key, lambda: counted.material, broken_parser)
    finally:
        reopened.close()
    assert counted.seen.creates == 1


def test_registered_semantic_rejection_remains_terminal(counted):
    with pytest.raises(DispatchTechnicalFailureV3):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: None)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    assert json.loads(counted.ledger.rows()[-1])["failure_code"] == "MAIN_SEMANTIC_RESULT_UNAVAILABLE"


def test_valid_wrong_answer_is_completed_not_technical_missingness(counted):
    assert counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: False) is False
    assert counted.ledger.state(counted.key.dispatch_id).kind == "COMPLETED"


@pytest.mark.parametrize("field,value", [("model", None), ("service_tier", None),
    ("cost_usd", float("nan")), ("cost_usd", -1), ("cost_usd", "invalid"),
    ("cost_usd", True)])
def test_sdk_does_not_mask_missing_identity_or_invalid_cost(counted, monkeypatch, field, value):
    create = counted.provider.client.create

    def altered(**body):
        response = create(**body)
        if value is None:
            delattr(response, field)
        else:
            setattr(response, field, value)
        return response

    monkeypatch.setattr(counted.provider.client, "create", altered)
    parsed = []
    with pytest.raises(DispatchTechnicalFailureV3):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material,
                                    lambda response: parsed.append(response.content))
    assert parsed == []
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"


def test_projected_374_does_not_prove_charged_input_below_378(counted, monkeypatch):
    counted.seen.tokens = 374
    create = counted.provider.client.create

    def charged(**body):
        response = create(**body)
        response.usage = {"input_tokens": 380, "output_tokens": 1}
        return response

    monkeypatch.setattr(counted.provider.client, "create", charged)
    with pytest.raises(DispatchTechnicalFailureV3):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material,
                                    lambda response: pytest.fail("over-gate response reached semantics"))
    assert (counted.seen.counts, counted.seen.creates) == (1, 1)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"


def test_projection_collides_for_distinct_compiled_options(counted):
    counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    original = counted.dispatcher.compiled_request(counted.key)
    material = counted.material.model_copy(update={"temperature": 0.7, "top_p": 0.9})
    changed = replace(original, material=material, request_bytes=compile_request_bytes(counted.key, material))
    assert original.request_bytes != changed.request_bytes
    assert count_projection(original) == count_projection(changed)


@pytest.mark.parametrize("incomplete", [False, True])
def test_sdk_preserves_valid_cost_currency_through_terminal_disposition(counted, monkeypatch, incomplete):
    create = counted.provider.client.create

    def priced(**body):
        response = create(**body)
        response.cost_usd = "0.0000768"
        response.currency = "USD"
        if incomplete:
            response.status = "incomplete"
        return response

    monkeypatch.setattr(counted.provider.client, "create", priced)
    if incomplete:
        with pytest.raises(DispatchTechnicalFailureV3) as failure:
            counted.dispatcher.receive(counted.key, lambda: counted.material)
        assert failure.value.realized_cost_krw == 1
    else:
        assert counted.dispatcher.dispatch(counted.key, lambda: counted.material,
                                            lambda response: response.content) == "final: 24"
    assert counted.ledger.state(counted.key.dispatch_id).attempt_costs[-1].currency == "USD"
