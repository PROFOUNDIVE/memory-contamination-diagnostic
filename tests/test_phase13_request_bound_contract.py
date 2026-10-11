from dataclasses import replace
import json

import pytest

from memcontam.readiness.phase13_main_request_dispatch import DispatchTechnicalFailureV3
from memcontam.readiness.phase13_v3_terminal_models import TerminalEvidenceError

from .test_phase13_v3_provider_count import counted as counted


@pytest.mark.parametrize("base_url", [
    "https://api.openai.com/v1", "https://invalid.example/v1",
    "https://fake.invalid.example/v1", "https://fake.invalid@api.openai.com/v1",
])
@pytest.mark.parametrize("tokens", [374, 378, 379])
def test_unproved_live_bound_rejects_before_count_or_generation(counted, monkeypatch, base_url, tokens):
    identity = counted.provider.count_identity_v3().model_copy(update={"base_url": base_url})
    monkeypatch.setattr(counted.provider, "count_identity_v3", lambda: identity)
    counted.seen.tokens = tokens

    with pytest.raises(TerminalEvidenceError, match="MAIN_COMPLETE_REQUEST_BOUND_UNPROVEN"):
        counted.dispatcher.receive(counted.key, lambda: counted.material)

    assert (counted.seen.counts, counted.seen.creates) == (0, 0)
    assert counted.ledger.count_record(counted.key.dispatch_id, "count-started") is None


def test_ack_contract_failure_preserves_valid_received_cost(counted):
    response = counted.dispatcher.receive(counted.key, lambda: counted.material)
    changed = replace(response, raw={**response.raw, "model": "wrong-model"})

    with pytest.raises(DispatchTechnicalFailureV3) as failure:
        counted.dispatcher.acknowledge(counted.key, changed, semantic_success=True)

    state = counted.ledger.state(counted.key.dispatch_id)
    assert state.kind == "ATTEMPTED_PROVIDER_FAILURE"
    assert state.attempt_costs[-1].usage.input_tokens == counted.seen.tokens
    assert failure.value.realized_cost_krw == 1


@pytest.mark.parametrize("deferred", [False, True])
def test_invalid_monetary_receipt_preserves_usage_for_reconciliation(counted, monkeypatch, deferred):
    send = counted.provider.send_compiled_v3

    def altered(compiled, before_request):
        response = send(compiled, before_request)
        return replace(response, raw={**response.raw,
            "authoritative_provider_cost_usd": "invalid", "currency": "USD"})

    monkeypatch.setattr(counted.provider, "send_compiled_v3", altered)

    with pytest.raises(DispatchTechnicalFailureV3):
        if deferred:
            counted.dispatcher.receive(counted.key, lambda: counted.material)
        else:
            counted.dispatcher.dispatch(counted.key, lambda: counted.material,
                lambda response: pytest.fail("invalid cost reached semantic parser"))

    state = counted.ledger.state(counted.key.dispatch_id)
    assert state.kind == "ATTEMPTED_PROVIDER_FAILURE"
    observation = json.loads(counted.ledger.read_record(f"{counted.key.dispatch_id}.observation.json"))
    assert observation["provider_cost_usd"] == "invalid"
    assert observation["usage"]["input_tokens"] == counted.seen.tokens


def test_ack_invalid_cost_preserves_rejected_receipt_in_observation(counted):
    response = counted.dispatcher.receive(counted.key, lambda: counted.material)
    changed = replace(response, raw={**response.raw,
        "authoritative_provider_cost_usd": "invalid", "currency": "USD"})

    with pytest.raises(DispatchTechnicalFailureV3):
        counted.dispatcher.acknowledge(counted.key, changed, semantic_success=True)

    observation = json.loads(counted.ledger.read_record(f"{counted.key.dispatch_id}.observation.json"))
    assert observation["provider_cost_usd"] == "invalid"
    assert observation["usage"]["input_tokens"] == counted.seen.tokens
