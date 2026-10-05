from __future__ import annotations

import pytest
import httpx
import json
from openai import APIStatusError

from memcontam.readiness.phase13_main_request_dispatch import DispatchTechnicalFailureV3, ProductionRequestDispatcherV3

from .test_phase13_v3_provider_count import counted as counted


@pytest.mark.parametrize("field,value", (
    ("provider_response_id", "ambiguous-response"),
    ("provider_status", "incomplete"),
    ("provider_usage", {"input_tokens": 1}),
    ("authoritative_provider_cost_usd", 0.001),
))
def test_response_or_acknowledgement_ambiguity_never_spends_retry(counted, field, value):
    failure = TimeoutError("request may have been accepted")
    setattr(failure, field, value)
    counted.seen.generation_failure = failure
    dispatcher = ProductionRequestDispatcherV3(
        counted.ledger, counted.dispatcher.binding, counted.dispatcher.parents,
        provider_factory=lambda _: counted.provider,
        retry_entitlements=frozenset({counted.key.dispatch_id}),
    )

    with pytest.raises(DispatchTechnicalFailureV3):
        dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)

    assert (counted.seen.counts, counted.seen.creates) == (1, 1)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"


def test_unacknowledged_timeout_cannot_resample_entitled_request(counted):
    counted.seen.generation_failure = TimeoutError("no unambiguous failure acknowledgement")
    dispatcher = ProductionRequestDispatcherV3(
        counted.ledger, counted.dispatcher.binding, counted.dispatcher.parents,
        provider_factory=lambda _: counted.provider,
        retry_entitlements=frozenset({counted.key.dispatch_id}),
    )

    with pytest.raises(DispatchTechnicalFailureV3):
        dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)

    assert (counted.seen.counts, counted.seen.creates) == (1, 1)


def test_acknowledged_pre_payload_http_failure_retries_same_request_once(counted):
    response = httpx.Response(429, request=httpx.Request("POST", "https://denied.invalid/v1/responses"))
    counted.seen.generation_failure = APIStatusError("rate limited", response=response, body=None)
    dispatcher = ProductionRequestDispatcherV3(
        counted.ledger, counted.dispatcher.binding, counted.dispatcher.parents,
        provider_factory=lambda _: counted.provider,
        retry_entitlements=frozenset({counted.key.dispatch_id}),
    )

    assert dispatcher.dispatch(counted.key, lambda: counted.material,
                               lambda result: result.content) == "final: 24"
    assert (counted.seen.counts, counted.seen.creates) == (1, 2)
    assert counted.ledger.state(counted.key.dispatch_id).kind == "COMPLETED"
    attempts = [json.loads(row) for row in counted.ledger.rows()
                if json.loads(row)["kind"] == "ATTEMPT_STARTED"]
    assert len({row["attempt_id"] for row in attempts}) == 2
    assert all(len(row["attempt_id"]) == 64 for row in attempts)
    from memcontam.readiness.phase13_v3_terminal_models import AttemptStartedV3
    with pytest.raises(ValueError, match="MAIN_RETRY_ATTEMPT_IDENTITY_MISMATCH"):
        AttemptStartedV3.model_validate({**attempts[1], "attempt_id": attempts[0]["attempt_id"]})


def test_local_approximation_cannot_replace_provider_input_gate(counted, monkeypatch):
    import memcontam.readiness.phase13_main_request_dispatch as dispatch
    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: 379)
    counted.seen.tokens = 378

    assert counted.dispatcher.dispatch(counted.key, lambda: counted.material,
                                       lambda result: result.content) == "final: 24"
    assert (counted.seen.counts, counted.seen.creates) == (1, 1)
