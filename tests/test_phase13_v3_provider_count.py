from __future__ import annotations

import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from memcontam.clients import openai_responses
from memcontam.clients.config import ProviderConfig
from memcontam.readiness.phase13_main_request_dispatch import ProductionRequestDispatcherV3
from memcontam.readiness.phase13_v3_request import (
    CompiledProviderRequestV3, MessageV3, PackageBindingV3, ParentTrajectoryV3,
    RequestKeyV3, RequestMaterialV3, compile_request_bytes,
)
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3

from .phase13_corrective_identity import corrective_identity


class CountCrash(BaseException):
    pass


@pytest.fixture
def counted(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("external transport denied")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("A1_FAKE_KEY", "not-a-study-credential")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://denied.invalid/v1")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(httpx.Client, "send", denied)
    binding = PackageBindingV3(identity=corrective_identity(), package_sha256="b" * 64,
                               authorization_sha256="c" * 64)
    parent = ParentTrajectoryV3(parent_id="a" * 64, kind="NO_MEMORY_SINGLETON")
    key = RequestKeyV3(parent_id=parent.parent_id, stage="rag_generate", ordinal=0)
    ledger = TerminalLedgerV3.create(tmp_path / "requests.sqlite3", {
        "schema_version": "phase13_main_run_ledger_v3", "unit_ids": [key.dispatch_id],
        "identity": binding.identity.model_dump(mode="json"),
        "package_sha256": binding.package_sha256, "authorization_sha256": binding.authorization_sha256,
    })
    seen = SimpleNamespace(counts=0, creates=0, tokens=378, object="response.input_tokens",
                           failure=None, generation_failure=None, cost="0.001", count_body=None, create_body=None)

    class SDK:
        def __init__(self, **options):
            assert options["max_retries"] == 0
            self.base_url = options["base_url"]
            self.api_key = options["api_key"]
            self.organization = self.project = None
            self.responses = self
            self.input_tokens = self

        def count(self, **body):
            seen.counts += 1
            seen.count_body = body
            if seen.failure is not None:
                raise seen.failure
            return SimpleNamespace(object=seen.object, input_tokens=seen.tokens, cost_usd=seen.cost)

        def create(self, **body):
            assert ledger.state(key.dispatch_id).kind == "ATTEMPT_STARTED"
            seen.creates += 1
            seen.create_body = body
            if seen.generation_failure is not None:
                failure = seen.generation_failure
                seen.generation_failure = None
                raise failure
            return SimpleNamespace(output_text="final: 24", model="gpt-5.6-luna",
                status="completed", service_tier="default", id="fake-response",
                usage={"input_tokens": seen.tokens, "output_tokens": 1})

    monkeypatch.setattr(openai_responses, "OpenAI", SDK)
    provider = openai_responses.OpenAIResponsesClient(ProviderConfig(
        provider="openai_responses", api_key_env="A1_FAKE_KEY", base_url="https://denied.invalid/v1",
        timeout_seconds=180, live_calls_enabled=True, retries_after_initial_attempt=0,
        input_per_million_usd=0.20, cached_input_per_million_usd=0.02, output_per_million_usd=1.20,
    ), allow_live_calls=True, v3_binding=binding)
    dispatcher = ProductionRequestDispatcherV3(ledger, binding, (parent,), provider_factory=lambda _: provider)
    material = RequestMaterialV3(messages=(MessageV3(role="user", content="immutable fixture"),),
                                 native_state=b"native")
    yield SimpleNamespace(ledger=ledger, dispatcher=dispatcher, key=key, material=material,
                          provider=provider, seen=seen)
    ledger.close()


@pytest.mark.parametrize("tokens", [377, 378, 379, 380])
def test_provider_count_controls_unchanged_gate(counted, tokens):
    counted.seen.tokens = tokens
    if tokens <= 378:
        assert counted.dispatcher.dispatch(counted.key, lambda: counted.material,
                                           lambda response: response.content) == "final: 24"
    else:
        with pytest.raises((ValueError, RuntimeError), match="MAIN_INPUT_ENVELOPE_EXCEEDED"):
            counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    assert (counted.seen.counts, counted.seen.creates) == (1, int(tokens <= 378))
    compiled = counted.dispatcher.compiled_request(counted.key)
    assert counted.seen.count_body == {name: json.loads(compiled.request_bytes)[name]
                                      for name in ("model", "input", "reasoning", "tools")}


@pytest.mark.parametrize("field,value", [("object", "response"), ("tokens", True),
    ("tokens", -1), ("tokens", 378.0), ("tokens", "378"), ("tokens", float("inf")),
    ("tokens", float("nan")), ("cost", None)])
def test_invalid_count_never_generates_or_recounts(counted, field, value):
    setattr(counted.seen, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    with pytest.raises((ValueError, RuntimeError)):
        counted.dispatcher.recover()
    assert (counted.seen.counts, counted.seen.creates) == (1, 0)


@pytest.mark.parametrize("failure", [TimeoutError("fake count timeout"), CountCrash()])
def test_count_failure_or_ambiguous_crash_holds_after_reopen(counted, failure):
    counted.seen.failure = failure
    with pytest.raises((ValueError, RuntimeError, CountCrash)):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    reopened = TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)
    try:
        dispatcher = ProductionRequestDispatcherV3(reopened, counted.dispatcher.binding,
            counted.dispatcher.parents, provider_factory=lambda _: counted.provider)
        with pytest.raises((ValueError, RuntimeError), match="MAIN_COUNT"):
            dispatcher.recover()
        with pytest.raises((ValueError, RuntimeError)):
            dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
        assert (counted.seen.counts, counted.seen.creates) == (1, 0)
    finally:
        reopened.close()


def test_sdk_cannot_generate_without_count_receipt(counted):
    compiled = CompiledProviderRequestV3(counted.dispatcher.binding, counted.key, counted.material,
        compile_request_bytes(counted.key, counted.material), 1)
    with pytest.raises(ValueError, match="COUNT"):
        counted.provider.send_compiled_v3(compiled, lambda: pytest.fail("generation marker before count"))
    assert (counted.seen.counts, counted.seen.creates) == (0, 0)


@pytest.mark.parametrize("boundary", ["count-started", "count-receipt"])
def test_durable_count_boundary_crash_does_not_duplicate(counted, monkeypatch, boundary):
    publish = counted.dispatcher._publish_bytes

    def crash(key, role, raw):
        publish(key, role, raw)
        if role == boundary:
            raise CountCrash()

    monkeypatch.setattr(counted.dispatcher, "_publish_bytes", crash)
    with pytest.raises(CountCrash):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    with pytest.raises((ValueError, RuntimeError), match="MAIN_COUNT"):
        counted.dispatcher.recover()
    assert (counted.seen.counts, counted.seen.creates) == (int(boundary == "count-receipt"), 0)


def test_stale_receipt_cannot_authorize_changed_request(counted):
    counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    compiled = counted.dispatcher.compiled_request(counted.key)
    changed = counted.material.model_copy(update={"native_state": b"other native state"})
    with pytest.raises(ValueError, match="COUNT"):
        counted.provider.send_compiled_v3(replace(compiled, material=changed), lambda: None)
    assert (counted.seen.counts, counted.seen.creates) == (1, 1)


def test_count_cost_survives_ledger_reopen(counted):
    counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    reopened = TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)
    try:
        assert reopened.realized_cost_krw() >= 2
    finally:
        reopened.close()


def test_deleted_count_file_cannot_turn_ambiguous_send_into_no_request(counted):
    counted.seen.failure = CountCrash()
    with pytest.raises(CountCrash):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    (counted.ledger.path.parent / f"{counted.key.dispatch_id}.count-started.json").unlink()
    with pytest.raises(ValueError, match="MAIN_COUNT"):
        counted.dispatcher.recover()
    assert (counted.seen.counts, counted.seen.creates) == (1, 0)


def test_default_production_factory_stays_blocked_without_frozen_count_pricing(counted, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-study-credential")
    dispatcher = ProductionRequestDispatcherV3(counted.ledger, counted.dispatcher.binding,
                                               counted.dispatcher.parents)
    with pytest.raises(ValueError, match="MAIN_COUNT_PRICING_NOT_FROZEN"):
        dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    assert (counted.seen.counts, counted.seen.creates) == (0, 0)


def test_completed_receipt_cannot_be_replayed_at_sdk_boundary(counted):
    counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    with pytest.raises(ValueError, match="MAIN_COUNT"):
        counted.provider.send_compiled_v3(counted.dispatcher.compiled_request(counted.key), lambda: None)
    assert (counted.seen.counts, counted.seen.creates) == (1, 1)


@pytest.mark.parametrize("identity", ["base_url", "api_key"])
def test_changed_provider_identity_cannot_reuse_count(counted, identity):
    receipt = counted.provider.count_compiled_v3

    def changed(compiled, before_count):
        result = receipt(compiled, before_count)
        setattr(counted.provider.client, identity, "changed")
        return result

    counted.provider.count_compiled_v3 = changed
    with pytest.raises(ValueError, match="MAIN_COUNT"):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    assert (counted.seen.counts, counted.seen.creates) == (1, 0)


def test_generation_crash_reopens_without_recount_or_regeneration(counted):
    counted.seen.generation_failure = CountCrash()
    with pytest.raises(CountCrash):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    reopened = TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)
    try:
        dispatcher = ProductionRequestDispatcherV3(reopened, counted.dispatcher.binding,
            counted.dispatcher.parents, provider_factory=lambda _: counted.provider)
        dispatcher.recover()
        assert reopened.state(counted.key.dispatch_id).kind == "AMBIGUOUS_ATTEMPT"
        with pytest.raises((ValueError, RuntimeError)):
            dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
        assert (counted.seen.counts, counted.seen.creates) == (1, 1)
    finally:
        reopened.close()


def test_entitled_generation_retry_reuses_one_durable_count(counted):
    from openai import APIStatusError
    counted.seen.generation_failure = APIStatusError("fake acknowledged 429",
        response=httpx.Response(429, request=httpx.Request("POST", "https://denied.invalid/v1/responses")),
        body=None)
    dispatcher = ProductionRequestDispatcherV3(counted.ledger, counted.dispatcher.binding,
        counted.dispatcher.parents, provider_factory=lambda _: counted.provider,
        retry_entitlements=frozenset({counted.key.dispatch_id}))
    assert dispatcher.dispatch(counted.key, lambda: counted.material,
                               lambda response: response.content) == "final: 24"
    assert (counted.seen.counts, counted.seen.creates) == (1, 2)


def test_v3_chat_cannot_bypass_receipt_with_another_model(counted):
    with pytest.raises(ValueError, match="MAIN_COUNT_RECEIPT_REQUIRED"):
        counted.provider.chat([{"role": "user", "content": "fixture"}], "other-model", {})
    assert (counted.seen.counts, counted.seen.creates) == (0, 0)


def test_real_sdk_count_response_without_cost_keeps_unknown_cost_hold(counted, monkeypatch):
    from openai.types.responses.input_token_count_response import InputTokenCountResponse

    original = counted.provider.client.input_tokens.count

    def count(**body):
        original(**body)
        return InputTokenCountResponse(object="response.input_tokens", input_tokens=378)

    monkeypatch.setattr(counted.provider.client.input_tokens, "count", count)
    with pytest.raises(ValueError, match="MAIN_COUNT_FAILED_RECONCILIATION_REQUIRED"):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda response: response.content)
    raw = counted.ledger.count_record(counted.key.dispatch_id, "count-receipt")
    assert raw is not None and json.loads(raw)["monetary_cost_usd"] is None
    with pytest.raises(ValueError, match="MAIN_COUNT_COST_UNKNOWN"):
        counted.ledger.realized_cost_krw()
    assert (counted.seen.counts, counted.seen.creates) == (1, 0)
