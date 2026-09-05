from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3


MODULE = "memcontam.readiness.phase13_main_request_dispatch"


def test_production_request_dispatch_contract_is_available():
    assert importlib.util.find_spec(MODULE), "per-request V3 production dispatch is missing"


@pytest.fixture
def api():
    assert importlib.util.find_spec(MODULE), "per-request V3 production dispatch is missing"
    return importlib.import_module(MODULE)


@pytest.fixture
def rig(api, tmp_path: Path, monkeypatch):
    binding = api.PackageBindingV3(package_sha256="b" * 64, authorization_sha256="c" * 64)
    parents = (
        api.ParentTrajectoryV3(parent_id="a" * 64, kind="CLEAN_PREFIX"),
        api.ParentTrajectoryV3(parent_id="d" * 64, kind="MEMORY_BEARING", prefix_parent_id="a" * 64),
        api.ParentTrajectoryV3(parent_id="e" * 64, kind="NO_MEMORY_SINGLETON"),
    )
    keys = tuple(api.RequestKeyV3(parent_id=parent.parent_id, stage="rag_generate", ordinal=ordinal)
                 for parent in parents for ordinal in (0, 1))
    ledger = TerminalLedgerV3.create(tmp_path / "requests.sqlite3", {
        "schema_version": "phase13_main_run_ledger_v3", "unit_ids": [key.dispatch_id for key in keys],
        "package_sha256": binding.package_sha256, "authorization_sha256": binding.authorization_sha256,
    })
    seen = SimpleNamespace(constructors=0, requests=0, count=378, outcome="ok", trace=[])
    real_sync = TerminalLedgerV3._sync

    def sync(instance):
        real_sync(instance)
        seen.trace.append("fsync")

    monkeypatch.setattr(TerminalLedgerV3, "_sync", sync)

    class Provider:
        def send_compiled_v3(self, compiled, before_request):
            assert ledger.state(compiled.key.dispatch_id).kind == "REQUEST_COMPILED"
            before_request()
            assert seen.trace[-1] == "fsync"
            assert ledger.state(compiled.key.dispatch_id).kind == "ATTEMPT_STARTED"
            seen.requests += 1
            if seen.outcome == "transport":
                raise TimeoutError("synthetic transport failure")
            return LLMResponse("{}" if seen.outcome == "parse" else "final: 24", {
                "status": "incomplete" if seen.outcome in ("max_output_tokens", "unknown") else "completed",
                "incomplete_reason": seen.outcome,
                "usage": ({"input_tokens": 0, "output_tokens": 0} if seen.outcome == "ok"
                          else {"output_tokens": 0} if seen.outcome == "partial" else None),
            }, {}, 0)

    def factory(bound):
        assert bound == binding
        seen.constructors += 1
        seen.trace.append("constructor")
        return Provider()

    def count(messages, encoding):
        assert encoding == "o200k_base"
        seen.trace.append("count")
        return seen.count

    monkeypatch.setattr(api, "count_prompt_tokens", count)
    dispatcher = api.ProductionRequestDispatcherV3(ledger, binding, parents, provider_factory=factory)

    def material():
        assert any(ledger.state(key.dispatch_id).kind == "DISPATCH_INTENT_PERSISTED" for key in keys)
        return api.RequestMaterialV3(messages=({"role": "user", "content": "fixture input"},),
                                     native_state=b"immutable native state")

    def semantic(response):
        if response.content == "{}":
            raise ValueError("required semantic result unavailable")
        return response.content

    return SimpleNamespace(api=api, binding=binding, parents=parents, keys=keys, ledger=ledger,
                           dispatcher=dispatcher, seen=seen, material=material, semantic=semantic, root=tmp_path)


@pytest.mark.parametrize("count", [377, 378])
def test_accepted_rag_compiles_before_one_constructor_and_request(rig, count):
    rig.seen.count = count
    result = rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    assert result == "final: 24"
    assert (rig.seen.constructors, rig.seen.requests) == (1, 1)
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == "COMPLETED"
    assert rig.seen.trace.index("count") < rig.seen.trace.index("constructor")


def test_rag_379_terminalizes_after_intent_before_provider_construction(rig):
    rig.seen.count = 379
    with pytest.raises(rig.api.DispatchTechnicalFailureV3, match="MAIN_INPUT_ENVELOPE_EXCEEDED"):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    rows = [json.loads(raw) for raw in rig.ledger.rows()]
    assert [row["kind"] for row in rows] == ["DISPATCH_INTENT", "REQUEST_COMPILED",
                                             "INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING"]
    assert rows[1]["compiled"] == rows[2]["compiled"] == rows[3]["compiled"]
    assert rows[3]["transport_attempts"] == rows[3]["realized_cost_krw"] == 0
    assert (rig.seen.constructors, rig.seen.requests) == (0, 0)
    assert rig.dispatcher.terminal_parents == frozenset({"a" * 64, "d" * 64})
    saved = rig.dispatcher.compiled_request(rig.keys[0])
    assert saved.native_state == b"immutable native state"
    assert hashlib.sha256(saved.request_bytes).hexdigest() == rows[1]["compiled"]["compiled_request_hash"]
    assert rows[1]["compiled"]["token_count"] == 379
    durable = json.loads((rig.root / f"{rig.keys[0].dispatch_id}.compiled.json").read_bytes())
    assert bytes.fromhex(durable["request_hex"]) == saved.request_bytes
    assert bytes.fromhex(durable["native_state_hex"]) == saved.native_state
    assert hashlib.sha256(bytes.fromhex(durable["input_hex"])).hexdigest() == rows[1]["compiled"]["immutable_input_hash"]


def test_nonprefix_overflow_does_not_fan_out(rig):
    rig.seen.count = 379
    with pytest.raises(rig.api.DispatchTechnicalFailureV3, match="MAIN_INPUT_ENVELOPE_EXCEEDED"):
        rig.dispatcher.dispatch(rig.keys[2], rig.material, rig.semantic)
    assert rig.dispatcher.terminal_parents == frozenset({"d" * 64})
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == "PENDING"
    assert (rig.seen.constructors, rig.seen.requests) == (0, 0)


@pytest.mark.parametrize("outcome", ["transport", "max_output_tokens", "parse", "unknown"])
@pytest.mark.parametrize("index", [0, 2])
def test_attempt_failure_is_terminal_nullable_and_blocks_later_parent_calls(rig, outcome, index):
    rig.seen.outcome = outcome
    with pytest.raises(rig.api.DispatchTechnicalFailureV3) as failure:
        rig.dispatcher.dispatch(rig.keys[index], rig.material, rig.semantic)
    assert failure.value.realized_cost_krw is None
    assert rig.ledger.state(rig.keys[index].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    assert rig.dispatcher.terminal_parents == (frozenset({"a" * 64, "d" * 64}) if index == 0 else frozenset({"d" * 64}))
    blocked = (rig.keys[index], rig.keys[index + 1]) + ((rig.keys[2],) if index == 0 else ())
    for key in blocked:
        with pytest.raises(rig.api.DispatchTechnicalFailureV3):
            rig.dispatcher.dispatch(key, rig.material, rig.semantic)
    assert (rig.seen.constructors, rig.seen.requests) == (1, 1)
    assert json.loads(rig.ledger.rows()[-1])["realized_cost_krw"] is None


def test_two_sequential_requests_have_distinct_single_attempt_rows(rig):
    rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    rig.dispatcher.dispatch(rig.keys[1], rig.material, rig.semantic)
    for key in rig.keys[:2]:
        assert [json.loads(raw)["kind"] for raw in rig.ledger.rows()
                if json.loads(raw)["unit_id"] == key.dispatch_id] == [
                    "DISPATCH_INTENT", "REQUEST_COMPILED", "ATTEMPT_STARTED", "COMPLETED"]
    assert rig.seen.requests == 2


def test_unknown_attempted_cost_blocks_independent_request(rig):
    rig.seen.outcome = "transport"
    with pytest.raises(rig.api.DispatchTechnicalFailureV3):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    with pytest.raises(ValueError, match="MAIN_TERMINAL_COST_UNKNOWN"):
        rig.dispatcher.dispatch(rig.keys[4], rig.material, rig.semantic)


@pytest.mark.parametrize("changed", ["parent", "stage", "ordinal"])
def test_dispatch_identity_is_stable_and_separates_semantic_roles(api, changed):
    values = {"parent_id": "a" * 64, "stage": "rag_generate", "ordinal": 0}
    original = api.RequestKeyV3(**values)
    values[{"parent": "parent_id", "stage": "stage", "ordinal": "ordinal"}[changed]] = {
        "parent": "b" * 64, "stage": "no_memory_generate", "ordinal": 1,
    }[changed]
    assert original.dispatch_id == api.RequestKeyV3(parent_id="a" * 64, stage="rag_generate", ordinal=0).dispatch_id
    assert original.dispatch_id != api.RequestKeyV3(**values).dispatch_id


@pytest.mark.parametrize("field,value", [("run_id", "phase13-main-a-corrected-20260905-v2"),
                                        ("package_id", "contains-v3"), ("authorization_id", "v3")])
def test_loose_or_mixed_v3_identity_is_rejected(api, field, value):
    with pytest.raises(ValueError):
        api.PackageBindingV3(identity={field: value}, package_sha256="b" * 64, authorization_sha256="c" * 64)


def test_ledger_binding_mismatch_prevents_provider_construction(rig):
    wrong = rig.binding.model_copy(update={"package_sha256": "f" * 64})
    with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        rig.api.ProductionRequestDispatcherV3(rig.ledger, wrong, rig.parents)
    assert (rig.seen.constructors, rig.seen.requests) == (0, 0)


@pytest.mark.parametrize("stage,cap", [
    ("full_history_generate", 512), ("rag_generate", 512), ("bot_problem_distill", 384),
    ("bot_instantiate_solve", 512), ("bot_thought_distill", 384), ("reflexion_generate", 512),
    ("reflexion_reflect", 384), ("dc_rs_generate", 512), ("dc_rs_synthesize", 8192),
    ("no_memory_generate", 512),
])
@pytest.mark.parametrize("outcome", ["ok", "transport", "max_output_tokens", "unknown"])
def test_bound_openai_v3_sends_exact_compiled_bytes_once(api, monkeypatch, stage, cap, outcome):
    from memcontam.clients import openai_responses
    from memcontam.readiness import phase13_v3_request as request_api

    seen = []

    class SDK:
        def __init__(self, **options):
            seen.append(("constructor", options))
            self.responses = self

        def create(self, **request):
            assert seen[-1] == "marker"
            seen.append(request)
            if outcome == "transport":
                raise TimeoutError("synthetic timeout")
            return SimpleNamespace(output_text="final: 24", usage=None,
                status="completed" if outcome == "ok" else "incomplete",
                incomplete_details=SimpleNamespace(reason=outcome))

    monkeypatch.setattr(openai_responses, "OpenAI", SDK)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-a-credential")
    binding = api.PackageBindingV3(package_sha256="b" * 64, authorization_sha256="c" * 64)
    key = api.RequestKeyV3(parent_id="a" * 64, stage=stage, ordinal=0)
    material = api.RequestMaterialV3(messages=({"role": "user", "content": "input"},), native_state=b"state")
    compiled = request_api.CompiledProviderRequestV3(binding, key, material,
        request_api.compile_request_bytes(key, material), 1)
    provider = api.production_provider(binding)
    if outcome == "ok":
        response = provider.send_compiled_v3(compiled, lambda: seen.append("marker"))
        assert response.raw["attempts"] == 1
        assert response.raw["cost_usd"] is None
    else:
        with pytest.raises((TimeoutError, openai_responses.LunaContractError)) as failure:
            provider.send_compiled_v3(compiled, lambda: seen.append("marker"))
        assert getattr(failure.value, "provider_attempts_count") == 1
    assert seen[0][1]["max_retries"] == 0
    assert seen[-1] == json.loads(compiled.request_bytes)
    assert seen[-1]["max_output_tokens"] == cap
    assert len(seen) == 3


def test_production_runtime_defers_provider_construction(monkeypatch):
    from memcontam.readiness import phase13_main_live_runtime as runtime

    def forbidden(*args, **kwargs):
        pytest.fail("provider constructed while assembling production runtime")

    monkeypatch.setattr(runtime, "OpenAIResponsesClient", forbidden)
    runtime.ProductionMainRuntime(Path(__file__).resolve().parents[1], Path("unused-offline-cache"))


@pytest.mark.parametrize("boundary", ["semantic", "marker"])
def test_marker_interruption_and_missing_semantic_result_have_distinct_evidence(rig, monkeypatch, boundary):
    if boundary == "marker":
        real_sync = TerminalLedgerV3._sync
        def interrupted(instance):
            real_sync(instance)
            if instance.state(rig.keys[0].dispatch_id).kind == "ATTEMPT_STARTED":
                raise OSError("marker fsync interrupted")
        monkeypatch.setattr(TerminalLedgerV3, "_sync", interrupted)
    with pytest.raises(rig.api.DispatchTechnicalFailureV3 if boundary == "semantic" else OSError):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, lambda response: None)
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == (
        "ATTEMPTED_PROVIDER_FAILURE" if boundary == "semantic" else "ATTEMPT_STARTED")
    assert rig.seen.requests == int(boundary == "semantic")


def test_partial_provider_usage_terminalizes_with_unknown_cost(rig):
    rig.seen.outcome = "partial"
    with pytest.raises(rig.api.DispatchTechnicalFailureV3) as failure:
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    assert failure.value.realized_cost_krw is None
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"


def test_unbound_v3_registry_string_cannot_enable_new_limits(monkeypatch):
    from memcontam.clients.config import ProviderConfig
    from memcontam.clients.openai_responses import OpenAIResponsesClient, LunaContractError

    client = OpenAIResponsesClient.__new__(OpenAIResponsesClient)
    client._config = ProviderConfig(provider="openai_responses", live_calls_enabled=True,
                                    timeout_seconds=180, retries_after_initial_attempt=2)
    client._allow_live_calls, client._v3_binding = True, None
    from memcontam.clients.cost_guard import CostGuard
    client.cost_guard = CostGuard()
    with pytest.raises(LunaContractError, match="LUNA_OUTPUT_CONTRACT_MISMATCH"):
        client.chat([{"role": "user", "content": "input"}], "gpt-5.6-luna", {
            "max_output_tokens": 512, "_phase13_execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V3",
        })


@pytest.mark.parametrize("tokens", [377, 378, 379])
def test_real_pinned_tokenizer_drives_the_rag_gate(rig, monkeypatch, tokens):
    from memcontam.baselines.prompt_budget import count_prompt_tokens

    monkeypatch.setattr(rig.api, "count_prompt_tokens", count_prompt_tokens)
    material = rig.api.RequestMaterialV3(
        messages=({"role": "user", "content": "x " * (tokens - 3)},), native_state=b"state",
    )
    if tokens == 379:
        with pytest.raises(rig.api.DispatchTechnicalFailureV3, match="MAIN_INPUT_ENVELOPE_EXCEEDED"):
            rig.dispatcher.dispatch(rig.keys[0], lambda: material, rig.semantic)
    else:
        rig.dispatcher.dispatch(rig.keys[0], lambda: material, rig.semantic)
    assert rig.dispatcher.compiled_request(rig.keys[0]).token_count == tokens
    assert rig.seen.constructors == rig.seen.requests == int(tokens <= 378)
