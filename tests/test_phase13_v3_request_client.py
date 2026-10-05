from __future__ import annotations

import threading
from typing import Never

import pytest
from pydantic import JsonValue, TypeAdapter

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.runtime_registry import NOMEM_SINGLETON, RuntimeTrialResult
from memcontam.readiness.phase13_main_request_client import MainRequestClientV3, TerminalTrialV3
from memcontam.readiness.phase13_main_request_dispatch import (
    DispatchTechnicalFailureV3,
    ProductionRequestDispatcherV3,
)
from memcontam.readiness.phase13_v3_entrypoint import EntrypointError
from memcontam.readiness.phase13_v3_entrypoint_paths import private_ledger
from memcontam.readiness.phase13_v3_request import (
    PackageBindingV3,
    ParentTrajectoryV3,
    RequestKeyV3,
    Stage,
)
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
from memcontam.readiness.phase13_v3_terminal_models import TerminalEvidenceError

from .phase13_corrective_identity import corrective_identity
from .phase13_count_fake import CountedProvider
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external
from .test_phase13_v3_entrypoint_fixture import entrypoint_bytes as entrypoint_bytes
from .test_phase13_runner_safety import entrypoint_fixture as entrypoint_fixture
from .test_phase13_runner_safety import local_authority as local_authority
from .test_phase13_runner_safety import source_selection as source_selection

pytestmark = pytest.mark.usefixtures("deny_external")

ClientFixture = tuple[MainRequestClientV3, TerminalLedgerV3, tuple[RequestKeyV3, ...], dict[str, int]]


@pytest.fixture
def client_fixture(tmp_path, monkeypatch, request):
    import memcontam.readiness.phase13_main_request_dispatch as dispatch

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: 1)
    binding = PackageBindingV3(identity=corrective_identity(), package_sha256="b" * 64, authorization_sha256="c" * 64)
    stage = TypeAdapter(Stage).validate_python(getattr(request, "param", "no_memory_generate"))
    keys = tuple(RequestKeyV3(parent_id="a" * 64, stage=stage, ordinal=index) for index in range(2))
    counts = {"constructor": 0, "requests": 0}

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            assert compiled.native_state == b"immutable native bytes"
            before_request()
            counts["requests"] += 1
            return LLMResponse("not a final answer", {"status": "completed", "model": "gpt-5.6-luna",
                "service_tier": "default", "usage": {"input_tokens": 1, "output_tokens": 1}}, {}, 0)

    def factory(_binding):
        counts["constructor"] += 1
        return Provider()

    with private_ledger(tmp_path / "fixture", create=True) as private:
        ledger = TerminalLedgerV3.create_guarded(private, {"schema_version": "phase13_main_run_ledger_v3",
            "identity": binding.identity.model_dump(mode="json"),
            "unit_ids": [key.dispatch_id for key in keys], "package_sha256": binding.package_sha256,
            "authorization_sha256": binding.authorization_sha256})
        dispatcher = ProductionRequestDispatcherV3(ledger, binding,
            (ParentTrajectoryV3(parent_id="a" * 64, kind="NO_MEMORY_SINGLETON"),), provider_factory=factory)
        client = MainRequestClientV3(dispatcher, "a" * 64, private.check)
        yield client, ledger, keys, counts


def call(client):
    return client.chat([{"role": "user", "content": "fixture"}], "gpt-5.6-luna", {"method_stage": "no_memory_generate"})


class RetryableTimeout(TimeoutError):
    phase13_retry_class = "TIMEOUT_BEFORE_SEMANTIC_PAYLOAD"
    provider_failure_acknowledged = True


class SimulatedCrash(BaseException):
    pass


def test_frozen_entitlement_retries_one_unambiguous_transport_failure(client_fixture):
    client, ledger, keys, counts = client_fixture

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            counts["requests"] += 1
            if counts["requests"] == 1:
                raise RetryableTimeout()
            return LLMResponse("final: 24", {"status": "completed", "model": "gpt-5.6-luna",
                "service_tier": "default", "usage": {"input_tokens": 1, "output_tokens": 1}}, {}, 0)

    def factory(_binding):
        counts["constructor"] += 1
        return Provider()

    client.dispatcher = ProductionRequestDispatcherV3(
        ledger, client.dispatcher.binding, client.dispatcher.parents,
        provider_factory=factory, retry_entitlements=frozenset({keys[0].dispatch_id}),
    )

    def execute():
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    client.trial(execute, lambda: b"immutable native bytes")

    events = [row.decode() for row in ledger.rows()]
    assert sum('"kind":"ATTEMPT_STARTED"' in row for row in events) == 2
    assert sum('"kind":"RETRYABLE_ATTEMPT_FAILURE"' in row for row in events) == 1
    assert ledger.state(keys[0].dispatch_id).kind == "COMPLETED"
    assert counts == {"constructor": 2, "requests": 2}


def test_successful_retry_keeps_unknown_first_attempt_cost_until_reconciled(client_fixture):
    from memcontam.readiness.phase13_v3_cost_models import CostError

    client, ledger, keys, counts = client_fixture

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            counts["requests"] += 1
            if counts["requests"] == 1:
                raise RetryableTimeout()
            return LLMResponse("final: 24", {"status": "completed", "model": "gpt-5.6-luna",
                "service_tier": "default", "usage": {"input_tokens": 1, "output_tokens": 1}}, {}, 0)

    client.dispatcher = ProductionRequestDispatcherV3(
        ledger, client.dispatcher.binding, client.dispatcher.parents,
        provider_factory=lambda _binding: Provider(),
        retry_entitlements=frozenset({keys[0].dispatch_id}),
    )

    client.trial(lambda: (call(client), RuntimeTrialResult(
        BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON))[1],
        lambda: b"immutable native bytes")
    state = ledger.state(keys[0].dispatch_id)
    assert state.kind == "COMPLETED"
    assert len(state.attempt_costs) == 2
    with pytest.raises(TerminalEvidenceError, match="MAIN_TERMINAL_COST_UNKNOWN"):
        client.realized_cost_krw()
    with pytest.raises(CostError, match="MAIN_TERMINAL_COST_UNKNOWN"):
        ledger.realized_cost_krw()
    assert ledger.guard is not None
    reopened = TerminalLedgerV3.open_guarded(ledger.guard, ledger.binding)
    try:
        assert len(reopened.state(keys[0].dispatch_id).attempt_costs) == 2
        with pytest.raises(CostError, match="MAIN_TERMINAL_COST_UNKNOWN"):
            reopened.realized_cost_krw()
        reopened.reconcile_cost(keys[0].dispatch_id, {"usage": {"input_tokens": 2, "output_tokens": 1}},
                                "f" * 64, attempt_index=0)
        assert reopened.realized_cost_krw() > 0
        assert client.realized_cost_krw() == reopened.realized_cost_krw()
    finally:
        reopened.close()


def test_terminal_parent_observed_total_includes_count_after_reopen(entrypoint_fixture, monkeypatch):
    import json
    from pathlib import Path

    import memcontam.readiness.phase13_main_request_dispatch as dispatch
    from .phase13_runner_safety_fixture import FakeProvider, open_run

    class Provider(FakeProvider):
        def send_compiled_v3(self, compiled, before_request):
            if compiled.key.ordinal == 1:
                before_request()
                self.requests.append(compiled.key.dispatch_id)
                raise RuntimeError("synthetic generation failure")
            return super().send_compiled_v3(compiled, before_request)

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: 1)
    provider = Provider()
    run = open_run(entrypoint_fixture, create=True)
    parent_id = run.selected.package.production[0].unit_id
    try:
        assert run.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000,
                           provider_factory=provider.factory).terminal_technical_missing_count == 1
        raw = run.ledger.read_record(f"{parent_id}.parent.json")
        parent = json.loads(raw)
        assert parent["observation_cost_krw"] == 3
        assert parent["whole_unit_cost_krw"] is None
    finally:
        run.close()
    reopened = open_run(entrypoint_fixture, create=False)
    try:
        assert reopened.status().terminal_technical_missing_count == 1
        assert reopened.ledger.read_record(f"{parent_id}.parent.json") == raw
        assert len(provider.requests) == 2
        assert len(reopened.ledger.count_records("count-started")) == 2
        key = RequestKeyV3(parent_id=parent_id, stage="no_memory_generate", ordinal=1)
        reopened.ledger.reconcile_cost(key.dispatch_id,
            {"usage": {"input_tokens": 1, "output_tokens": 0}}, "f" * 64)
        assert reopened.ledger.realized_cost_krw() == 6
        assert reopened.status().terminal_technical_missing_count == 1
    finally:
        reopened.close()


def test_retryable_event_cannot_hide_observed_first_attempt_cost() -> None:
    from memcontam.readiness.phase13_v3_terminal_models import RetryableAttemptFailureV3

    with pytest.raises(ValueError, match="MAIN_RETRY_ENTITLEMENT_INVALID"):
        RetryableAttemptFailureV3.model_validate({
            "schema_version": "phase13_main_dispatch_evidence_v3", "kind": "RETRYABLE_ATTEMPT_FAILURE",
            "unit_id": "a" * 64, "revision": 3, "previous_hash": "b" * 64,
            "compiled": {"stage": "no_memory_generate", "token_count": 1,
                         "compiled_request_hash": "c" * 64, "immutable_input_hash": "d" * 64,
                         "native_state_hash": "e" * 64},
            "attempt_index": 0, "failure_code": "TIMEOUT_BEFORE_SEMANTIC_PAYLOAD",
            "observation_hash": "f" * 64,
            "cost": {"usage": {"input_tokens": 1, "output_tokens": 1}},
            "realized_cost_krw": None,
        })


def test_transport_failure_without_frozen_entitlement_is_not_retried(client_fixture):
    client, ledger, keys, counts = client_fixture

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            counts["requests"] += 1
            raise RetryableTimeout()

    client.dispatcher._factory = lambda _binding: Provider()

    terminal = client.trial(
        lambda: (call(client), RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON))[1],
        lambda: b"immutable native bytes",
    )

    assert isinstance(terminal, TerminalTrialV3)
    assert terminal.result is None
    assert terminal.failure.code == "MAIN_ATTEMPTED_PROVIDER_FAILURE"
    assert ledger.state(keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    assert counts["requests"] == 1


def test_provider_failure_preserves_actual_failed_baseline_result(client_fixture):
    client, ledger, keys, counts = client_fixture

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            counts["requests"] += 1
            raise RuntimeError("provider unavailable")

    client.dispatcher._factory = lambda _binding: Provider()

    def execute():
        try:
            call(client)
        except DispatchTechnicalFailureV3:
            return RuntimeTrialResult(BaselineExecutionOutcome("failed", error_type="ProviderCallFailure",
                failure_disposition="provider_call_failed", scientific_ineligibility_reason="provider_call_failed"),
                NOMEM_SINGLETON)
        pytest.fail("provider did not fail")

    terminal = client.trial(execute, lambda: b"immutable native bytes")

    assert isinstance(terminal, TerminalTrialV3)
    assert terminal.result is not None
    assert terminal.result.outcome.failure_disposition == "provider_call_failed"
    assert terminal.failure.code == "MAIN_ATTEMPTED_PROVIDER_FAILURE"
    assert ledger.state(keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    assert counts["requests"] == 1


def test_restart_after_durable_retryable_failure_issues_only_second_attempt(
    client_fixture, monkeypatch: pytest.MonkeyPatch,
):
    client, ledger, keys, counts = client_fixture

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            counts["requests"] += 1
            if counts["requests"] == 1:
                raise RetryableTimeout()
            return LLMResponse("final: 24", {"status": "completed", "model": "gpt-5.6-luna",
                "service_tier": "default", "usage": {"input_tokens": 1, "output_tokens": 1}}, {}, 0)

    def factory(_binding):
        counts["constructor"] += 1
        return Provider()

    dispatcher = ProductionRequestDispatcherV3(
        ledger, client.dispatcher.binding, client.dispatcher.parents,
        provider_factory=factory, retry_entitlements=frozenset({keys[0].dispatch_id}),
    )
    client.dispatcher = dispatcher
    append = dispatcher._append

    def crash_after_retryable(key, kind, extra=None):
        append(key, kind, extra)
        if kind == "RETRYABLE_ATTEMPT_FAILURE":
            raise SimulatedCrash()

    monkeypatch.setattr(dispatcher, "_append", crash_after_retryable)
    with pytest.raises(SimulatedCrash):
        client.trial(
            lambda: (call(client), RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON))[1],
            lambda: b"immutable native bytes",
        )
    assert ledger.state(keys[0].dispatch_id).kind == "RETRYABLE_ATTEMPT_FAILURE"

    ledger.reconcile_cost(keys[0].dispatch_id,
        {"usage": {"input_tokens": 2, "output_tokens": 1}}, "f" * 64, attempt_index=0)
    assert ledger.state(keys[0].dispatch_id).kind == "RETRYABLE_ATTEMPT_FAILURE"
    ledger.require_known_costs()

    resumed = ProductionRequestDispatcherV3(
        ledger, dispatcher.binding, dispatcher.parents,
        provider_factory=factory, retry_entitlements=frozenset({keys[0].dispatch_id}),
    )
    resumed.recover()
    client.dispatcher = resumed
    client.trial(
        lambda: (call(client), RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON))[1],
        lambda: b"immutable native bytes",
    )

    assert ledger.state(keys[0].dispatch_id).kind == "COMPLETED"
    assert counts == {"constructor": 2, "requests": 2}


@pytest.mark.parametrize("client_fixture", (
    "no_memory_generate", "full_history_generate", "rag_generate", "bot_problem_distill",
    "bot_instantiate_solve", "bot_thought_distill", "reflexion_generate",
    "reflexion_reflect", "dc_rs_generate", "dc_rs_synthesize",
), indirect=True)
def test_entitled_second_failure_exhausts_once_after_first_attempt_crash_and_reopen(
    client_fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, ledger, keys, counts = client_fixture
    assert keys[0].stage == keys[1].stage
    attempted_indices: list[int] = []

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            counts["requests"] += 1
            attempted_indices.append(counts["requests"] - 1)
            raise RetryableTimeout()

    def factory(_binding):
        return Provider()
    dispatcher = ProductionRequestDispatcherV3(ledger, client.dispatcher.binding,
        client.dispatcher.parents, provider_factory=factory,
        retry_entitlements=frozenset({keys[0].dispatch_id}))
    client.dispatcher = dispatcher
    original_append = dispatcher._append

    def crash_after_first(key, kind, extra=None):
        original_append(key, kind, extra)
        if kind == "RETRYABLE_ATTEMPT_FAILURE":
            raise SimulatedCrash()

    monkeypatch.setattr(dispatcher, "_append", crash_after_first)
    messages = [{"role": "user", "content": "fixture"}]

    def invoke():
        return client.trial(lambda: (client.chat(messages, "gpt-5.6-luna",
            {"method_stage": keys[0].stage}), RuntimeTrialResult(
                BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON))[1],
            lambda: b"immutable native bytes")

    with pytest.raises(SimulatedCrash):
        invoke()
    assert attempted_indices == [0]
    assert ledger.guard is not None
    reopened = TerminalLedgerV3.open_guarded(ledger.guard, ledger.binding)
    try:
        reopened.reconcile_cost(keys[0].dispatch_id,
            {"usage": {"input_tokens": 1, "output_tokens": 1}}, "f" * 64, attempt_index=0)
        resumed = ProductionRequestDispatcherV3(reopened, dispatcher.binding, dispatcher.parents,
            provider_factory=factory, retry_entitlements=frozenset({keys[0].dispatch_id}))
        resumed.recover()
        client.dispatcher = resumed
        terminal = invoke()
        assert isinstance(terminal, TerminalTrialV3)
        assert terminal.failure.code == "MAIN_ATTEMPTED_PROVIDER_FAILURE"
        assert reopened.state(keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
        assert len(reopened.state(keys[0].dispatch_id).attempt_costs) == 2
        assert attempted_indices == [0, 1]
        assert sum(b'"kind":"ATTEMPT_STARTED"' in row for row in reopened.rows()) == 2
        assert isinstance(invoke(), TerminalTrialV3)
        assert attempted_indices == [0, 1]
    finally:
        reopened.close()


def test_trial_ordinal_base_is_schedule_stable_after_skipped_occurrence(client_fixture):
    client, ledger, keys, _counts = client_fixture

    client.trial(
        lambda: RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON),
        lambda: b"immutable native bytes", ordinal_base=0,
    )

    def execute():
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    client.trial(execute, lambda: b"immutable native bytes", ordinal_base=1)

    assert ledger.state(keys[0].dispatch_id).kind == "PENDING"
    assert ledger.state(keys[1].dispatch_id).kind == "COMPLETED"


def test_semantic_failure_from_actual_baseline_outcome_is_terminal(client_fixture):
    client, ledger, keys, counts = client_fixture

    def execute():
        call(client)
        assert ledger.state(keys[0].dispatch_id).kind == "ATTEMPT_STARTED"
        return RuntimeTrialResult(BaselineExecutionOutcome("failed", error_type="BaselineOutputError",
            failure_disposition="no_memory_invalid_final_answer", scientific_ineligibility_reason="invalid_final_answer"), NOMEM_SINGLETON)

    terminal = client.trial(execute, lambda: b"immutable native bytes")
    assert isinstance(terminal, TerminalTrialV3)
    assert terminal.result is not None
    assert terminal.result.outcome.failure_disposition == "no_memory_invalid_final_answer"
    assert terminal.failure.code == "MAIN_ATTEMPTED_PROVIDER_FAILURE"
    assert ledger.state(keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    assert counts == {"constructor": 1, "requests": 1}


def test_next_call_acknowledges_previous_semantic_success(client_fixture):
    client, ledger, keys, counts = client_fixture

    def execute():
        call(client)
        call(client)
        assert ledger.state(keys[0].dispatch_id).kind == "COMPLETED"
        assert ledger.state(keys[1].dispatch_id).kind == "ATTEMPT_STARTED"
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    client.trial(execute, lambda: b"immutable native bytes")
    assert ledger.state(keys[1].dispatch_id).kind == "COMPLETED"
    assert counts == {"constructor": 2, "requests": 2}


def test_invalid_receipt_stops_native_path_before_next_request(client_fixture):
    client, ledger, keys, counts = client_fixture

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            counts["requests"] += 1
            return LLMResponse("final: 24", {"status": "completed", "model": "wrong-model",
                "service_tier": "default", "usage": {"input_tokens": 1, "output_tokens": 1}}, {}, 0)

    client.dispatcher._factory = lambda _binding: Provider()

    def execute():
        call(client)
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    terminal = client.trial(execute, lambda: b"immutable native bytes")
    assert isinstance(terminal, TerminalTrialV3)
    assert ledger.state(keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    assert ledger.state(keys[1].dispatch_id).kind == "PENDING"
    assert counts["requests"] == 1


def test_crash_before_semantic_ack_is_ambiguous_and_never_redispatched(client_fixture):
    client, ledger, keys, counts = client_fixture

    def crash():
        call(client)
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        client.trial(crash, lambda: b"immutable native bytes")
    client.dispatcher.recover()
    assert ledger.state(keys[0].dispatch_id).kind == "AMBIGUOUS_ATTEMPT"
    assert "a" * 64 in client.dispatcher.terminal_parents
    assert counts == {"constructor": 1, "requests": 1}


def test_guarded_request_lock_serializes_threads(client_fixture):
    from memcontam.readiness.phase13_main_request_recovery import request_lock

    _client, ledger, _keys, _counts = client_fixture
    started = threading.Event()
    entered = threading.Event()

    def worker():
        started.set()
        with request_lock(ledger):
            entered.set()

    with request_lock(ledger):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        assert started.wait(2)
        assert not entered.wait(0.2), "same ledger lock admitted a concurrent request"
    thread.join(2)
    assert entered.is_set()
    assert not thread.is_alive()


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, OSError, TypeError, KeyError])
@pytest.mark.parametrize("recovered", [False, True])
def test_uncoded_native_failure_maps_after_durable_intent(
    client_fixture: ClientFixture, error_type: type[Exception], recovered: bool,
) -> None:
    client, ledger, keys, counts = client_fixture
    failure = error_type("native serialization failed")
    intent_rows: tuple[bytes, ...] = ()

    def fail() -> Never:
        nonlocal intent_rows
        assert ledger.state(keys[0].dispatch_id).kind == "DISPATCH_INTENT_PERSISTED"
        intent_rows = ledger.rows()
        raise failure

    def execute() -> RuntimeTrialResult:
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    if recovered:
        with pytest.raises(error_type):
            client.dispatcher.receive(keys[0], fail)
        client.dispatcher.recover()
        assert ledger.state(keys[0].dispatch_id).kind == "PENDING"

    with pytest.raises(TerminalEvidenceError) as raised:
        client.trial(execute, fail)

    assert raised.value.code == "MAIN_RUN_POST_INTENT_RUNTIME_FAILURE"
    assert raised.value.__cause__ is failure
    assert ledger.rows() == intent_rows
    assert ledger.state(keys[0].dispatch_id).kind == "DISPATCH_INTENT_PERSISTED"
    assert counts == {"constructor": 0, "requests": 0}


@pytest.mark.parametrize("failure", [
    EntrypointError("MAIN_PATH_UNSAFE"),
    TerminalEvidenceError("MAIN_NATIVE_STATE_UNAVAILABLE"),
    DispatchTechnicalFailureV3("MAIN_ATTEMPTED_PROVIDER_FAILURE", "a" * 64, None),
    KeyboardInterrupt(), SystemExit(17),
])
def test_coded_and_process_control_failures_preserve_identity_after_intent(
    client_fixture: ClientFixture, failure: BaseException,
) -> None:
    client, ledger, keys, counts = client_fixture
    intent_rows: tuple[bytes, ...] = ()

    def fail() -> Never:
        nonlocal intent_rows
        intent_rows = ledger.rows()
        raise failure

    def execute() -> RuntimeTrialResult:
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    with pytest.raises(type(failure)) as raised:
        client.trial(execute, fail)

    assert raised.value is failure
    assert ledger.rows() == intent_rows
    assert ledger.state(keys[0].dispatch_id).kind == "DISPATCH_INTENT_PERSISTED"
    assert counts == {"constructor": 0, "requests": 0}


@pytest.mark.parametrize("failure", [ValueError("preflight"), EntrypointError("MAIN_PATH_UNSAFE")])
def test_preflight_failure_preserves_identity_without_intent(
    client_fixture: ClientFixture, failure: ValueError,
) -> None:
    client, ledger, _keys, counts = client_fixture

    def fail() -> Never:
        raise failure

    client.preflight = fail

    def execute() -> RuntimeTrialResult:
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    with pytest.raises(type(failure)) as raised:
        client.trial(execute, lambda: b"immutable native bytes")

    assert raised.value is failure
    assert ledger.rows() == ()
    assert counts == {"constructor": 0, "requests": 0}


def test_identity_publication_failure_is_not_post_intent(
    client_fixture: ClientFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, ledger, _keys, counts = client_fixture
    failure = OSError("identity publication failed")

    def fail(_key: RequestKeyV3, _role: str, _raw: bytes) -> Never:
        raise failure

    monkeypatch.setattr(client.dispatcher, "_publish_bytes", fail)

    def execute() -> RuntimeTrialResult:
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    with pytest.raises(OSError) as raised:
        client.trial(execute, lambda: b"immutable native bytes")

    assert raised.value is failure
    assert ledger.rows() == ()
    assert counts == {"constructor": 0, "requests": 0}


@pytest.mark.parametrize("point", ["REQUEST_COMPILED", "ATTEMPT_STARTED"])
def test_uncoded_marker_failure_preserves_in_flight_evidence(
    client_fixture: ClientFixture, monkeypatch: pytest.MonkeyPatch, point: str,
) -> None:
    client, ledger, keys, counts = client_fixture
    append = client.dispatcher._append
    failure = OSError("marker acknowledgement failed")
    persisted_rows: tuple[bytes, ...] = ()

    def fail(key: RequestKeyV3, kind: str, extra: dict[str, JsonValue] | None = None) -> None:
        nonlocal persisted_rows
        append(key, kind, extra)
        if kind == point:
            persisted_rows = ledger.rows()
            raise failure

    monkeypatch.setattr(client.dispatcher, "_append", fail)

    def execute() -> RuntimeTrialResult:
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    with pytest.raises(TerminalEvidenceError) as raised:
        client.trial(execute, lambda: b"immutable native bytes")

    assert raised.value.code == "MAIN_RUN_POST_INTENT_RUNTIME_FAILURE"
    assert raised.value.__cause__ is failure
    assert ledger.rows() == persisted_rows
    assert ledger.state(keys[0].dispatch_id).kind == point
    assert counts == {"constructor": int(point == "ATTEMPT_STARTED"), "requests": 0}


def test_parent_client_total_includes_each_count_once(client_fixture):
    client, ledger, _keys, _counts = client_fixture

    def execute():
        call(client)
        call(client)
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

    client.trial(execute, lambda: b"immutable native bytes")
    assert client.realized_cost_krw() == ledger.realized_cost_krw() == 6


def test_parent_cost_rejects_receipt_for_another_request(client_fixture):
    import json

    client, ledger, keys, _counts = client_fixture
    client.trial(lambda: (call(client), RuntimeTrialResult(
        BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON))[1],
        lambda: b"immutable native bytes")
    raw = ledger.count_record(keys[0].dispatch_id, "count-receipt")
    assert raw is not None
    receipt = json.loads(raw)
    receipt["operation"]["key"]["ordinal"] = 1
    with ledger.connection() as connection:
        connection.execute("UPDATE provider_counts_v1 SET raw=? WHERE unit_id=? AND role='count-receipt'",
                           (json.dumps(receipt).encode(), keys[0].dispatch_id))
    with pytest.raises(TerminalEvidenceError, match="MAIN_COUNT_REQUEST_BINDING_MISMATCH"):
        client.realized_cost_krw()


def test_count_failure_persists_only_registered_code(client_fixture):
    import json

    client, ledger, keys, counts = client_fixture

    class UntrustedFailure(TimeoutError):
        code = "secret-key-and-raw-prompt"

    class Provider(CountedProvider):
        def count_compiled_v3(self, compiled, before_count):
            before_count()
            raise UntrustedFailure("secret-key-and-raw-prompt")

    client.dispatcher._factory = lambda _: Provider()
    with pytest.raises(TerminalEvidenceError, match="MAIN_COUNT"):
        client.trial(lambda: (call(client), RuntimeTrialResult(
            BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON))[1],
            lambda: b"immutable native bytes")
    raw = ledger.count_record(keys[0].dispatch_id, "count-failure")
    assert raw is not None
    assert json.loads(raw)["failure_code"] == "MAIN_COUNT_FAILED_RECONCILIATION_REQUIRED"
    assert b"secret-key-and-raw-prompt" not in raw
    assert ledger.read_record(f"{keys[0].dispatch_id}.count-failure.json") == raw
    assert counts["requests"] == 0


def test_durable_parent_total_matches_count_and_generation_after_reopen(entrypoint_fixture, monkeypatch):
    import json
    from pathlib import Path

    import memcontam.readiness.phase13_main_request_dispatch as dispatch
    from .phase13_runner_safety_fixture import FakeProvider, open_run

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: 1)
    provider = FakeProvider()
    run = open_run(entrypoint_fixture, create=True)
    parent_id = run.selected.package.production[0].unit_id
    try:
        assert run.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000,
                           provider_factory=provider.factory).completed_count == 1
        raw = run.ledger.read_record(f"{parent_id}.parent.json")
        assert json.loads(raw)["unit_evidence"]["realized_cost_krw"] == run.ledger.realized_cost_krw() == 150
    finally:
        run.close()
    reopened = open_run(entrypoint_fixture, create=False)
    try:
        assert reopened.status().completed_count == 1
        assert reopened.ledger.read_record(f"{parent_id}.parent.json") == raw
        assert reopened.ledger.realized_cost_krw() == 150
        assert reopened.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000,
                                provider_factory=provider.factory).completed_count == 1
        assert len(provider.requests) == len(set(provider.requests)) == 50
        assert len(reopened.ledger.count_records("count-started")) == 50
    finally:
        reopened.close()
