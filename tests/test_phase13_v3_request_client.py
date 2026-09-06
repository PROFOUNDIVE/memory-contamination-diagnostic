from __future__ import annotations

import threading
from typing import Never

import pytest
from pydantic import JsonValue

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.runtime_registry import NOMEM_SINGLETON, RuntimeTrialResult
from memcontam.readiness.phase13_main_request_client import MainRequestClientV3
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
)
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
from memcontam.readiness.phase13_v3_terminal_models import TerminalEvidenceError

ClientFixture = tuple[MainRequestClientV3, TerminalLedgerV3, tuple[RequestKeyV3, ...], dict[str, int]]


@pytest.fixture
def client_fixture(tmp_path, monkeypatch):
    import memcontam.readiness.phase13_main_request_dispatch as dispatch

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: 1)
    binding = PackageBindingV3(package_sha256="b" * 64, authorization_sha256="c" * 64)
    keys = tuple(RequestKeyV3(parent_id="a" * 64, stage="no_memory_generate", ordinal=index) for index in range(2))
    counts = {"constructor": 0, "requests": 0}

    class Provider:
        def send_compiled_v3(self, compiled, before_request):
            assert compiled.native_state == b"immutable native bytes"
            before_request()
            counts["requests"] += 1
            return LLMResponse("not a final answer", {"usage": {"input_tokens": 1, "output_tokens": 1}}, {}, 0)

    def factory(_binding):
        counts["constructor"] += 1
        return Provider()

    with private_ledger(tmp_path / "fixture", create=True) as private:
        ledger = TerminalLedgerV3.create_guarded(private, {"schema_version": "phase13_main_run_ledger_v3",
            "unit_ids": [key.dispatch_id for key in keys], "package_sha256": binding.package_sha256,
            "authorization_sha256": binding.authorization_sha256})
        dispatcher = ProductionRequestDispatcherV3(ledger, binding,
            (ParentTrajectoryV3(parent_id="a" * 64, kind="NO_MEMORY_SINGLETON"),), provider_factory=factory)
        client = MainRequestClientV3(dispatcher, "a" * 64, private.check)
        yield client, ledger, keys, counts


def call(client):
    return client.chat([{"role": "user", "content": "fixture"}], "gpt-5.6-luna", {"method_stage": "no_memory_generate"})


def test_semantic_failure_from_actual_baseline_outcome_is_terminal(client_fixture):
    client, ledger, keys, counts = client_fixture

    def execute():
        call(client)
        assert ledger.state(keys[0].dispatch_id).kind == "ATTEMPT_STARTED"
        return RuntimeTrialResult(BaselineExecutionOutcome("failed", error_type="BaselineOutputError",
            failure_disposition="no_memory_invalid_final_answer", scientific_ineligibility_reason="invalid_final_answer"), NOMEM_SINGLETON)

    with pytest.raises(RuntimeError, match="MAIN_ATTEMPTED_PROVIDER_FAILURE"):
        client.trial(execute, lambda: b"immutable native bytes")
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
